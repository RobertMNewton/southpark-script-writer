import torch
import dataset
import models
import json
from torch import optim, nn
from typing import Dict

test_prompt = f"at the bus st"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset._init_scripts()

def get_MLP_LM() -> models.SimpleLM:
    return models.SimpleLM(
        dataset.get_vocab_size(),
        64,
        256,
        2,
        6,
        architecture="MLP",
    )

def get_KAN_LM() -> models.SimpleLM:
    return models.SimpleLM(
        dataset.get_vocab_size(),
        16,
        64,
        2,
        6,
        architecture="KAN",
    )
    
def predict_sequence(model: nn.Module, prompt: str, max_length: int = 50) -> str:
    # Tokenize the input prompt
    input_ids = dataset.text_to_ids(prompt).to(device)

    # Initialize hidden states as None
    hidden = None

    # List to store the generated token IDs
    generated_ids = input_ids.squeeze(0).tolist()

    # Loop until the max length is reached or the <EOS> token is generated
    while len(generated_ids) < max_length:
        # Embed the token IDs
        #embedding = dataset.encode_token_ids(input_ids).to(device)

        # Forward pass through the model
        hidden, pred = model(input_ids, hidden)

        # Get the ID of the predicted token (choose the token with the highest probability)
        pred_id = torch.argmax(pred[:, -1], dim=-1).item()

        # Append the predicted token ID to the generated_ids list
        generated_ids.append(pred_id)

        # Check if the <EOS> token is generated
        if pred_id == "`":
            break

        # Update the input_ids for the next iteration
        input_ids = torch.tensor([[pred_id]], device=device)

    # Convert the generated token IDs back to text
    generated_text = dataset.ids_to_text([generated_ids])

    return generated_text[0]

def predict_probs(model: nn.Module, prompt: str) -> Dict[str, float]:
    prompt_ids = dataset.text_to_ids(prompt)
    #prompt_encodings = dataset.encode_token_ids(prompt_ids)
    
    _, output = model(prompt_ids)
    
    # output formatting should be (B, L, N), we can then construct our dictionary from this.
    res = {}
    for i in range(output.shape[-1]):
        try:
            res[dataset._reverse_tokenizer[i]] = output[0, -1, i].item()
        except: 
            pass
        
    return res

if __name__ == "__main__":
    
    epochs = 50

    # First, let's train the MLP model
    mlp_model = get_MLP_LM().to(device).to(torch.float32)
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(mlp_model.parameters(), lr=3E-6)
    
    print(f"Model: {mlp_model.get_num_parameters()} params")
    
    for epoch in range(epochs):
        test_script = predict_sequence(mlp_model, test_prompt, max_length=25)
        test_probs = predict_probs(mlp_model, test_prompt)
        with open(f"models/demos/mlp_model_{epoch}.txt", "w") as f:
            f.write(str(test_script))
        
        json.dump(test_probs, open(f"models/demos/mlp_model_{epoch}.json", "w"))
        
        scripts = iter(dataset.get_scripts_tokens(as_tensors=True))
        
        batch_size = 128
        n_batches = len(dataset._scripts) // batch_size
        
        for b in range(n_batches):
            batch = torch.cat([next(scripts) for _ in range(batch_size)], 0)
            batch = batch.to(device)
            
            optimiser.zero_grad()
            
            #embedding = dataset.encode_token_ids(batch).to(device).to(torch.float32)
            
            _, pred = mlp_model(batch)
            
            loss = criterion(pred[:, :-1], pred[:, 1:])
            
            loss.backward()
            
            optimiser.step()
            
            print(f"MLP MODEL Epoch: {epoch}/{epochs}, batch: {b}/{n_batches}, Loss: {loss.item():.3f}")
            
    models.save_model(mlp_model, "models/mlp_lm.pt")

    # Second, let's train the KAN model
    kan_model = get_KAN_LM().to(device).to(torch.float32)
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(kan_model.parameters(), lr=3E-2)
    
    for epoch in range(epochs):
        test_script = predict_sequence(kan_model, test_prompt, max_length=5000)
        with open(f"models/demos/kan_model_{epoch}.txt", "w") as f:
            f.write(test_script)
            
        scripts = dataset.get_scripts_tokens(as_tensors=True)
        for step, script_tokens in enumerate(scripts):
            script_tokens = script_tokens.to(device)
            optimiser.zero_grad()
            
            encoding = dataset.encode_token_ids(script_tokens)[:-1].to(device).to(torch.float32)
            
            _, pred = kan_model(encoding)
            target = script_tokens[0, 1:].to(device)
            
            loss = criterion(pred, target)
            loss.backward()
            
            optimiser.step()
            
            print(f"KAN MODEL... Epoch: {epoch}, Step: {step}, Loss: {loss.item():.3f}")
            
    models.save_model(kan_model, "models/kan_lm.pt")