import torch
import dataset
import models
from torch import optim, nn

test_prompt = f"Cartman: "
device = torch.device("cpu")

dataset._init_scripts()

def get_MLP_LM() -> models.SimpleLM:
    return models.SimpleLM(
        dataset.get_vocab_size(),
        16,
        16,
        16,
        2,
        2,
        architecture="MLP",
    )

def get_KAN_LM() -> models.SimpleLM:
    return models.SimpleLM(
        dataset.get_vocab_size(),
        16,
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
        embedding = dataset.embed_token_ids(input_ids).to(device)

        # Forward pass through the model
        hidden, pred = model(embedding[0], hidden)

        # Get the ID of the predicted token (choose the token with the highest probability)
        pred_id = torch.argmax(pred[-1], dim=-1).item()

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

if __name__ == "__main__":
    
    epochs = 10

    # First, let's train the MLP model
    mlp_model = get_MLP_LM().to(device).to(torch.float32)
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(mlp_model.parameters(), lr=3E-2)
    
    for epoch in range(epochs):
        test_script = predict_sequence(mlp_model, test_prompt, max_length=1024)
        with open(f"models/demos/mlp_model_{epoch}.txt", "w") as f:
            f.write(test_script)
        
        scripts = dataset.get_scripts_tokens(as_tensors=True)
        for step, script_tokens in enumerate(scripts):
            script_tokens = script_tokens.to(device)
            optimiser.zero_grad()
            
            embedding = dataset.embed_token_ids(script_tokens).to(device).to(torch.float32)
            
            _, pred = mlp_model(embedding[0])
            target = script_tokens[0, :-1].to(device)
            
            loss = criterion(pred[1:], target)
            loss.backward()
            
            optimiser.step()
            
            print(f"MLP MODEL... Epoch: {epoch}, Step: {step}, Loss: {loss.item():.3f}")
            
    models.save_model(mlp_model, "models/mlp_lm.pt")

    # Second, let's train the KAN model
    kan_model = get_KAN_LM().to(device).to(torch.float32)
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(kan_model.parameters(), lr=3E-4)
    
    for epoch in range(epochs):
        test_script = predict_sequence(kan_model, test_prompt, max_length=5000)
        with open(f"models/demos/kan_model_{epoch}.txt", "w") as f:
            f.write(test_script)
            
        scripts = dataset.get_scripts_tokens(as_tensors=True)
        for step, script_tokens in enumerate(scripts):
            script_tokens = script_tokens.to(device)
            optimiser.zero_grad()
            
            embedding = dataset.embed_token_ids(script_tokens)[:-1].to(device).to(torch.float32)
            
            _, pred = kan_model(embedding)
            target = script_tokens[0, 1:].to(device)
            
            loss = criterion(pred, target)
            loss.backward()
            
            optimiser.step()
            
            print(f"KAN MODEL... Epoch: {epoch}, Step: {step}, Loss: {loss.item():.3f}")
            
    models.save_model(kan_model, "models/kan_lm.pt")