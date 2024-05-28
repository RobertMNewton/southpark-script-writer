import torch
import dataset
import models
import json
from torch import optim, nn, Tensor
from typing import Dict, Optional

test_prompt = "Kyle:Yeah, they're almost as big as his mom's.\nScene Description:The others laugh.\nCartman:"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset._init_scripts()

def get_MLP_LM() -> models.SimpleLM:
    return models.SimpleLM(
        dataset.get_vocab_size(),
        256,
        1024,
        2,
        2,
        architecture="MLP"
    )
    
def get_transformer_LM() -> models.SimpleTransformerLM:
    return models.SimpleTransformerLM(
        dataset.get_vocab_size(),
        1024,
        256,
        2,
        2,
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
    
def predict_sequence(model: nn.Module, prompt: str, max_length: int = 100, type: str = "rnn") -> str:
    # Tokenize the input prompt
    input_ids, _ = dataset.text_to_ids(prompt)
    cls_token, _ = dataset.text_to_ids("`")

    # Initialize hidden states as None
    hidden = None

    # List to store the generated token IDs
    generated_ids = input_ids.squeeze(0).tolist()

    # Loop until the max length is reached or the <EOS> token is generated
    while len(generated_ids) < max_length:
        # Embed the token IDs
        #embedding = dataset.encode_token_ids(input_ids).to(device)

        # Forward pass through the model
        pred = None
        if type == "rnn":
            hidden, pred = model(input_ids, hidden)
        else:
            pred = model(input_ids)

        # Get the ID of the predicted token (choose the token with the highest probability)
        pred_id = torch.argmax(pred[:, -1], dim=-1)

        # Append the predicted token ID to the generated_ids list
        generated_ids.append(pred_id.item())

        # Update the input_ids for the next iteration
        input_ids = torch.cat((input_ids, pred_id.unsqueeze(0)), 1)

    # Convert the generated token IDs back to text
    generated_text = dataset.ids_to_text([generated_ids])

    return generated_text[0]

def predict_probs(model: nn.Module, prompt: str, type: str = "rnn") -> Dict[str, float]:
    prompt_ids, _ = dataset.text_to_ids(prompt)
    #prompt_encodings = dataset.encode_token_ids(prompt_ids)
    
    output = None
    if type == "rnn":
        _, output = model(prompt_ids)
    else:
        output = model(prompt_ids)
    
    # output formatting should be (B, L, N), we can then construct our dictionary from this.
    res = {}
    for i in range(output.shape[-1]):
        try:
            res[dataset._reverse_tokenizer[i]] = output[0, -1, i].item()
        except: 
            pass
        
    return res

if __name__ == "__main__":
    
    epochs = 300

    # First, let's train the MLP model
    mlp_model = get_MLP_LM().to(device)
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(mlp_model.parameters(), lr=6E-5)
    
    json.dump(dataset._tokenizer, open("tokeniser.json", "w"))    
    
    print(f"Model: {mlp_model.get_num_parameters()} params")
    
    
    for epoch in range(epochs):
        test_script = predict_sequence(mlp_model, test_prompt, max_length=500)
        test_probs = predict_probs(mlp_model, test_prompt)
        with open(f"models/rnn/mlp_model_{epoch}.txt", "w") as f:
            f.write(str(test_script))
        
        json.dump(test_probs, open(f"models/rnn/{epoch}.json", "w"))
        
        batch_size, sequence_size = 16, 8
        
        running_loss, running_acc = 0, 0
        
        script_tokens = iter(dataset.get_scripts_tokens(batch_size, sequence_size))
        hidden: Optional[Tensor] = None
        
        for b, batch_info in enumerate(script_tokens):
            batch, mask = batch_info[0].to(device), batch_info[1].to(device)
            
            reset_hidden = batch_info[2]
            if reset_hidden:
                hidden = None
            elif isinstance(hidden, Tensor):
                hidden = hidden.data
            
            optimiser.zero_grad()
            
            hidden, pred = mlp_model(batch, hidden)
            
            #print(pred.shape, batch.shape, mask.shape)
            
            pred, batch = pred[:, :-1][mask[:, :-1]], batch[:, 1:][mask[:, :-1]]
            
            loss = criterion(pred, batch)
            
            if torch.isnan(loss):
                continue
            
            loss.backward()
            
            optimiser.step()
            
            accuracy = torch.sum(torch.argmax(pred, -1) == batch).float().mean()
            
            running_acc += accuracy.item()
            running_loss += loss.item()
            
            print(f"RNN MODEL Epoch: {epoch}/{epochs}, step: {b}, Loss: {loss.item():.2f}, Accuracy: {accuracy.item():.1f}%, Mean Loss: {running_loss / (b + 1):.2f}, Running Accuracy: {running_acc / (b + 1):.2f}%")
            
    models.save_model(mlp_model, "models/rnn_lm.pt")

    # Second, let's train the transformer model
    mlp_model = get_transformer_LM().to(device)
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(mlp_model.parameters(), lr=1E-4)
    
    print(f"Model: {mlp_model.get_num_parameters()} params")
    
    for epoch in range(epochs):
        test_script = predict_sequence(mlp_model, test_prompt, max_length=100, type="transformer")
        test_probs = predict_probs(mlp_model, test_prompt, type="transformer")
        with open(f"models/transformer/{epoch}.txt", "w") as f:
            f.write(str(test_script))
        
        json.dump(test_probs, open(f"models/transformer/{epoch}.json", "w"))
        
        batch_size, sequence_size = 16, 100
        
        script_tokens = iter(dataset.get_scripts_tokens(batch_size, sequence_size))
        cls_tokens = dataset.text_to_ids(["`"]*batch_size)
        
        for b, batch_info in enumerate(script_tokens):
            batch, mask = batch_info[0].to(device), batch_info[1].to(device)
            
            optimiser.zero_grad()
            
            labels = batch
            
            #features = torch.cat((batch[:, :-1], cls_tokens), 2)
            pred = mlp_model(batch[:, :-1])
            
            #print(pred.shape, batch.shape, mask.shape)
            
            pred, batch = pred[mask[:, :-1]], batch[:, 1:][mask[:, :-1]]
            
            loss = criterion(pred, batch)
            
            loss.backward()
            
            optimiser.step()
            
            accuracy = torch.sum(torch.argmax(pred, -1) == batch).float().mean()
            
            print(f"Transformer MODEL Epoch: {epoch}/{epochs}, step: {b}, Loss: {loss.item():.2f}, Accuracy: {accuracy.item():.1f}%")
        
        models.save_model(mlp_model, "models/transformer_LM.pt")