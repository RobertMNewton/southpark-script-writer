import os
import torch
import dataset
import better_tokens
import models
import json
from torch import optim, nn, Tensor
from typing import Dict, Optional
import random


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

dataset._init_scripts()

#mean_sequence_length = sum([len(better_tokens.text_to_ids(episode, dataset.tokeniser)) for episode in dataset._scripts]) / len(dataset._scripts)
#print(f"Mean Sequence Length: {mean_sequence_length}")

test_prompt = ""
model_name = "rnn-small-1024-300-3"

def get_MLP_LM() -> models.SimpleLM:
    return models.SimpleLM(
        dataset.get_vocab_size(),
        1024,
        300,
        2,
        3,
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
    input_ids = better_tokens.text_to_ids(prompt, dataset.tokeniser, add_eos_token=False)[0].unsqueeze(0).to(device)

    # Initialize hidden states as None
    hidden = None

    # List to store the generated token IDs
    generated_ids = input_ids.squeeze(0).tolist()

    # Loop until the max length is reached or the <EOS> token is generated
    for _ in range(max_length):
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
    generated_text = better_tokens.ids_to_text(generated_ids, dataset.tokeniser)

    return generated_text

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
    
    epochs = 101

    # First, let's train the MLP model
    mlp_model = get_MLP_LM().to(device)
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(mlp_model.parameters(), lr=7E-5)    
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimiser,
        mode='min',
        factor=0.5,
        patience=5,
        threshold=0.001,
        threshold_mode='rel',
        cooldown=0,
        min_lr=0,
        eps=1e-08,
        )
    
    print(f"Model: {mlp_model.get_num_parameters()} params")
    
    batch_sizes, sequence_sizes = [148, 120, 96, 72, 48, 24], [8, 16, 32, 64, 128, 256]
    batch_size, sequence_size = None, None
    
    for epoch in range(epochs):
        if epoch % 10 == 0:
            test_script = predict_sequence(mlp_model, test_prompt, max_length=500)
            os.makedirs(os.path.dirname(f"./models/{model_name}/test_sequences/e{epoch}.txt"), exist_ok=True)
            with open(f"models/{model_name}/test_sequences/e{epoch}.txt", "w") as f:
                f.write(str(test_script))
            
            os.makedirs(os.path.dirname(f"./models/{model_name}/checkpoints/e{epoch}.pt"), exist_ok=True)
            models.save_model(mlp_model, f"models/{model_name}/checkpoints/e{epoch}.pt")
            
            if epoch < 60:
                batch_size, sequence_size = batch_sizes[epoch // 10], sequence_sizes[epoch // 10]
        
        running_loss, running_acc = 0, 0
        
        script_tokens = iter(dataset.get_scripts_tokens(batch_size, sequence_size))
        hidden: Optional[Tensor] = None
        
        for b, batch_info in enumerate(script_tokens):
            batch, batch_masks = batch_info[0].to(device), batch_info[1].to(device)
            
            if isinstance(hidden, Tensor):
                hidden = hidden.data
            
            optimiser.zero_grad()
            
            hidden, pred = mlp_model(batch, hidden)
            
            pred, batch = pred[:, :-1][batch_masks[:, :-1]], batch[:, 1:][batch_masks[:, :-1]]
            
            try:
                loss = criterion(pred, batch)
                
                if torch.isnan(loss):
                    continue
                
                loss.backward()
                
                optimiser.step()
            except Exception as e:
                print(f'Warning: {e}')
            
            accuracy = (torch.argmax(pred, -1) == batch).sum().float() / batch.numel() * 100
            
            running_acc += accuracy.item()
            running_loss += loss.item()
            
            print(f"E: {epoch}/{epochs}, s: {b}, L: {loss.item():.2f}, A: {accuracy.item():.1f}%, Lavg: {running_loss / (b + 1):.2f}, Aavg: {running_acc / (b + 1):.2f}%, lr: {scheduler.get_last_lr()[0]:.2E}")
            
            reset_hidden = batch_info[2]
            if reset_hidden:
                hidden = None
        
        scheduler.step(running_loss / (b + 1))
            
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
        
        batch_size, sequence_size = 16, 30
        
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