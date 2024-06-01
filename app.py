from models import SimpleLM, load_model
import better_tokens
import dataset
import torch


def get_MLP_LM(vocab_size: int) -> SimpleLM:
    return SimpleLM(
        vocab_size,
        512,
        300,
        2,
        2,
        architecture="MLP"
    )


if __name__ == "__main__":
    tokeniser = better_tokens.load_tokeniser("South_Park_Tokeniser_2.json")
    
    model = get_MLP_LM(tokeniser["vocab_size"])
    model = load_model(model, "models/rnn-small-256-100-2/checkpoints/e90.pt")
    
    title = input("Enter a title for the episode: ").lower()
    description = input("Enter a description for the episode: ").lower()
    
    prompt = f"[title: {title}, description: {description}]\n\n"
    input_ids = better_tokens.text_to_ids(prompt, tokeniser, add_eos_token=False)[0].unsqueeze(0)

    # Initialize hidden states as None
    hidden = None
    
    print("\n")

    # Loop until the max length is reached or the <EOS> token is generated
    while True:
        for _ in range(1000):
            # Forward pass through the model
            hidden, pred = model(input_ids, hidden)

            # Get the ID of the predicted token (choose the token with the highest probability)
            pred_id = torch.argmax(pred[:, -1], dim=-1)

            # Update the input_ids for the next iteration
            input_ids = pred_id.unsqueeze(0)
            
            print(better_tokens.ids_to_text(pred_id, tokeniser), end="")
        
        input("")
        
        