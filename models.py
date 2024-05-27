import torch
from torch import optim, nn, Tensor
from fastkan import FastKAN as KAN
from typing import List, Optional, Tuple


def get_mlp(input_dim: int, output_dim: int, width: int, depth: int, activation: nn.Module = nn.ReLU) -> nn.Module:
    model = []
    
    model.append(nn.Linear(input_dim, width))
    
    for _ in range(depth):
        model.append(activation())
        model.append(nn.Linear(width, width))
    
    model.append(activation())
    model.append(nn.Linear(width, output_dim))
    
    return nn.Sequential(*model)


def get_kan(input_dim: int, output_dim: int, width: int, depth: int) -> nn.Module:
    return nn.Sequential(
        KAN([input_dim] + [width for _ in range(depth)] + [output_dim]),
    )


class RNNLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, width: int, depth: int, architecture: str = "MLP"):
        """
        Make an RNN Layer. Can specify architecture as either 'MLP' or 'KAN'
        """
        super().__init__()
        
        if architecture == "MLP":
            self.model = get_mlp(
                input_dim + hidden_dim,
                output_dim + hidden_dim,
                width,
                depth,
            )
        elif architecture == "KAN":
            self.model = get_kan(
                input_dim + hidden_dim,
                output_dim + hidden_dim,
                width,
                depth,
            )
            
        self.hidden = nn.Parameter(torch.rand((1, hidden_dim)))
        
        self._hidden_dim = hidden_dim
        self._output_dim = output_dim
        
    def forward(self, x: Tensor, hidden: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Performs forward pass. Only works on unbatched inputs. Assumes shape is (sequence length, input dim).
        Output is (hidden state, output)
        """
        if hidden is None:
            hidden = self.hidden
        
        output = torch.zeros((x.shape[0], self._hidden_dim + self._output_dim))
        for i in range(x.shape[0]):
            features = torch.cat((x[i].view(1, -1), hidden.view(1, -1)), -1)
            
            output[i] = self.model(features)
            
            hidden = output[i, :self._hidden_dim].unsqueeze(0)
        
        return hidden, output[:, self._hidden_dim:] 
    

class SimpleLM(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, embedding_dim: int, depth: int, rnn_layers: int, architecture: str = "MLP"):
        super().__init__()
        
        self.embedder = nn.Sequential(nn.Embedding(vocab_size, embedding_dim), nn.Softmax(-1))
        self.cls = None
        
        self.post_process = None
        if architecture == "MLP":
            self.cls = nn.Linear(hidden_dim, vocab_size)
        else:
            self.cls = get_kan(
                hidden_dim,
                vocab_size,
                embedding_dim,
                depth,
            )
            
        self.rnn = nn.RNN(
            embedding_dim,
            hidden_dim,
            rnn_layers,
            batch_first=True,
        )
        
        self.hidden_size = hidden_dim
        self.vocab_size = vocab_size
        
        
        
    def forward(self, x: Tensor, hidden: Optional[List[Tensor]] = None) -> Tuple[List[Tensor], Tensor]:
        """
        Performs forward pass. Returns output hidden state and output embeddings in that order.
        """
        x, h = self.rnn(self.embedder(x), hidden) if hidden is not None else self.rnn(self.embedder(x))
        return h, self.cls(x.reshape(-1, self.hidden_size)).reshape(x.shape[0], x.shape[1], self.vocab_size)
        

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
    

def save_model(model: nn.Module, path: str) -> None:
    torch.save(model.state_dict(), path)
    
def load_model(model: nn.Module, path: str) -> nn.Module:
    model.load_state_dict(torch.load(path))
    return model