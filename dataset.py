import pandas as pd
import torch
from torch import Tensor
from tqdm import tqdm
from transformers import AutoTokenizer
from typing import List, Iterator, Union


# constants

episode_data_path = "data/SouthPark_Episodes.csv"
line_data_path = "data/SouthPark_Lines.csv"

# tokenizer code

_tokenizer = None

def _init_tokenizer():
    global _tokenizer
    
    _tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    _tokenizer.add_special_tokens({"eos_token": "<EOS>"})

def text_to_ids(text: str, as_tensors:  bool = True) -> List[int]:
    if _tokenizer is None:
        _init_tokenizer()
    
    if as_tensors:
        return _tokenizer(text, return_tensors="pt")["input_ids"]
    else:
        return _tokenizer(text)["input_ids"]

def ids_to_text(ids: List[int]) -> str:
    if _tokenizer is None:
        _init_tokenizer()
    
    return _tokenizer.decode(ids)

def get_vocab_size() -> int:
    if _tokenizer is None:
        _init_tokenizer()
        
    return _tokenizer.vocab_size + 1


# dataset code

_scripts = None

def _init_scripts():
    """
    Initializes script data from csvs. Each script is formatted episode by episode, the idea being we want
    this data to be together in order to predict plausible South Park episodes that sound 'logical'.
    """
    global _scripts
    _scripts = []
    
    episodes = pd.read_csv(episode_data_path)
    lines = pd.read_csv(line_data_path)
    
    for episode in tqdm(episodes.iloc, desc="Loading Scripts"):
        script = [f"title: {episode['Title']}, description: {episode['Description']}"]
        
        for line in lines.loc[lines["Title"] == episode["Title"]].iloc:
            script.append(f"{line['Character']}: {line['Line']}")
            
        script[-1] += "<EOS>"
        _scripts.append("\n".join(script))
        


def get_scripts() -> Iterator[str]:
    """
    Returns iterator that yield Southpark scripts.
    """
    global _scripts
    if _scripts is None:
        _init_scripts()
        
    for script in _scripts:
        yield script


def get_scripts_tokens(as_tensors: bool = False) -> Iterator[Union[List[int], Tensor]]:
    """
    Returns iterator that yields tokenised Southpark scripts.
    """
    scripts = iter(get_scripts())
    
    for script in scripts:
        yield text_to_ids(script, as_tensors=as_tensors)
        

# embedding functions

def embed_token_ids(tokens: Union[List[int], Tensor]) -> Tensor:
    if isinstance(tokens, list):
        tokens = Tensor(tokens).to(torch.int64).reshape(1, -1)
    
    embedding = torch.zeros((tokens.shape[1], get_vocab_size()), dtype=torch.float32)
    embedding.scatter_(1, tokens, 1.0)
    
    return embedding
    
        