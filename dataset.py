import pandas as pd
import torch
from torch import Tensor
from tqdm import tqdm
from typing import List, Iterator, Union


# constants

episode_data_path = "data/SouthPark_Episodes.csv"
line_data_path = "data/SouthPark_Lines.csv"

truncate_to = 1024

# tokenizer code

_tokenizer = None
_reverse_tokenizer = None

def _train_tokenizer(dataset: List[str]) -> None:
    global _tokenizer, _reverse_tokenizer
    
    _tokenizer = {char: i for i, char in enumerate(set("".join(dataset)))}
    
    _tokenizer["\n"] = len(_tokenizer) # use this as padding
    _tokenizer["~"] = len(_tokenizer) # use this as padding
    _tokenizer["`"] = len(_tokenizer) # use this as EOS token
    
    _reverse_tokenizer = {val: key for key, val in _tokenizer.items()}      

def text_to_ids(text: List[str]) -> List[List[int]]:
    if not isinstance(text, list):
        text = [text]
    
    res = []
    for s in text:
        res.append([])
        for char in s:
            res[-1].append(_tokenizer[char])
    
    return Tensor(res).to(torch.int64)

def ids_to_text(ids: List[List[int]]) -> List[str]:
    res = []
    for sentence in ids:
        res.append([])
        for id in sentence:
            res[-1].append(_reverse_tokenizer[id])
        res[-1] = "".join(res[-1])
        
    return res

def get_vocab_size() -> int:
    return len(_tokenizer) + 1

# dataset code

_scripts = None

def _init_scripts():
    """
    Initializes script data from csvs. Each script is formatted episode by episode, the idea being we want
    this data to be together in order to predict plausible South Park episodes that sound 'logical'.
    """
    global _scripts
    _scripts = []
    
    lines = pd.read_csv(line_data_path)
    
    for line in lines.iloc:
        new_line = f"{line['Character']}: {line['Line']}"
        if len(_scripts) == 0:
            _scripts.append(new_line[:truncate_to - 1])
        elif len(_scripts[-1]) + len(new_line) < truncate_to - 1:
            _scripts[-1] += new_line
        else:
            _scripts[-1] += "`" + "~"*(truncate_to - len(_scripts[-1]) - 1)
            _scripts.append(new_line[:truncate_to - 1])
    
    _train_tokenizer(_scripts)


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
        yield text_to_ids(script)
        

# embedding functions

def embed_token_ids(tokens: Union[List[int], Tensor]) -> Tensor:
    if isinstance(tokens, list):
        tokens = Tensor(tokens).to(torch.int64)
    
    embedding = torch.zeros((tokens.shape[0], tokens.shape[1], get_vocab_size()), dtype=torch.float32)
    
    embedding.scatter_(2, tokens.unsqueeze(-1), 1.0)
    
    return embedding
    
        