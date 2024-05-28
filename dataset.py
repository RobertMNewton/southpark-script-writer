import pandas as pd
import torch
from torch import Tensor
from tqdm import tqdm
from typing import List, Iterator, Union
import random
import re


# constants

episode_data_path = "data/SouthPark_Episodes.csv"
line_data_path = "data/SouthPark_Lines.csv"

max_episode_size = 4000

# tokenizer code

_tokenizer = None
_reverse_tokenizer = None

def custom_tokenize(text: str) -> List[str]:
    pattern = re.compile(r'(\s+|[^\w\s])')
    tokens = pattern.split(text)
    tokens = [token for token in tokens if token]
    return tokens

def _train_tokenizer(dataset: List[str]) -> None:
    global _tokenizer, _reverse_tokenizer
    
    unique_tokens = set(custom_tokenize(" ".join(dataset).lower()))
    
    _tokenizer = {char: i for i, char in enumerate(unique_tokens)}
    
    _tokenizer["~"] = len(_tokenizer) # use this as padding
    _tokenizer["`"] = len(_tokenizer) # use this as CLS token
    
    _reverse_tokenizer = {val: key for key, val in _tokenizer.items()}
    

def text_to_ids(text: List[str]) -> Tensor:
    if not isinstance(text, list):
        text = [text]
    
    ids, masks, max_seq = [], [], 0
    for s in text:
        s = "".join([char for char in s.lower() if s.isascii()])
        
        ids.append([])
        for token in custom_tokenize(s):
            ids[-1].append(_tokenizer[token])

        masks.append([1] * len(ids[-1]))
        
        if len(ids[-1]) > max_seq:
            max_seq = len(ids[-1])
            
    for i in range(len(ids)):
        ids[i].extend([_tokenizer["~"]] * (max_seq - len(ids[i])))
        masks[i].extend([0] * (max_seq - len(masks[i])))
    
    return torch.LongTensor(ids), torch.ByteTensor(masks)

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
_scripts_tokens = None
_scripts_tokens_masks = None

def _init_scripts():
    """
    Initializes script data from csvs. Each script is formatted episode by episode, the idea being we want
    this data to be together in order to predict plausible South Park episodes that sound 'logical'.
    """
    global _scripts, _scripts_tokens, _scripts_tokens_masks
    _scripts = []
    
    lines = pd.read_csv(line_data_path)
    episodes = pd.read_csv(episode_data_path)
    
    lines.dropna(inplace=True)
    
    for episode in episodes.iloc:
        episode_script, bad_episode = [], False
        for line in lines.loc[lines["Title"] == episode["Title"]].iloc:
            if line["Line"].isascii():
                episode_script.append(f"{line['Character']}:{line['Line']}".lower())
            else:
                bad_episode = True
                break
        if not bad_episode:
            _scripts.append("\n".join(episode_script)[:max_episode_size])
    
    _train_tokenizer(_scripts)
    
    _scripts_tokens, _scripts_tokens_masks = text_to_ids(_scripts)


def get_scripts_tokens(batch_size, sequence_size) -> Iterator[Tensor]:
    """
    Returns iterator that yields tokenised Southpark scripts.
    """
    if _scripts_tokens is None:
        _init_scripts()
        
    shuffle_indices = torch.randperm(_scripts_tokens.shape[0])
    
    n_batches = _scripts_tokens.shape[0] // batch_size
    n_sequences = _scripts_tokens.shape[1] // sequence_size
    
    for batch_num in range(n_batches):
        for sequence_num in range(n_sequences):
            yield (
                _scripts_tokens[shuffle_indices[batch_num*batch_size:(batch_num + 1)*batch_size], sequence_num*sequence_size:(sequence_num+1)*sequence_size],
                _scripts_tokens_masks[shuffle_indices[batch_num*batch_size:(batch_num + 1)*batch_size], sequence_num*sequence_size:(sequence_num+1)*sequence_size],
                sequence_num == n_sequences - 1,
            )
        

# embedding functions

def encode_token_ids(tokens: Union[List[int], Tensor]) -> Tensor:
    if isinstance(tokens, list):
        tokens = Tensor(tokens).to(torch.long)
    
    embedding = torch.zeros((tokens.shape[0], tokens.shape[1], get_vocab_size()), dtype=torch.float32).to(tokens.device)
    
    embedding.scatter_(2, tokens.unsqueeze(-1), 1.0)
    
    return embedding
    
        