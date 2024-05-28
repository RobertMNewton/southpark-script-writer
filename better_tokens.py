import json
import os
from torch import LongTensor
import torch
from typing import List, Dict, Optional, Union
import pandas as pd


class BreakLoopTo(Exception):
    pass


TokeniserData = Dict[Union[int, str], Dict[str, int]]

def train_tokeniser_from_text(text: str, max_iterations: int  = 1000, extra: List[str] = ["<EOS>", "<PAD>", "<CLS>", "<UNK>", "<MASK>", "<BOS>", "<SEP>"]) -> TokeniserData:    
    tokeniser, max_token_length = {1: list(set(text))}, 1
    for _ in range(max_iterations):
        # tokenise text using greedy tokenisation
        tokenised_text, i = [], 0
        while i < len(text):
            try:
                for token_length in range(max_token_length):
                    token_length = max_token_length - token_length
                    if len(text) - i < token_length:
                        continue
                    
                    for token in tokeniser[token_length]:
                        if text[i:i+token_length] == token:
                            i += token_length
                            tokenised_text.append(token)
                            
                            raise BreakLoopTo()
            except BreakLoopTo:
                continue
        
        # determine the frequency each token pair occurs    
        pair_freqs: Dict[str, int] = {}
        for i in range(len(tokenised_text) - 1):
            pair = (tokenised_text[i], tokenised_text[i+1])
            if pair in pair_freqs:
                pair_freqs[pair] += 1
            else:
                pair_freqs[pair] = 1
        
        # Merge most common token pair! And repeat 
        max_freq = max(pair_freqs.values())
        for k, v in pair_freqs.items():
            if v == max_freq:
                if not len(k) in tokeniser:
                    tokeniser[len(k)] = []
                    if len(k) > max_token_length:
                        max_token_length = len(k)
                
                tokeniser[len(k[0])].remove(k[0])
                tokeniser[len(k[1])].remove(k[1])
                
                if len(tokeniser[len(k[0])]) == 0:
                    tokeniser.pop(len(k[0]))
                if len(tokeniser[len(k[1])]) == 0:
                    tokeniser.pop(len(k[1]))

                new_token = "".join(k)
                
                tokeniser[len(new_token)].append(new_token)
                
                break
                
    tokeniser["extra"] = extra
    tokeniser["token_lengths"] = sorted([k for k in tokeniser.keys() if isinstance(k, int)], reverse=True)
    
    current_id, detokeniser = 0, {}
    for k, v in tokeniser.items():
        if k == "token_lengths":
            continue
            
        tokeniser[k] = {token: id for token, id in enumerate(tokeniser[k], current_id)}
        detokeniser.update({id: token for token, id in tokeniser[k].items()})
        
        current_id += len(tokeniser[k])
        
    tokeniser["vocab_size"] = current_id
    tokeniser["detokeniser"] = detokeniser
    
    return tokeniser


def save_tokeniser(filepath: str, tokeniser: dict) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.jump(tokeniser, f)
        
        
def load_tokeniser(filepath: str, tokeniser: dict) -> None:
    with open(filepath, "w") as f:
        json.jump(tokeniser, f)
        

def tokenise_text(text: Union[str, List[str]], tokeniser: TokeniserData, as_tensor: bool = True) -> Union[Union[List[int], List[List[int]]], LongTensor]:
    if isinstance(text, str):
        text = [text]
    
    for sequence in text:
        tokenised_sequence, i = [], 0
    
    pass

# test train_tokeniser real quick

episode_data_path = "data/SouthPark_Episodes.csv"
line_data_path = "data/SouthPark_Lines.csv"

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
        _scripts.append("\n".join(episode_script))
        
tokeniser = train_tokeniser_from_text("\n".join(_scripts))
save_tokeniser("southpark_tokens.json", tokeniser)
        
        
    