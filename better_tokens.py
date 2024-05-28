import json
import os
from torch import LongTensor
import torch
from typing import List, Dict, Optional, Union
import pandas as pd
from tqdm import tqdm


class BreakLoopTo(Exception):
    pass


TokeniserData = Dict[Union[int, str], Dict[str, int]]

def train_tokeniser_from_text(text: str, max_iterations: int  = 2000, extra: List[str] = ["<EOS>", "<PAD>", "<CLS>", "<UNK>", "<MASK>", "<BOS>", "<SEP>"]) -> TokeniserData:    
    tokeniser, max_token_length = {1: list(set(text))}, 1
    
    pg = tqdm(range(max_iterations), dynamic_ncols=True)
    for _ in pg:
        pg.set_description_str(f"Training Tokeniser (N = {sum(len(tokens) for tokens in tokeniser.values())})")
        pg.set_postfix_str(s = "Tokenising...")
        
        # tokenise text using greedy tokenisation
        tokenised_text, i = [], 0
        while i < len(text):
            #print(i, " < ", len(text))
            try:
                for token_length in range(max_token_length):
                    token_length = max_token_length - token_length
                    if len(text) - i < token_length or token_length not in tokeniser:
                        continue
                    
                    for token in tokeniser[token_length]:
                        if text[i:i+token_length] == token:
                            i += token_length
                            tokenised_text.append(token)
                            
                            raise BreakLoopTo()
            except BreakLoopTo:
                continue
            
            # if we got here then no token pair fits the text. We should thus add the next char to the token
            tokenised_text.append(text[i])
            tokeniser[1].append(text[i])
            
            i += 1
        
        pg.set_postfix_str(s="Counting frequencies...")
        
        # determine the frequency each token pair occurs    
        pair_freqs: Dict[str, int] = {}
        for i in range(len(tokenised_text) - 1):
            pair = (tokenised_text[i], tokenised_text[i+1])
            if pair in pair_freqs:
                pair_freqs[pair] += 1
            else:
                pair_freqs[pair] = 1
        
        pg.set_postfix_str(s="Merging...")
        
        # Merge most common token pair! And repeat
        max_freq = max(pair_freqs.values())
        for k, v in pair_freqs.items():
            if v == max_freq:
                new_token = "".join(k)
                if not len(new_token) in tokeniser:
                    tokeniser[len(new_token)] = []
                    if len(new_token) > max_token_length:
                        max_token_length = len(new_token)
                
                
                tokeniser[len(k[0])].remove(k[0])
                if k[0] != k[1]:    
                    tokeniser[len(k[1])].remove(k[1])
                
                tokeniser[len(new_token)].append(new_token)
                
                if len(tokeniser[len(k[0])]) == 0:
                    tokeniser.pop(len(k[0]))
                if len(tokeniser[len(k[1])]) == 0:
                    tokeniser.pop(len(k[1]))
                
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
        
    tokeniser["vocab_size"] = current_id + len(extra)
    tokeniser["detokeniser"] = detokeniser
    
    return tokeniser


def save_tokeniser(filepath: str, tokeniser: dict) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(tokeniser, f)
        
        
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
save_tokeniser("./southpark_tokens_all.json", tokeniser)
        
        
    