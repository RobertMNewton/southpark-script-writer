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

def train_tokeniser_from_text(text: str, max_iterations: int  = 1000, extra: List[str] = ["<EOS>", "<PAD>", "<CLS>", "<UNK>", "<MASK>", "<BOS>", "<SEP>"], tokeniser: Optional[TokeniserData] = None) -> TokeniserData:    
    tokeniser, max_token_length = {1: list(set(text))} if tokeniser is None else tokeniser, 1
    
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
    tokeniser = None
    with open(filepath, "r") as f:
        tokeniser = json.load(tokeniser, f)
    
    # need to correct strings to integer ids inside json decodings
    for k, v in tokeniser:
        if k.isnumeric():
            tokeniser[int(k)] = {int(k1): v1 for k1, v1 in v.items()}
            del tokeniser[k]
    
    return tokeniser
        

def text_to_ids(text: Union[str, List[str]], tokeniser: TokeniserData, as_tensor: bool = True, truncate_to: Optional[int] = None, pad_to: Optional[int] = None) -> Union[Union[List[int], List[List[int]]], LongTensor]:
    """
    Converts text to ids. If the text is batched and as_tensor = True and truncate_to = None then will automatically pad to maximum sequence length.
    """
    if isinstance(text, str):
        text = [text]
    
    res, max_seq_len = [], 0
    for sequence in text:
        tokenised_sequence, i = [], 0
        
        while i < len(sequence):
            try:
                # should check for special tokens first
                for id, token in tokeniser["extra"].items():
                    if len(sequence) - i < len(token):
                        continue
                    elif tokenised_sequence[i:i+len(token)] == token:
                        tokenised_sequence.append(id)
                        i += len(token)
                        
                        raise BreakLoopTo()
                    
                # should then check for other tokens in order of largest tokens to smallest tokens
                for token_length in tokeniser["token_lengths"]:
                    if len(sequence) - i < token_length:
                        continue
                    
                    token_id = tokeniser["detokeniser"].get(sequence[i:i+token_length])
                    if token_id is not None:
                        tokenised_sequence.append(token_id)
                        i += token_length
                        
                        raise BreakLoopTo()
                
                # if we are here then there is no match. We should use "<UNK>" token to mark an unknown token.
                i += 1
                tokenised_sequence += tokeniser["detokeniser"]["<UNK>"]
                
            except BreakLoopTo:
                if len(tokenised_sequence) == truncate_to:
                    break
                
                continue
        
        if len(tokenised_sequence) > max_seq_len:
            max_seq_len = len(tokenised_sequence)
        
        # if pad_to is specified and the sequence is not truncated then we should also pad the sequence. This is also important if as_tensor = True
        
        if (as_tensor and truncate_to is not None) or pad_to:
            pad_to = max_seq_len if pad_to is None or pad_to < max_seq_len else pad_to
            tokenised_sequence += [tokeniser["detokeniser"]["<UNK>"]]*max(0, pad_to - len(tokenised_sequence))
        
        res.append(tokenised_sequence)
    
    # Unbatch this if there is only one input string
    if len(res) == 1:
        return res[0]
    elif not as_tensor:
        return res
    
    return LongTensor(res)
        
              

def ids_to_text(ids: Union[LongTensor, Union[List[int], List[List[int]]]], tokeniser: TokeniserData) -> Union[List[str], str]:
    # first we convert to Python list and batch input if it is not already. We will undo this at the end if necessary.
    if isinstance(ids, LongTensor):
        ids = ids.tolist()
    if isinstance(ids[0], int):
        ids = [ids]
        
    detokeniser = {id: token for token, id in tokeniser["detokeniser"].items()}  # TODO: fix this. Is backwards the way I programmed it above!
    res = ["".join([detokeniser[id] for id in sequence_ids]) for sequence_ids in ids]
    
    return res if len(res) > 1 else res[0]
  
        
    