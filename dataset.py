import pandas as pd
import torch
import better_tokens
from torch import Tensor
from typing import List, Iterator, Union



# constants

episode_data_path = "data/SouthPark_Episodes.csv"
line_data_path = "data/SouthPark_Lines.csv"

max_episode_size = 8000

# tokeniser code

tokeniser: better_tokens.TokeniserData = None

# dataset code

_scripts = None
_scripts_masks = None
_scripts_tokens = None

def get_vocab_size() -> int:
    return tokeniser["vocab_size"]

def _init_scripts(load_from_file: str = "South_Park_Tokeniser_2.json"):
    """
    Initializes script data from csvs. Each script is formatted episode by episode, the idea being we want
    this data to be together in order to predict plausible South Park episodes that sound 'logical'. Also loads tokeniser. If
    load_from_file is left as None, then will retrain tok
    """
    global _scripts, _scripts_masks, _scripts_tokens, tokeniser
    _scripts = []
    
    lines = pd.read_csv(line_data_path)
    episodes = pd.read_csv(episode_data_path)
    
    lines.dropna(inplace=True)
    
    for episode in episodes.iloc:
        episode_script, bad_episode = [], False
        
        episode_script.append(
            f"[Title: {episode['Title']}, Description: {episode['Description']}]\n\n".lower()
        )
        
        for line in lines.loc[lines["Title"] == episode["Title"]].iloc:
            if line["Line"].isascii():
                episode_script.append(f"{line['Character']}:{line['Line']}".lower())
            else:
                bad_episode = True
                break
        if not bad_episode:
            _scripts.append("\n".join(episode_script))
    
    tokeniser = better_tokens.load_tokeniser(load_from_file)
    
    _scripts_tokens, _scripts_masks = better_tokens.text_to_ids(_scripts, tokeniser, truncate_to=max_episode_size)


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
                _scripts_masks[shuffle_indices[batch_num*batch_size:(batch_num + 1)*batch_size], sequence_num*sequence_size:(sequence_num+1)*sequence_size],
                False,
            )
        yield (
            _scripts_tokens[shuffle_indices[batch_num*batch_size:(batch_num + 1)*batch_size], n_sequences*sequence_size:],
            _scripts_masks[shuffle_indices[batch_num*batch_size:(batch_num + 1)*batch_size], n_sequences*sequence_size:],
            True
        )
    
    for sequence_num in range(n_sequences):
        yield (
                _scripts_tokens[shuffle_indices[n_batches*batch_size:], sequence_num*sequence_size:(sequence_num+1)*sequence_size],
                _scripts_masks[shuffle_indices[n_batches*batch_size:], sequence_num*sequence_size:(sequence_num+1)*sequence_size],
                False,
            )
    yield (
                _scripts_tokens[shuffle_indices[n_batches*batch_size:], n_sequences*sequence_size:],
                _scripts_masks[shuffle_indices[n_batches*batch_size:], n_sequences*sequence_size:],
                True,
            )
    
        