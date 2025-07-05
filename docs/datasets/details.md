# Data Details

This page provides specifics on how to acquire the AH2AC2 datasets, understand their structure, and load the raw game data.

## Getting Started

### Downloading the Data

The AH2AC2 datasets are hosted on the [Hugging Face](https://huggingface.co/datasets/ah2ac2/datasets). We provide a utility script to download the data easily into the codebase we provide. This script is available in `ah2ac2/datasets/download_datasets.py` and looks something like this: 
```python
import huggingface_hub
from huggingface_hub import snapshot_download

def download_dataset(local_dir="./data"):
    snapshot_download(repo_id="ah2ac2/datasets", repo_type="dataset", local_dir=local_dir)
```


Once you run `ah2ac2/datasets/download_datasets.py`, data is downloaded to `ah2ac2/datasets/data`. This also prepares the data for running the training pipeline for  the baseline results we report in our paper.

### Available Data & Splits 

You will notice the data is stored in the [`.safetensors`](https://github.com/huggingface/safetensors) format and you will find 6 `.safetensors` files in `ah2ac2/datasets/data`:

*      `2_players_games.safetensors`: Contains raw data for all two-player open sourced games. There are 1,858 two-player games in this file.
*      `3_players_games.safetensors`: Contains raw data for all three-player open sourced games. There are 1,221 three-player games in this file.
*      `2_player_games_train_1k.safetensors`: Contains raw data for two-player open sourced games we used for training our baselines. There are 1,000 two-player games in this file. This is a result of train/val split.
*      `2_player_games_val.safetensors`: Contains raw data for two-player open sourced games we used for validation. There are 858 two-player games in this file. This is a result of train/val split.
*      `3_player_games_train_1k.safetensors`: Contains raw data for three-player open sourced games we used for training our baselines. There are 1,000 three-player games in this file. This is a result of train/val split.
*      `3_player_games_val.safetensors`: Contains raw data for two-player open sourced games we used for validation. There are 221 two-player games in this file. This is a result of train/val split.


You can feel free to use different splits if you prefer to do that, but we do recommend using the same split as we do for our baselines to allow for direct comparsion of methods. If you use our dataset in any way, we kindly ask you to cite our paper. TODO: Add link to paper.

## Understanding the Data

### Data Format and Structure

Each `.safetensors` file, when loaded, yields a dictionary. Precisely, type is `dict[str, np.ndarray]`. The keys you will find in our dataset files are:

*   **`game_ids`**:
    *   _Description_: Unique identifiers for each game. Can be useful for debugging or tracking specific games.
    *   _Shape_: `(num_games,)`
*   **`scores`**:
    *   _Description_: The final score achieved in each respective game.
    *   _Shape_: `(num_games,)`
*   **`decks`**:
    *   _Description_: The initial deck of 50 cards for each game, before any cards are dealt to players. The top of the deck corresponds to the first element in the array for that game.
    *   _Shape_: `(num_games, 50, 2)`
    *   Card Representation: The last dimension `(2)` represents `[color_id, rank_id]`. Colors and ranks are typically 0-indexed.
*   **`actions`**:
    *   _Description_: Actions taken by each player in each game. Games are padded with a special no-op action to ensure all action sequences have the same length (`max_game_length`).
    *   _Shape_: `(num_games, max_game_length, num_players)`
*   **`num_actions`**:
    *   _Description_: The actual number of actions (turns) taken in each game before termination or padding. If you need the index of the last valid action, subtract 1 from this value.
    *   _Shape_: `(num_games,)`
*   **`num_players`**:
    *   _Description_: The number of players participating in the games within this specific file (e.g., 2 for 2-player games).

### Using Raw Data

We provide utility classes for loading the data and iterating over the dataset (see [Dataset Usage Guide](./classes.md)) and we highly recommend using these utilities when developing your methods.

If these utilities don't suit your needs, you can also load the raw data directly from the `.safetensors` files. This might be useful for quick inspection or custom data processing pipelines. Also, our utilities are aimed towards using datasets with [JaxMARL Hanabi environment](https://jaxmarl.foersterlab.com/environments/hanabi/). If you are looking to use the dataset with the [Hanabi Learning Environment (HLE)](https://arxiv.org/pdf/1902.00506), you will have to transform the data yourself as we don't provide utilities for HLE.


To load the raw data, you would do something like this:
```python
import os
import numpy as np

from safetensors.numpy import load_file

data_path = "./data/2_players_games.safetensors"
raw_game_data: dict[str, np.ndarray] = load_file(data_path)

print(f"'num_players' have shape: {raw_game_data['num_players'].shape}")
# 'num_players' have shape: ()
print(f"'actions' have shape: {raw_game_data['actions'].shape}")
# 'actions' have shape: (1858, 89, 2)
print(f"'decks' have shape: {raw_game_data['decks'].shape}")
# 'decks' have shape: (1858, 50, 2)
print(f"'game_ids' have shape: {raw_game_data['game_ids'].shape}")
# 'game_ids' have shape: (1858,)
print(f"'num_actions' have shape: {raw_game_data['num_actions'].shape}")
# 'num_actions' have shape: (1858,)
print(f"'num_players' have shape: {raw_game_data['num_players'].shape}")
# 'num_players' have shape: ()
print(f"'scores' have shape: {raw_game_data['scores'].shape}")
# 'scores' have shape: (1858,)
```
Loading the data this way gives you direct access to the NumPy arrays, allowing for maximum flexibility if the provided utility classes do not fit your specific needs. 