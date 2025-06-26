# Dataset Object & Data Loader

To streamline and simplify the process of working with the open sourced data, we provide two main utility classes: `HanabiLiveGamesDataset` and `HanabiLiveGamesDataloader`. These classes offer convenient ways to load, access, augment, and batch the game data. Below we go through the basic usage. Also, these classes are open-sourced and come with our codebase, so you feel free to modify or extend them in any way you like.

## Download Data

You should have downloaded data. If you haven't done this yet, run `ah2ac2/datasets/download_dataset.py`. This will download the data into `ah2ac2/datasets/data/`.

## Dataset Object (`HanabiLiveGamesDataset`)


Once you downloaded the data, you can take advatage of `HanabiLiveGamesDataset` which serves as a primary interface to the game data stored in `.safetensors` files. It handles loading the data and converting it into JAX arrays, making it ready for use with [JaxMARL](https://github.com/FLAIROx/JaxMARL).

**Key Features:**

*   **Direct Loading from `.safetensors`**: Initializes directly from a specified dataset file path.
*   **JAX Compatibility**: All game data is loaded and exposed as JAX objects.
*   **Data Augmentation via Color Shuffling**: If a `color_shuffle_key` (a JAX PRNGKey) is provided during initialization, the class can perform on-the-fly color permutation for card colors in decks and hint actions.
*   **Support for Limited Data Regimes**: Facilitates experiments with smaller portions of the dataset. You can specify either a percentage of the data to use (via `limited_data_regime_ratio` and `limited_data_regime_key`) or a fixed number of games (via `limited_data_regime_games` and `limited_data_regime_key`). This is useful for studying data efficiency.
*   **Structured Game Access**: Accessing an item from the dataset (e.g., `dataset[i]`) returns a `_Games` NamedTuple. This tuple neatly organizes all related JAX arrays for a game (or a batch of games if slicing), including `game_ids`, `scores`, `decks`, `actions`, `num_actions`, and `game_len_masks`.

The `_Games` NamedTuple is defined in `ah2ac2.datasets.dataset` and its fields correspond to the data arrays described in the [Dataset Details](./details.md#data-format-and-structure) section, but as JAX arrays.

You will probably not use the `HanabiLiveGamesDataset` without the dataloader, which we introduce next.

## Dataloader Object (`HanabiLiveGamesDataloader`)

`HanabiLiveGamesDataloader` class, found in `ah2ac2.datasets.dataloader`, is designed to work with `HanabiLiveGamesDataset` to provide an iterable over the dataset, yielding batches of game data.

**Key Features:**

*   **Batching**: Automatically groups games from the `HanabiLiveGamesDataset` into batches of a specified `batch_size`.
*   **Shuffling**: If a `shuffle_key` (a JAX PRNGKey) is provided, the dataloader will shuffle the order of games at the beginning of each iteration, ensuring that models see data in a varied order across epochs.
*   **Iterable Interface**: Implements the Python iterator protocol, making it easy to loop through.

**Basic Usage:**

```python
import jax
from ah2ac2.datasets.dataset import HanabiLiveGamesDataset, _Games
from ah2ac2.datasets.dataloader import HanabiLiveGamesDataloader

# We assume we are in `ah2ac2/datasets`.
dataset_file_path = "./data/2_players_games.safetensors"
train_dataset = HanabiLiveGamesDataset(file=dataset_file_path)

# Initialize Dataloader with the dataset, batch size, and shuffle key
train_loader = HanabiLiveGamesDataloader(
    dataset=train_dataset,
    batch_size=32,
    shuffle_key=jax.random.PRNGKey(42)
)

game_batch: _Games
for i, game_batch in enumerate(train_loader):
    print(f"Processing Batch {i+1}:")
    print(f"  Decks tensor shape: {game_batch.decks.shape}")
    print(f"  Scores tensor shape: {game_batch.scores.shape}")
    print(f"  Actions tensor shape: {game_batch.actions.shape}")
    print(f"  Game lengths tensor shape: {game_batch.num_actions.shape}")

```
Using these , you significantly simplify data handling, allowing you to focus more on model development and experimentation.
The tutorial on how to use this data to play out games in JaxMARL is also available [here](tutorial.md)