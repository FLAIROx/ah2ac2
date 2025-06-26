import jax
import jax.numpy as jnp
import chex

from os import PathLike
from typing import Optional, Union, NamedTuple
from safetensors.numpy import load_file

class _Games(NamedTuple):
    """
    A named tuple representing a collection of Hanabi games.

    Attributes:
        game_ids (jnp.ndarray):  Useful for debugging, will most probably not be very useful for much else. Shape `(num_games, )`.
        scores (jnp.ndarray): Final score for each respective game. Shape `(num_games, )`.
        decks (jnp.ndarray): Decks for the games in the collection. This is the deck before dealing. Top of the deck is beginning of the array. Shape `(num_games, 50, 2)`.
        actions (jnp.ndarray): Actions performed in each game. First actions is first element. Padded with no-op. Shape `(num_games, 89, num_players)`.
        num_actions (jnp.ndarray): Number of actions for each game. This is NOT the last timestep of the game; if you need the index of the last action taken, you need to subtract 1 from this value. Shape `(num_games, )`.
        game_len_masks (jnp.ndarray): Mask for the game length. True if game still has moves to go. Shape `(num_games, 89)`.
    """

    game_ids: jnp.ndarray
    scores: jnp.ndarray
    decks: jnp.ndarray
    actions: jnp.ndarray
    num_actions: jnp.ndarray
    game_len_masks: jnp.ndarray


class HanabiLiveGamesDataset:
    """
    A dataset class for loading and processing Hanabi game data.

    This class loads game data from a file, optionally shuffles colors (data augmentation), and supports limited data regime for experiments with reduced dataset sizes.
    """

    def __init__(
        self,
        file: Union[PathLike, str],
        color_shuffle_key: Optional[chex.PRNGKey] = None,
        limited_data_regime_ratio: Optional[float] = None,
        limited_data_regime_key: Optional[chex.PRNGKey] = None,
        limited_data_regime_games: Optional[int] = None,
    ):
        """
        Initialize the HanabiLiveGamesDataset.

        Args:
            file (Union[PathLike, str]): Path to the dataset file.
            color_shuffle_key (Optional[chex.PRNGKey]): Initial key used as a seed for shuffling colors.
            limited_data_regime_ratio (Optional[float]): Percentage of data to use from the dataset.
                If None, use all data. If set, you have to specify limited_data_regime_key.
                Has to be None if limited_data_regime_ratio is specified.
            limited_data_regime_key (Optional[chex.PRNGKey]): Key used to get the data for the limited data regime.
            limited_data_regime_games (Optional[int]): Number of games to use for the limited data regime.
                If None, use all data.
                Has to be None if limited_data_regime_ratio is specified.

        Raises:
            ValueError: If limited_data_regime_ratio is set without limited_data_regime_key or vice versa.
        """
        # Load the dataset from file to dictionary.
        dataset_dict = load_file(file)

        if limited_data_regime_games is not None and limited_data_regime_ratio is not None:
            raise ValueError(
                "Only limited_data_regime_games or limited_data_regime_ratio can be set."
            )
        using_limited_data_regime = (
            limited_data_regime_games is not None or limited_data_regime_ratio is not None
        )
        if using_limited_data_regime and limited_data_regime_key is None:
            raise ValueError(
                "If using limited data regime, you have to specify limited_data_regime_key."
            )
        if not using_limited_data_regime and limited_data_regime_key is not None:
            raise ValueError(
                "If not using limited data regime, don't specify limited_data_regime_key."
            )

        # Housekeeping.
        self.num_players: int = int(dataset_dict["num_players"])
        self.max_game_len = int(dataset_dict["actions"].shape[1])
        self.hand_size = 5 if self.num_players <= 3 else 4
        self.num_colors = 5  # NOTE: We always assume 5 colors.
        self.hint_color_range = jnp.arange(
            2 * self.hand_size,
            2 * self.hand_size + (self.num_players - 1) * self.num_colors,
        )

        # Handling limited data regime.
        num_games = len(dataset_dict["game_ids"])
        if using_limited_data_regime and limited_data_regime_key is not None:
            if limited_data_regime_games is not None:
                num_games_to_use = limited_data_regime_games
            elif limited_data_regime_ratio is not None:
                num_games_to_use = int(num_games * limited_data_regime_ratio)
            else:
                raise ValueError("Invalid limited data regime configuration.")

            limited_data_regime_key, limited_data_rng = jax.random.split(limited_data_regime_key)
            idx_to_use = jax.random.permutation(limited_data_rng, num_games)[:num_games_to_use]
        else:
            idx_to_use = jnp.arange(num_games)

        # Build the dataset structure.
        game_len_masks = jax.vmap(
            lambda game_len: jnp.where(jnp.arange(self.max_game_len) < game_len, True, False),
            in_axes=0,
        )(dataset_dict["num_actions"])
        self.games = _Games(
            game_ids=jnp.array(dataset_dict["game_ids"][idx_to_use]),
            scores=jnp.array(dataset_dict["scores"][idx_to_use]),
            decks=jnp.array(dataset_dict["decks"][idx_to_use]),
            actions=jnp.array(dataset_dict["actions"][idx_to_use]),
            num_actions=jnp.array(dataset_dict["num_actions"][idx_to_use]),
            game_len_masks=game_len_masks[idx_to_use],
        )

        self.color_shuffle_key = color_shuffle_key

    def __len__(self):
        """
        Number of games in the dataset.

        Returns:
            int: The number of games in the dataset.
        """
        return len(self.games.game_ids)

    def __getitem__(self, idx):
        """
        Get a game/batch of games by index.

        This method supports color shuffling if a color_shuffle_key was provided during initialization.

        Args:
            idx: Index or slice to retrieve games.

        Returns:
            _Games: A named tuple containing the requested game data.
        """
        games = jax.tree_map(lambda x: x[idx], self.games)

        decks, actions = games.decks, games.actions
        if self.color_shuffle_key is not None:
            self.color_shuffle_key, _rng = jax.random.split(self.color_shuffle_key)
            color_map = jax.random.permutation(_rng, self.num_colors)

            # Shuffle colors for decks and actions.
            decks = decks.at[..., 0].set(color_map[decks[..., 0]])
            actions = self._shuffle_hint_color_actions(color_map, actions)

        return _Games(
            game_ids=games.game_ids,
            scores=games.scores,
            decks=decks,
            actions=actions,
            num_actions=games.num_actions,
            game_len_masks=games.game_len_masks,
        )

    def _shuffle_hint_color_actions(self, color_map, actions):
        """
        Shuffle the hint color actions based on the provided color map.

        Args:
            color_map (jnp.ndarray): A permutation of color indices.
            actions (jnp.ndarray): The original actions array.

        Returns:
            jnp.ndarray: The actions array with shuffled hint color actions.
        """
        should_augment_mask = jnp.logical_and(
            actions >= self.hint_color_range[0],
            actions <= self.hint_color_range[-1],
        )

        hint_idx = (actions - self.hint_color_range[0]) % self.num_colors
        new_hint_idx = color_map[hint_idx]
        new_colors = actions - hint_idx + new_hint_idx
        shuffled_actions = jnp.where(should_augment_mask, new_colors, actions)
        return shuffled_actions

