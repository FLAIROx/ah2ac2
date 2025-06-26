import jax
import jax.numpy as jnp
import chex

from collections.abc import Iterator
from typing import Optional
from ah2ac2.datasets.dataset import HanabiLiveGamesDataset, _Games


class HanabiLiveGamesDataloader:
    """
    A data loader class for iterating over the HanabiLiveGamesDataset in batches.

    This class supports optional shuffling of the dataset during iteration.
    """

    def __init__(
        self,
        dataset: HanabiLiveGamesDataset,
        batch_size: Optional[int] = None,
        shuffle_key: Optional[chex.PRNGKey] = None,
    ) -> None:
        """
        Initialize the HanabiLiveGamesDataloader.

        Args:
            dataset (HanabiLiveGamesDataset): The dataset to load data from.
            batch_size (Optional[int]): The number of games per batch. If None, the entire dataset is treated as one batch.
            shuffle_key (Optional[chex.PRNGKey]): Key used for shuffling the dataset during iteration. If not provided, the dataset is not shuffled.
        """
        self.dataset = dataset
        self.batch_size = batch_size if batch_size is not None else len(dataset)
        self.shuffle_key = shuffle_key

    def __iter__(self):
        """
        Iterator for the data loader.

        Returns:
            _BatchIter: An iterator that yields batches of game data.
        """
        if self.shuffle_key is not None:
            self.shuffle_key, _ = jax.random.split(self.shuffle_key)
        return _BatchIter(self)

    def __len__(self):
        """
        Get the number of batches in the data loader.

        Returns:
            int: The number of batches.
        """
        from math import ceil

        dataset_len = len(self.dataset)
        return ceil(dataset_len / self.batch_size)


class _BatchIter(Iterator):
    """
    An iterator class for yielding batches of game data from the HanabiLiveGamesDataloader.
    """

    def __init__(self, loader: HanabiLiveGamesDataloader):
        """
        Initialize the _BatchIter.

        Args:
            loader (HanabiLiveGamesDataloader): The data loader to iterate over.
        """
        self._loader: HanabiLiveGamesDataloader = loader
        self._dataset: HanabiLiveGamesDataset = loader.dataset
        self._current: int = 0
        self._batch_size = self._loader.batch_size

        # If key is not available, we will not shuffle the data.
        key = loader.shuffle_key
        if key is None:
            self._iter_indices: jnp.array = jnp.arange(len(self._dataset))
        else:
            rng, _rng = jax.random.split(key)
            self._iter_indices: jnp.array = jax.random.permutation(_rng, len(self._dataset))

    def __iter__(self):
        """
        Return the iterator object.

        Returns:
            _BatchIter: The iterator itself.
        """
        return self

    def __next__(self) -> _Games:
        """
        Get the next batch of game data.

        Returns:
            _Games: A named tuple containing the next batch of game data.
        """
        if self._current >= len(self._iter_indices):
            raise StopIteration

        if self._current + self._batch_size > len(self._iter_indices):
            batch_indices = self._iter_indices[self._current :]
        else:
            batch_indices = self._iter_indices[self._current : self._current + self._batch_size]
        self._current += self._batch_size

        return self._dataset[batch_indices]
