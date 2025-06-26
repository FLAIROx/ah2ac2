import jax.random

from ah2ac2.datasets.dataset import HanabiLiveGamesDataset
from ah2ac2.datasets.dataloader import HanabiLiveGamesDataloader

import os
import jax.random
import numpy as np

from absl.testing import absltest
from safetensors.numpy import save_file


class DataloaderTests(absltest.TestCase):
    def setUp(self):
        super().setUp()

        mock_data_dir = "./temp"
        os.makedirs(mock_data_dir, exist_ok=True)

        self.two_player_games_file = f"{mock_data_dir}/mock_two_player_dataloader_data.safetensors"
        mock_two_player_games = {
            "num_players": np.array(2),
            "game_ids": np.array([0, 1]),
            "scores": np.array([22, 23]),
            "num_actions": np.array([6, 6]),
            "decks": np.array(
                [
                    [
                        [3, 2],
                        [4, 2],
                        [0, 2],
                        [1, 3],
                        [0, 2],
                        [1, 4],
                        [1, 2],
                        [2, 0],
                        [4, 0],
                        [4, 3],
                    ],
                    [
                        [0, 0],
                        [3, 4],
                        [3, 0],
                        [1, 3],
                        [2, 4],
                        [3, 2],
                        [1, 0],
                        [1, 3],
                        [1, 0],
                        [4, 1],
                    ],
                    [
                        [3, 2],
                        [4, 2],
                        [0, 2],
                        [1, 3],
                        [0, 2],
                        [1, 4],
                        [1, 2],
                        [2, 0],
                        [4, 0],
                        [4, 3],
                    ],
                    [
                        [0, 0],
                        [3, 4],
                        [3, 0],
                        [1, 3],
                        [2, 4],
                        [3, 2],
                        [1, 0],
                        [1, 3],
                        [1, 0],
                        [4, 1],
                    ],
                    [
                        [3, 2],
                        [4, 2],
                        [0, 2],
                        [1, 3],
                        [0, 2],
                        [1, 4],
                        [1, 2],
                        [2, 0],
                        [4, 0],
                        [4, 3],
                    ],
                    [
                        [0, 0],
                        [3, 4],
                        [3, 0],
                        [1, 3],
                        [2, 4],
                        [3, 2],
                        [1, 0],
                        [1, 3],
                        [1, 0],
                        [4, 1],
                    ],
                ]
            ),
            "actions": np.array(
                [
                    # D1, D1, HC1, HC3, H1, P5
                    [[0, 20], [20, 0], [10, 20], [20, 12], [15, 20], [20, 9], [20, 20]],
                    # P5, HC1, HC2, HC3, HC4, HC5, H1
                    [[9, 20], [20, 10], [11, 20], [20, 12], [13, 20], [20, 14], [15, 20]],
                    # D1, D1, HC1, HC3, H1, P5
                    [[0, 20], [20, 0], [10, 20], [20, 12], [15, 20], [20, 9], [20, 20]],
                    # P5, HC1, HC2, HC3, HC4, HC5, H1
                    [[9, 20], [20, 10], [11, 20], [20, 12], [13, 20], [20, 14], [15, 20]],
                    # D1, D1, HC1, HC3, H1, P5
                    [[0, 20], [20, 0], [10, 20], [20, 12], [15, 20], [20, 9], [20, 20]],
                    # P5, HC1, HC2, HC3, HC4, HC5, H1
                    [[9, 20], [20, 10], [11, 20], [20, 12], [13, 20], [20, 14], [15, 20]],
                ]
            ),
        }
        save_file(mock_two_player_games, self.two_player_games_file)

    def test_dataloader_iter(self):
        # Test iterating with batch size.
        batch_size = 2
        dataset = HanabiLiveGamesDataset(
            file=self.two_player_games_file,
            color_shuffle_key=jax.random.PRNGKey(0),
            limited_data_regime_ratio=1.0,
            limited_data_regime_key=jax.random.PRNGKey(1),
        )
        dataloader = HanabiLiveGamesDataloader(dataset, batch_size=batch_size)
        for batch in dataloader:
            batch_objects_shapes = jax.tree_map(lambda x: x.shape, batch)
            for shapes in batch_objects_shapes:
                self.assertEqual(shapes[0], batch_size)

        # Test getting all data i.e. batch size = None.
        batch_size = None
        dataset = HanabiLiveGamesDataset(
            file=self.two_player_games_file,
            color_shuffle_key=jax.random.PRNGKey(0),
            limited_data_regime_ratio=1.0,
            limited_data_regime_key=jax.random.PRNGKey(1),
        )
        dataloader = HanabiLiveGamesDataloader(dataset, batch_size=batch_size)
        for batch in dataloader:
            batch_objects_shapes = jax.tree_map(lambda x: x.shape, batch)
            for shapes in batch_objects_shapes:
                self.assertEqual(shapes[0], len(dataset))
