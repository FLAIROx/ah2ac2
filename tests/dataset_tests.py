import os
import jax.random
import numpy as np
import jax.numpy as jnp

from absl.testing import absltest
from ah2ac2.datasets.dataset import HanabiLiveGamesDataset
from safetensors.numpy import save_file


class DatasetTests(absltest.TestCase):
    def setUp(self):
        super().setUp()

        mock_data_dir = "./temp"
        os.makedirs(mock_data_dir, exist_ok=True)

        self.two_player_games_file = f"{mock_data_dir}/mock_two_player_data.safetensors"
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
                ]
            ),
            "actions": np.array(
                [
                    # D1, D1, HC1, HC3, H1, P5
                    [[0, 20], [20, 0], [10, 20], [20, 12], [15, 20], [20, 9], [20, 20]],
                    # P5, HC1, HC2, HC3, HC4, HC5, H1
                    [[9, 20], [20, 10], [11, 20], [20, 12], [13, 20], [20, 14], [15, 20]],
                ]
            ),
        }
        save_file(mock_two_player_games, self.two_player_games_file)

        self.three_player_games_file = f"{mock_data_dir}/mock_three_player_data.safetensors"
        three_player_games = {
            "num_players": np.array(3),
            "game_ids": np.array([2, 3]),
            "scores": np.array([24, 19]),
            "num_actions": np.array([5, 5]),
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
                ]
            ),
            "actions": np.array(
                [
                    # D1, D1, HC1@1, HC1@2, H1, P5
                    [[0, 30, 30], [30, 0, 30], [30, 30, 10], [15, 30, 30], [30, 20, 30]],
                    # P5, HC1@1, HC1@2, HC1@2, HC5@2, H1
                    [[9, 30, 30], [30, 10, 30], [30, 30, 15], [19, 30, 30], [30, 20, 30]],
                ]
            ),
        }
        save_file(three_player_games, self.three_player_games_file)

    def test_init_and_access_for_two_player_games(self):
        dataset = HanabiLiveGamesDataset(
            file=self.two_player_games_file,
            color_shuffle_key=None,
            limited_data_regime_ratio=None,
            limited_data_regime_key=None,
        )
        # Length of the dataset should be 4.
        self.assertEqual(len(dataset), 2)

        # Accessing the first element should return the first game.
        game = dataset[0]
        self.assertEqual(game.game_ids, 0)
        self.assertEqual(game.scores, 22)
        self.assertEqual(game.decks[0][0], 3)  # First color of the first deck.

        # Slicing should return both games.
        games = dataset[jnp.array([0, 1])]
        self.assertEqual(len(games.game_ids), 2)
        self.assertEqual(games.scores[1], 23)
        self.assertEqual(games.decks[1][0][0], 0)

    def test_init_and_access_for_three_player_games(self):
        dataset = HanabiLiveGamesDataset(
            file=self.three_player_games_file,
            color_shuffle_key=None,
            limited_data_regime_ratio=None,
            limited_data_regime_key=None,
        )
        # Length of the dataset should be 4.
        self.assertEqual(len(dataset), 2)

        # Accessing the first element should return the first game.
        game = dataset[0]
        self.assertEqual(game.game_ids, 2)
        self.assertEqual(game.scores, 24)
        self.assertEqual(game.decks[0][0], 3)  # First color of the first deck.

        # Index Slicing should return both games.
        games = dataset[jnp.array([0, 1])]
        self.assertEqual(len(games.game_ids), 2)
        self.assertEqual(games.scores[1], 19)
        self.assertEqual(games.decks[1][0][0], 0)

    def test_color_shuffle_for_two_players(self):
        dataset = HanabiLiveGamesDataset(
            file=self.two_player_games_file,
            color_shuffle_key=jax.random.PRNGKey(0),
            limited_data_regime_ratio=None,
            limited_data_regime_key=None,
        )

        # Get expected color map.
        _, _rng = jax.random.split(dataset.color_shuffle_key)
        expected_color_map = jax.random.permutation(_rng, dataset.num_colors)

        # Get the first game with shuffled data.
        game = dataset[0]
        self.assertEqual(game.game_ids, 0)
        self.assertEqual(game.scores, 22)

        # Values in the deck should remain the same.
        values_unchanged = jnp.allclose(game.decks[:, 1], dataset.games.decks[0][:, 1])
        self.assertTrue(values_unchanged)

        # Colors should be augmented as expected.
        game_colors = game.decks[:, 0]
        expected_colors = expected_color_map[dataset.games.decks[0][:, 0]]
        colors_match = jnp.allclose(game_colors, expected_colors)
        self.assertTrue(colors_match)

        # Some actions should remain the same.
        first_two_actions_match = jnp.allclose(game.actions[:2], dataset.games.actions[0][:2])
        self.assertTrue(first_two_actions_match)
        last_two_actions_match = jnp.allclose(game.actions[-2:], dataset.games.actions[0][-2:])
        self.assertTrue(last_two_actions_match)

        # Actions 2, 3 should be augmented.
        action_two = game.actions[2][0]  # Player 0 HC.
        action_three = game.actions[3][1]  # Player 1 HC.
        self.assertEqual(
            action_two - dataset.hint_color_range[0],
            expected_color_map[dataset.games.actions[0][2][0] - dataset.hint_color_range[0]],
        )
        self.assertEqual(
            action_three - dataset.hint_color_range[0],
            expected_color_map[dataset.games.actions[0][3][1] - dataset.hint_color_range[0]],
        )

    def test_color_shuffle_for_three_players(self):
        dataset = HanabiLiveGamesDataset(
            file=self.three_player_games_file,
            color_shuffle_key=jax.random.PRNGKey(0),
            limited_data_regime_ratio=None,
            limited_data_regime_key=None,
        )

        # Get expected color map.
        _, _rng = jax.random.split(dataset.color_shuffle_key)
        expected_color_map = jax.random.permutation(_rng, dataset.num_colors)

        # Get the second. game with shuffled data.
        game = dataset[1]
        self.assertEqual(game.game_ids, 3)
        self.assertEqual(game.scores, 19)

        # Values in the deck should remain the same.
        values_unchanged = jnp.allclose(game.decks[:, 1], dataset.games.decks[1][:, 1])
        self.assertTrue(values_unchanged)

        # Colors should be augmented as expected.
        game_colors = game.decks[:, 0]
        expected_colors = expected_color_map[dataset.games.decks[1][:, 0]]
        colors_match = jnp.allclose(game_colors, expected_colors)
        self.assertTrue(colors_match)

        # First and last actions should remain the same.
        first_turn_actions_match = jnp.allclose(game.actions[0], dataset.games.actions[1][0])
        self.assertTrue(first_turn_actions_match)
        last_turn_actions_match = jnp.allclose(game.actions[-1], dataset.games.actions[1][-1])
        self.assertTrue(last_turn_actions_match)

        # Actions @ idx 1, 2, 3 should be augmented.
        action_two = game.actions[1][1]  # Player 1.
        action_three = game.actions[2][2]  # Player 2.
        action_four = game.actions[3][0]  # Player 0.
        self.assertEqual(
            action_two - (dataset.hand_size * 2),
            expected_color_map[dataset.games.actions[1][1][1] - (dataset.hand_size * 2)],
        )
        self.assertEqual(
            action_three - (dataset.hand_size * 2) - dataset.num_colors,
            expected_color_map[
                dataset.games.actions[1][2][2] - (dataset.hand_size * 2) - dataset.num_colors
            ],
        )
        self.assertEqual(
            action_four - (dataset.hand_size * 2) - dataset.num_colors,
            expected_color_map[
                dataset.games.actions[1][3][0] - (dataset.hand_size * 2) - dataset.num_colors
            ],
        )

    def test_limited_data_regime_functionality(self):
        self.assertRaises(
            ValueError,
            lambda: HanabiLiveGamesDataset(
                file=self.two_player_games_file,
                color_shuffle_key=None,
                limited_data_regime_ratio=0.1,
                limited_data_regime_key=None,
            ),
        )
        self.assertRaises(
            ValueError,
            lambda: HanabiLiveGamesDataset(
                file=self.two_player_games_file,
                color_shuffle_key=None,
                limited_data_regime_ratio=None,
                limited_data_regime_key=jax.random.PRNGKey(0),
            ),
        )

        dataset_0 = HanabiLiveGamesDataset(
            file=self.two_player_games_file,
            color_shuffle_key=jax.random.PRNGKey(0),
            limited_data_regime_ratio=0.1,
            limited_data_regime_key=jax.random.PRNGKey(0),
        )
        self.assertEqual(len(dataset_0), 0)

        dataset_1 = HanabiLiveGamesDataset(
            file=self.two_player_games_file,
            color_shuffle_key=jax.random.PRNGKey(0),
            limited_data_regime_ratio=0.5,
            limited_data_regime_key=jax.random.PRNGKey(0),
        )
        self.assertEqual(len(dataset_1), 1)

        dataset_2 = HanabiLiveGamesDataset(
            file=self.two_player_games_file,
            color_shuffle_key=jax.random.PRNGKey(0),
            limited_data_regime_ratio=1.0,
            limited_data_regime_key=jax.random.PRNGKey(0),
        )
        self.assertEqual(len(dataset_2), 2)

    def test_dataset_iteration(self):
        dataset = HanabiLiveGamesDataset(
            file=self.two_player_games_file,
            color_shuffle_key=jax.random.PRNGKey(0),
            limited_data_regime_ratio=1.0,
            limited_data_regime_key=jax.random.PRNGKey(0),
        )

        for i in range(len(dataset)):
            game = dataset[i]
            self.assertEqual(game.game_ids, dataset.games.game_ids[i])
