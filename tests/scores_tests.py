import jax
import chex
import numpy as onp
import jax.numpy as jnp

from absl.testing import absltest
from tqdm import tqdm
from typing import NamedTuple
from jaxmarl import make
from jaxmarl.environments.hanabi import hanabi_game
from ah2ac2.datasets.download_datasets import download_dataset
from ah2ac2.datasets.dataset import HanabiLiveGamesDataset
from ah2ac2.datasets.dataloader import HanabiLiveGamesDataloader
from jaxmarl.environments.hanabi.hanabi import HanabiEnv

def batchify(x, agent_list):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((len(agent_list), -1))


class Transition(NamedTuple):
    current_timestep: int  # We know there is `turn` in env_state, but game might reset!
    env_state: hanabi_game.State  # Current state of the environment.
    reached_terminal: jnp.bool_

    cum_score: jnp.int32  # Score acquired by adding up the rewards.
    final_score: jnp.int32  # Score kept by env_state reach terminal OR when we have no more moves.


def make_play(num_players):
    hand_size = 5 if num_players <= 3 else 4
    env: HanabiEnv = make("hanabi", hand_size=hand_size, num_agents=int(num_players))

    def play(
        rng: chex.PRNGKey,
        deck: chex.Array,
        actions: chex.Array,
        final_game_timestep: int,
    ):
        # Initialize the environment.
        _, initial_env_state = env.reset_from_deck_of_pairs(deck)

        def _step(transition: Transition, step_actions: jax.Array):
            # Check if all moves are legal.
            legal_moves = batchify(env.get_legal_moves(transition.env_state), env.agents)
            one_hot_actions = jax.nn.one_hot(step_actions, num_classes=env.num_actions)
            legal_moves_taken = jnp.all((legal_moves * one_hot_actions).sum(axis=-1) == 1)
            legal_moves_taken = jnp.where(
                transition.current_timestep > final_game_timestep, True, legal_moves_taken
            )

            # Unbatchify actions
            env_act = {a: step_actions[i] for i, a in enumerate(env.agents)}

            # Step the environment with selected actions.
            new_obs, new_env_state, reward, dones, infos = env.step_env(
                rng,  # NOTE: This is not really important, not stochastic.
                transition.env_state,
                env_act,
            )

            # Update final score.
            # Option 1 - Accumulate scores:
            cum_score = jnp.where(
                jnp.logical_or(
                    transition.reached_terminal,
                    transition.current_timestep > final_game_timestep,
                ),
                transition.cum_score,  # Keep old if done.
                transition.cum_score + reward["__all__"].astype(int),
            )
            # Option 2 - Record the final score:
            new_final_score = jnp.where(
                jnp.logical_or(
                    transition.reached_terminal,
                    transition.current_timestep > final_game_timestep,
                ),
                transition.final_score,  # Keep old if done.
                new_env_state.score,  # Update if more moves and not terminal.
            )

            is_episode_end = jnp.logical_or(dones["__all__"], transition.reached_terminal)
            return Transition(
                current_timestep=transition.current_timestep + 1,
                env_state=new_env_state,
                reached_terminal=is_episode_end,
                # Score keeping:
                cum_score=cum_score,
                final_score=new_final_score,
            ), legal_moves_taken

        initial_transition = Transition(
            current_timestep=0,
            env_state=initial_env_state,
            final_score=0,
            cum_score=0,
            reached_terminal=False,
        )
        final_transition, legal_moves_taken = jax.lax.scan(_step, initial_transition, actions)
        return final_transition, jnp.all(legal_moves_taken)

    return play


class TestScores(absltest.TestCase):
    def setUp(self):
        download_dataset("./temp")

        self.batch_size = 512
        self.paths = [
            "./temp/2_player_games_train_5k.safetensors",
            "./temp/2_player_games_train_1k.safetensors",
            "./temp/2_player_games_val.safetensors",
            "./temp/2_player_games_val.safetensors",
            "./temp/3_player_games_train_5k.safetensors",
            "./temp/3_player_games_train_1k.safetensors",
            "./temp/3_player_games_val.safetensors",
            "./temp/3_player_games_val.safetensors",
        ]

    def test_scores(self):
        key = jax.random.PRNGKey(0)
        for file in self.paths:
            with self.subTest(msg=f"Testing scores for {file} without shuffle:"):
                print(f"\nTesting scores for {file} without shuffle:")
                dataset = HanabiLiveGamesDataset(file)
                dataloader = HanabiLiveGamesDataloader(dataset, self.batch_size)
                rng, dataset_rng, validation_rng = jax.random.split(key, 3)
                self.validate_dataset_scores(validation_rng, dataloader)

                print(f"\nTesting scores for {file} with shuffle:")
                rng, color_shuffle_rng, data_shuffle_rng = jax.random.split(key, 3)
                dataset = HanabiLiveGamesDataset(file, color_shuffle_rng)
                dataloader = HanabiLiveGamesDataloader(dataset, self.batch_size, data_shuffle_rng)
                rng, dataset_rng, validation_rng = jax.random.split(key, 3)
                self.validate_dataset_scores(validation_rng, dataloader)

    def validate_dataset_scores(self, key, loader: HanabiLiveGamesDataloader):
        env_scores, dataset_scores, game_ids = [], [], []

        play_game_vjit = jax.jit(jax.vmap(make_play(loader.dataset.num_players), in_axes=0))
        for game_batch in tqdm(loader):
            batch_actions = game_batch.actions
            batch_decks = game_batch.decks
            batch_final_timesteps = game_batch.num_actions - 1
            batch_scores = game_batch.scores
            batch_game_ids = game_batch.game_ids

            play_game_keys = jax.random.split(key, batch_game_ids.size)

            final_transition, contains_only_legal_moves = play_game_vjit(
                play_game_keys, batch_decks, batch_actions, batch_final_timesteps
            )

            if not jnp.all(contains_only_legal_moves):
                idxs = jnp.argwhere(contains_only_legal_moves == 0).ravel()
                illegal_move_containing_game_ids = game_batch.game_ids[idxs]
                self.assertTrue(
                    jnp.all(contains_only_legal_moves),
                    f"Game Ids with Illegal Moves: {illegal_move_containing_game_ids}"
                )

            # Check if all methods for score keeping match.
            self.assertTrue(jnp.array_equal(final_transition.cum_score, final_transition.final_score), "Cumulative scores don't match final scores.")

            # Save game scores.
            env_scores.append(onp.array(final_transition.final_score))
            dataset_scores.append(onp.array(batch_scores))
            game_ids.append(onp.array(batch_game_ids))

        env_scores = onp.concatenate(env_scores)
        dataset_scores = onp.concatenate(dataset_scores)
        all_game_ids = onp.concatenate(game_ids)

        """Print information about mismatches, if any."""
        non_matching_indices = onp.where(env_scores != dataset_scores)[0]
        for non_match_i in non_matching_indices:
            print(f"Index={non_match_i}", end=" | ")
            print(f"ID={all_game_ids[non_match_i]}", end=" | ")
            print(f"JAX Score={env_scores[non_match_i]}", end=" | ")
            print(f"DB Score={dataset_scores[non_match_i]}")

        # Are there any mismatches?
        num_matches = onp.equal(env_scores, dataset_scores).sum()
        num_mismatches = len(loader.dataset) - num_matches
        print(f"Matches={num_matches}, Mismatches={num_mismatches}")

        self.assertEqual(num_mismatches, 0)