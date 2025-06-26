# Tutorial: Unrolling Games in JaxMARL

This page explains how to use the AH2AC2 dataset, in conjunction with the `HanabiLiveGamesDataset` and `HanabiLiveGamesDataloader` utilities, to unroll game trajectories within the [JaxMARL](https://github.com/InstaDeepAI/JaxMARL) framework.

**Primary Resource: `tutorial.ipynb`**

While this page provides an overview and conceptual code snippets, the most comprehensive and runnable example is located in the Jupyter Notebook: `ah2ac2/datasets/tutorial.ipynb`. You can refer to this notebook for the complete implementation details.

## Core Concept

The process of unrolling a game involves taking a recorded game (its initial deck and sequence of actions) and replaying it step-by-step in a Hanabi game environment.

The main steps are:

1.  **Initialize JaxMARL's Hanabi Environment**: Create an instance of the Hanabi environment configured for the correct number of players (obtainable from the dataset).
2.  **Reset Environment with Game Deck**: For each game from our dataset, reset the JaxMARL environment using the specific deck recorded for that game. JaxMARL's Hanabi environment has a `reset_from_deck_of_pairs` method suitable for this.
3.  **Iteratively Step Through Actions**: Use `jax.lax.scan` to iterate over the sequence of recorded actions for the game. In each iteration, apply the recorded action(s) for the current turn to the environment using its `step_env` method.
4.  **Batch Processing with `jax.vmap`**: To efficiently process multiple games (e.g., a batch from `HanabiLiveGamesDataloader`), vectorize the entire unrolling process (steps 1-3) using `jax.vmap`.

## Implementation

```python

import jax
import chex
import jax.numpy as jnp

from typing import NamedTuple
from jaxmarl import make
from jaxmarl.environments.hanabi import hanabi_game
from jaxmarl.environments.hanabi.hanabi import HanabiEnv

from ah2ac2.datasets.dataset import HanabiLiveGamesDataset, _Games
from ah2ac2.datasets.dataloader import HanabiLiveGamesDataloader


class Transition(NamedTuple):
    current_timestep: int  # We know there is `turn` in env_state, but game might reset!
    env_state: hanabi_game.State  # Current state of the environment.
    reached_terminal: jnp.bool_


dataset_file_path = "./data/2_players_games.safetensors"
train_dataset = HanabiLiveGamesDataset(file=dataset_file_path)

# Initialize Dataloader with the dataset, batch size, and shuffle key
train_loader = HanabiLiveGamesDataloader(
    dataset=train_dataset,
    batch_size=32,
    shuffle_key=jax.random.PRNGKey(42)
)


def make_play(num_players):
    env: HanabiEnv = make("hanabi", num_agents=int(num_players))

    def play(
            rng: chex.PRNGKey,
            deck: chex.Array,
            actions: chex.Array,
    ):
        # Initialize the environment from the deck.
        _, initial_env_state = env.reset_from_deck_of_pairs(deck)

        def _step(transition: Transition, step_actions: jax.Array):
            # Unbatchify actions
            env_act = {a: step_actions[i] for i, a in enumerate(env.agents)}

            # Step the environment with selected actions.
            new_obs, new_env_state, reward, dones, infos = env.step_env(
                rng,  # NOTE: This is not really important, not stochastic.
                transition.env_state,
                env_act,
            )

            is_episode_end = jnp.logical_or(dones["__all__"], transition.reached_terminal)
            jax.debug.print("Current Score={s}", s=new_env_state.score)
            return Transition(
                current_timestep=transition.current_timestep + 1,
                env_state=new_env_state,
                reached_terminal=is_episode_end
            ), None

        initial_transition = Transition(
            current_timestep=0,
            env_state=initial_env_state,
            reached_terminal=False,
        )
        return jax.lax.scan(_step, initial_transition, actions)

    return play


play_game_vjit = jax.vmap(make_play(train_loader.dataset.num_players), in_axes=0)
for game_batch in train_loader:
    batch_actions = game_batch.actions
    batch_decks = game_batch.decks

    play_game_keys = jax.random.split(jax.random.PRNGKey(0), game_batch.game_ids.size)
    final_transition, _ = play_game_vjit(play_game_keys, batch_decks, batch_actions)
```