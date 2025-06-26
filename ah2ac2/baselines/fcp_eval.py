from __future__ import annotations

import asyncio
import logging
import time
import distrax
import functools
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np

from flax.linen.initializers import constant, orthogonal
from typing import Sequence, Any, Callable
from jaxmarl.wrappers.baselines import load_params
from omegaconf import OmegaConf
from flax import struct
from flax import core
from ah2ac2.evaluation.evaluation_space import EvaluationSpace

logging.basicConfig(level=logging.INFO)

_TEST_API_KEY = "<API_KEY>"
_EVALUATION_API_KEY = "<API_KEY>"
_NUM_CONCURRENT_EVAL_INSTANCES = 3


class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins = x
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1], name='GRUCell_1')(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(
            jax.random.PRNGKey(0), (batch_size, hidden_size)
        )


class ActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    config: ...

    @nn.compact
    def __call__(self, hidden, x):
        obs, avail_actions = x
        embedding = nn.Dense(
            self.config.fc_dim_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = embedding
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(self.config.gru_hidden_dim, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        unavail_actions = 1 - avail_actions
        action_logits = actor_mean - (unavail_actions * 1e10)
        pi = distrax.Categorical(logits=action_logits)

        critic = nn.Dense(self.config.fc_dim_size, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return hidden, action_logits


class BrBcRnnAgent(struct.PyTreeNode):
    apply_fn: Callable = struct.field(pytree_node=False)
    params: core.FrozenDict[str, Any]
    hidden_state: Any

    def act(self, obs, legal_actions) -> tuple[int, BrBcRnnAgent]:
        ac_in = (
            jnp.expand_dims(obs, axis=(0, 1)),
            jnp.expand_dims(legal_actions, axis=(0, 1))
        )
        new_hidden_state, batch_logits = self.apply_fn(
            {"params": self.params},
            self.hidden_state,
            ac_in
        )
        batch_logits = batch_logits.squeeze()  # Remove batch & time dimensions.

        # Choose move from logits.
        legal_logits = batch_logits - (1 - legal_actions) * 1e10
        greedy_action = jnp.argmax(legal_logits, axis=-1)
        return int(greedy_action), self.replace(hidden_state=new_hidden_state)

    @classmethod
    def create(cls, config, params):
        num_actions = 21 if config.num_players == 2 else 31
        hidden_state = ScannedRNN.initialize_carry(batch_size=1, hidden_size=config.gru_hidden_dim)
        network = ActorCriticRNN(num_actions, config=config)

        return cls(apply_fn=network.apply, params=params, hidden_state=hidden_state)


class AgentSpecification:
    def __init__(self, name: str, path: str):
        self.name = name
        self.path = path

    def init_agent(self):
        agent_config = OmegaConf.load(f"{self.path}.yaml")
        agent_params = load_params(f"{self.path}.safetensors")["params"]
        return BrBcRnnAgent.create(config=agent_config, params=agent_params)


async def interaction_test_2p():
    agent = AgentSpecification(
        "FCP-3p",
        "../models/fcp_1k/2p/seed30_step132894_ret11.380859375_2p"
    )
    num_players = 2
    candidate_positions = [0]
    # Create evaluation space.
    eval_space = EvaluationSpace(_TEST_API_KEY)
    env = eval_space.new_test_environment(num_players, candidate_positions)
    # Reset environment -> get initial local observations and legal moves.
    obs, legal_moves = await env.reset()
    # Once reset is called, there is info about the environment.
    my_agents = {c: agent.init_agent() for c in env.info.candidate_controlling}

    done, score = False, None
    while not done:
        env_act = {}
        for c in env.info.candidate_controlling:
            env_act[c], my_agents[c] = my_agents[c].act(obs[c], legal_moves[c])

        obs, score, done, legal_moves = await env.step(env_act)
        logging.info(f"Current Score: {score}, Game Done: {done}")


async def interaction_test_3p():
    agent = AgentSpecification(
        "FCP-2p",
        "../models/fcp_1k/3p/seed30_step53850_ret3.75_3p"
    )
    num_players = 3
    candidate_positions = [0, 2]
    # Create evaluation space.
    eval_space = EvaluationSpace(_TEST_API_KEY)
    env = eval_space.new_test_environment(num_players, candidate_positions)
    # Reset environment -> get initial local observations and legal moves.
    obs, legal_moves = await env.reset()
    # Once reset is called, there is info about the environment.
    my_agents = {c: agent.init_agent() for c in env.info.candidate_controlling}

    done, score = False, None
    while not done:
        env_act = {}
        for c in env.info.candidate_controlling:
            env_act[c], my_agents[c] = my_agents[c].act(obs[c], legal_moves[c])

        obs, score, done, legal_moves = await env.step(env_act)
        logging.info(f"Current Score: {score}, Game Done: {done}")


async def main():
    agent_spec_2p = AgentSpecification(
        "FCP-2p",
        "../models/fcp_1k/2p/seed30_step132894_ret11.380859375_2p"
    )
    agent_spec_3p = AgentSpecification(
        "FCP-3p",
        "../models/fcp_1k/3p/seed30_step53850_ret3.75_3p"
    )

    eval_space = EvaluationSpace(_EVALUATION_API_KEY)

    counter = 1
    while not eval_space.info.human_ai_eval_done:
        # Start timing this game
        start_time = time.time()

        env = eval_space.next_environment()
        obs, legal_moves = await env.reset()

        if env.info.num_players == 2:
            agents = {c: agent_spec_2p.init_agent() for c in env.info.candidate_controlling}
        elif env.info.num_players == 3:
            agents = {c: agent_spec_3p.init_agent() for c in env.info.candidate_controlling}
        else:
            raise ValueError("Invalid environment info.")

        done, score = False, None
        while not done:
            env_act = {}
            for c in env.info.candidate_controlling:
                env_act[c], agents[c] = agents[c].act(obs[c], legal_moves[c])

            obs, score, done, legal_moves = await env.step(env_act)

        logging.info(f"Final Game Score = {score}")
        logging.info("Game duration = %.2f seconds", time.time() - start_time)
        logging.info(f"Games finished in this loop = {counter}")
        counter = counter + 1


async def _run_all() -> None:
    logging.info("Running 2P interaction test.")
    await interaction_test_2p()

    logging.info("Running 3P interaction test.")
    await interaction_test_3p()

    logging.info(f"Running evaluation with {_NUM_CONCURRENT_EVAL_INSTANCES} concurrent instances.")

    async def _run_one_instance(idx: int):
        """Wrapper around main() so exceptions don’t bubble out."""
        try:
            logging.info(f"Starting main instance {idx}.", )
            await main()
            logging.info(f"Main instance {idx} finished successfully.")
        except Exception as exc:
            logging.error(
                "Main instance %d crashed: %s", idx, exc, exc_info=True
            )

    evaluation_loops = [
        asyncio.create_task(_run_one_instance(i + 1), name=f"main-{i + 1}")
        for i in range(_NUM_CONCURRENT_EVAL_INSTANCES)
    ]
    await asyncio.gather(*evaluation_loops)


if __name__ == '__main__':
    asyncio.run(_run_all())
