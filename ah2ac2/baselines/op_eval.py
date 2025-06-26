from __future__ import annotations

import asyncio
import logging
import time

import distrax
import pickle
import chex
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np

from flax.linen.initializers import constant, orthogonal
from flax import struct
from flax import core
from typing import Callable, Any, Dict, Sequence
from ah2ac2.evaluation.evaluation_space import EvaluationSpace

logging.basicConfig(level=logging.INFO)

_TEST_API_KEY = "<API_KEY>"
_EVALUATION_API_KEY = "<API_KEY>"
_NUM_CONCURRENT_EVAL_INSTANCES = 3

class ActorCritic(nn.Module):
    action_dim: Sequence[int]

    @nn.compact
    def __call__(self, x):
        obs = x
        embedding = nn.Dense(
            512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)

        actor_mean = nn.Dense(512, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        critic = nn.Dense(512, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return actor_mean


class OPERSAgent(struct.PyTreeNode):
    apply_fn: Callable = struct.field(pytree_node=False)
    params: core.FrozenDict[str, Any]

    def act(self, obs: chex.Array, legal_actions: chex.Array) -> tuple[int, OPERSAgent]:
        # Fixed for consistency
        rng = jax.random.PRNGKey(0)

        obs = jnp.expand_dims(obs, axis=(0, 1))

        actor_mean = self.apply_fn(
            self.params,
            x=obs
        )
        actor_mean = actor_mean.squeeze()
        unavail_actions = 1 - legal_actions
        action_logits = actor_mean - (unavail_actions * 1e10)
        pi = distrax.Categorical(logits=action_logits)
        return int(pi.sample(seed=rng)), self

    @classmethod
    def create(cls, network: ActorCritic, params: Dict):
        return cls(apply_fn=network.apply, params=params)


class AgentSpecification:
    def __init__(self, name: str, path: str, num_players: int):
        self.name = name
        self.path = path
        self.num_players = num_players

    def init_agent(self):
        action_dim = 21 if self.num_players == 2 else 31
        params = load_ippo_ff(self.path)
        network = ActorCritic(action_dim=action_dim)
        return OPERSAgent.create(network=network, params=params)


def load_ippo_ff(checkpoint_path):
    with open(checkpoint_path, 'rb') as f:
        params_dict = pickle.load(f)
    if 'layer1' in params_dict['params']:
        key_map = {'layer1': 'Dense_0', 'layer2': 'Dense_1', 'layer3': 'Dense_2', 'layer4': 'Dense_3', 'layer5': 'Dense_4'}
        params_dict['params'] = {key_map[key]: value for key, value in params_dict['params'].items() if key in key_map}
    return params_dict


async def interaction_test_2p():
    agent = AgentSpecification(
        "OP-2p",
        "../models/op/op_2p.pkl",
        2
    )
    candidate_positions = [0]
    # Create evaluation space.
    eval_space = EvaluationSpace(_TEST_API_KEY)
    env = eval_space.new_test_environment(agent.num_players, candidate_positions)
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
        "OP-3p",
        "../models/op/op_3p.pkl",
        3
    )
    candidate_positions = [0, 1]
    # Create evaluation space.
    eval_space = EvaluationSpace(_TEST_API_KEY)
    env = eval_space.new_test_environment(agent.num_players, candidate_positions)
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
        "OP-2p",
        "../models/op/op_2p.pkl",
        2
    )
    agent_spec_3p = AgentSpecification(
        "OP-3p",
        "../models/op/op_3p.pkl",
        3
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

