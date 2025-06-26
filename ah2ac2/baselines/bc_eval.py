from __future__ import annotations

import asyncio
import logging
import time
import jax.numpy as jnp

from typing import Callable, Any
from flax import core, struct
from jaxmarl.wrappers.baselines import load_params
from omegaconf import OmegaConf
from ah2ac2.nn.multi_layer_lstm import MultiLayerLstm
from ah2ac2.evaluation.evaluation_space import EvaluationSpace

logging.basicConfig(level=logging.INFO)

_TEST_API_KEY = "<API_KEY>"
_EVALUATION_API_KEY = "<API_KEY>"
_NUM_CONCURRENT_EVAL_INSTANCES = 3


class LstmBcAgent(struct.PyTreeNode):
    params: core.FrozenDict[str, Any]
    hidden_state: Any
    apply_fn: Callable = struct.field(pytree_node=False)

    def act(self, obs, legal_actions) -> tuple[int, LstmBcAgent]:
        obs = jnp.expand_dims(obs, axis=(0, 1))  # Add batch & time dimensions.
        new_hidden_state, batch_logits = self.apply_fn(
            {"params": self.params},
            x=obs,
            carry=self.hidden_state,
            training=False,
        )
        batch_logits = batch_logits.squeeze()  # Remove batch & time dimensions.

        # Choose move from logits.
        legal_logits = batch_logits - (1 - legal_actions) * 1e10
        greedy_action = jnp.argmax(legal_logits, axis=-1)
        return int(greedy_action), self.replace(hidden_state=new_hidden_state)

    @classmethod
    def create(cls, agent_config, params):
        network = MultiLayerLstm(
            preprocessing_features=agent_config.preprocessing_features,
            lstm_features=agent_config.lstm_features,
            postprocessing_features=agent_config.postprocessing_features,
            action_dim=agent_config.action_dim,
            dropout_rate=agent_config.dropout,
            activation_fn_name=agent_config.act_fn,
        )
        hidden_state = network.initialize_carry(batch_size=1)
        return cls(apply_fn=network.apply, params=params, hidden_state=hidden_state)


class AgentSpecification:
    def __init__(self, name: str, path: str):
        self.name = name
        self.path = path

    def init_agent(self):
        agent_config = OmegaConf.load(f"{self.path}.yaml")
        agent_params = load_params(f"{self.path}.safetensors")
        return LstmBcAgent.create(agent_config=agent_config, params=agent_params)


async def interaction_test_2p():
    agent = AgentSpecification(
        "BC-1k-2p",
        "../models/bc_1k/epoch18_seed1_valacc0.467_2p"
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
    agent_spec_3p = AgentSpecification(
        "BC-1k-3p",
        "../models/bc_1k/epoch22_seed0_valacc0.397_3p"
    )
    num_players = 3
    candidate_positions_collection = [[0], [1, 2]]
    for candidate_positions in candidate_positions_collection:
        eval_space = EvaluationSpace(_TEST_API_KEY)
        env = eval_space.new_test_environment(num_players, candidate_positions)
        # Reset environment -> get initial local observations and legal moves.
        obs, legal_moves = await env.reset()
        # Once reset is called, there is info about the environment.
        my_agents = {c: agent_spec_3p.init_agent() for c in env.info.candidate_controlling}

        done, score = False, None
        while not done:
            env_act = {}
            for c in env.info.candidate_controlling:
                env_act[c], my_agents[c] = my_agents[c].act(obs[c], legal_moves[c])

            obs, score, done, legal_moves = await env.step(env_act)
            logging.info(f"Current Score: {score}, Game Done: {done}")


async def main():
    agent_spec_2p = AgentSpecification(
        "BC-1k-2p",
        "../models/bc_1k/epoch18_seed1_valacc0.467_2p"
    )
    agent_spec_3p = AgentSpecification(
        "BC-1k-3p",
        "../models/bc_1k/epoch22_seed0_valacc0.397_3p"
    )

    eval_space = EvaluationSpace(_EVALUATION_API_KEY)

    counter = 1
    while not eval_space.info.human_ai_eval_done:
        # Start timing this game
        start_time = time.time()

        # Get new environment instance for evaluation.
        env = eval_space.next_environment()
        # Connect to the server and get the first observation.
        obs, legal_moves = await env.reset()
        # Once connected, more info about the current environment becomes available.
        logging.info(f"Game ID: {env.info.game_id}")

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
