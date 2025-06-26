from __future__ import annotations

import asyncio
import logging
import time

import chex
import jax
import flax.linen as nn
import jax.numpy as jnp

from typing import Any, Callable
from flax import core, struct
from jaxmarl.wrappers.baselines import load_params
from typing import Tuple
from chex import Array, PRNGKey
from flax.linen.module import compact, nowrap

from ah2ac2.evaluation.evaluation_space import EvaluationSpace

_TEST_API_KEY = "<API_KEY>"
_EVALUATION_API_KEY = "<API_KEY>"
_NUM_CONCURRENT_EVAL_INSTANCES = 1


class MultiLayerLSTM(nn.RNNCellBase):
    num_layers: int
    features: int

    @compact
    def __call__(self, carry, inputs):
        new_hs, new_cs = [], []
        y = inputs
        for layer in range(self.num_layers):
            new_carry, y = nn.LSTMCell(self.features, name=f"l{layer}")(
                jax.tree.map(lambda x: x[layer], carry), inputs
            )
            new_cs.append(new_carry[0])
            new_hs.append(new_carry[1])
            inputs = y

        new_final_carry = (jnp.stack(new_cs), jnp.stack(new_hs))
        return new_final_carry, y

    @nowrap
    def initialize_carry(self, rng: PRNGKey, batch_dims: Tuple[int, ...]) -> Tuple[Array, Array]:
        mem_shape = (self.num_layers,) + batch_dims + (self.features,)
        c = jnp.zeros(mem_shape)
        h = jnp.zeros(mem_shape)
        return c, h

    @property
    def num_feature_axes(self) -> int:
        return 1


class OblR2D2(nn.Module):
    hid_dim: int = 512
    out_dim: int = 21
    num_lstm_layer: int = 2
    num_ff_layer: int = 1

    @compact
    def __call__(self, carry, inputs):
        priv_s, publ_s = inputs

        # Private network.
        priv_o = nn.Sequential(
            [
                nn.Dense(self.hid_dim, name="priv_net_dense_0"),
                nn.relu,
                nn.Dense(self.hid_dim, name="priv_net_dense_1"),
                nn.relu,
                nn.Dense(self.hid_dim, name="priv_net_dense_2"),
                nn.relu,
            ]
        )(priv_s)

        # Public network (MLP+lstm)
        x = nn.Sequential([nn.Dense(self.hid_dim, name="publ_net_dense_0"), nn.relu])(publ_s)
        carry, publ_o = MultiLayerLSTM(
            num_layers=self.num_lstm_layer, features=self.hid_dim, name="lstm"
        )(carry, x)

        o = priv_o * publ_o
        a = nn.Dense(self.out_dim, name="fc_a")(o)
        return carry, a

    @nowrap
    def initialize_carry(self, rng: PRNGKey, batch_dims: Tuple[int, ...]) -> Tuple[Array, Array]:
        return MultiLayerLSTM(
            num_layers=self.num_lstm_layer, features=self.hid_dim
        ).initialize_carry(rng, batch_dims)


class OblR2D2Agent(struct.PyTreeNode):
    apply_fn: Callable = struct.field(pytree_node=False)
    params: core.FrozenDict[str, Any]
    hidden_state: Any

    def act(self, observation: chex.Array, legal_actions: chex.Array) -> tuple[int, OblR2D2Agent]:
        obs = jnp.expand_dims(observation, axis=0)  # Add batch dimension.
        priv_s = obs
        publ_s = obs[..., 125:]
        new_hidden_state, adv = self.apply_fn(self.params, self.hidden_state, (priv_s, publ_s))
        adv = adv.squeeze()  # Remove the batch dimension.
        legal_adv = (1 + adv - adv.min()) * legal_actions
        greedy_action = jnp.argmax(legal_adv, axis=-1)

        return int(greedy_action), self.replace(hidden_state=new_hidden_state)

    @classmethod
    def create(cls, model: OblR2D2, params):
        hidden_state = model.initialize_carry(jax.random.PRNGKey(0), batch_dims=(1,))
        return cls(apply_fn=model.apply, params=params, hidden_state=hidden_state)


class AgentSpecification:
    def __init__(self, name: str, path: str):
        self.name = name
        self.path = path

    def init_agent(self):
        model = OblR2D2()
        params = load_params(self.path)
        return OblR2D2Agent.create(model, params)


async def interaction_test_2p():
    agent = AgentSpecification(
        "OBL-2p",
        "../models/obl/OFF_BELIEF1_SHUFFLE_COLOR0_LOAD1_BZA0_BELIEF_a.safetensors",
    )
    candidate_positions = [0]
    # Create evaluation space.
    eval_space = EvaluationSpace(_TEST_API_KEY)
    env = eval_space.new_test_environment(2, candidate_positions)
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
        "OBL-2p",
        "../models/obl/OFF_BELIEF1_SHUFFLE_COLOR0_LOAD1_BZA0_BELIEF_a.safetensors"
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
