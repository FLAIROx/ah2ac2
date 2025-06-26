import glob
import os

import jax
jax.config.update("jax_debug_nans", True)
import jax.numpy as jnp
import flax.linen as nn
import ml_collections
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Dict, List
from flax.training.train_state import TrainState
import distrax
import jaxmarl
from jaxmarl.wrappers.baselines import LogWrapper, load_params, save_params
import wandb
import functools
import hydra
from omegaconf import OmegaConf
from ah2ac2.nn.multi_layer_lstm import LstmCellWithHiddenStateReset

class ActorLstm(nn.Module):
    preprocessing_features: List[int]
    lstm_features: List[int]
    postprocessing_features: List[int]
    action_dim: int
    dropout_rate: float
    activation_fn_name: str

    def setup(self):
        self.lstm = nn.scan(
            LstmCellWithHiddenStateReset,
            variable_broadcast="params",
            in_axes=0,
            out_axes=0,
            split_rngs={"params": False},
        )
        if self.activation_fn_name == "relu":
            self.act_fn = nn.relu
        elif self.activation_fn_name == "gelu":
            self.act_fn = nn.gelu
        else:  # Default to relu
            self.act_fn = nn.relu

    @nn.compact
    def __call__(self, carry, x, training: bool = False):
        x, resets = x

        # Preprocess features:
        for feature_size in self.preprocessing_features:
            x = nn.Dense(feature_size)(x)
            x = nn.LayerNorm()(x)
            x = self.act_fn(x)

        # Run LSTM layers and get new hidden states:
        new_cs, new_hs = [], []
        for i, feature_size in enumerate(self.lstm_features):
            layer = self.lstm(feature_size)

            layer_carry = jax.tree_map(lambda c: c[i], carry)

            rnn_in = (x, resets)
            layer_carry, x = layer(layer_carry, rnn_in)
            new_c, new_h = layer_carry
            new_cs.append(new_c)
            new_hs.append(new_h)

        # Postprocess features:
        embedding = x
        for feature_size in self.postprocessing_features:
            x = nn.Dense(feature_size)(x)
            x = self.act_fn(x)
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)

        x = nn.Dense(self.action_dim)(x)
        new_carry = (jnp.stack(new_cs), jnp.stack(new_hs))
        return new_carry, x, embedding

    @nn.nowrap
    def initialize_carry(self, batch_size):
        carry_hs = []
        carry_cs = []

        in_features = self.preprocessing_features[-1]
        for feature_size in self.lstm_features:
            layer = nn.OptimizedLSTMCell(feature_size)

            c, h = layer.initialize_carry(jax.random.PRNGKey(0), (batch_size, in_features))
            carry_cs.append(c)
            carry_hs.append(h)

            in_features = feature_size

        return jnp.stack(carry_cs), jnp.stack(carry_hs)


class ActorCriticLSTM(nn.Module):
    actor: ActorLstm
    critic_features: int

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones, avail_actions = x

        # Actor:
        hidden, actor_mean, embedding = self.actor(hidden, (obs, dones))
        action_logits = actor_mean - (1 - avail_actions) * 1e10
        pi = distrax.Categorical(logits=action_logits)

        # Critic:
        critic = nn.Dense(
            self.critic_features, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return hidden, pi, jnp.squeeze(critic, axis=-1)

    @nn.nowrap
    def initialize_carry(self, batch_size):
        return self.actor.initialize_carry(batch_size)


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
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[0], ins.shape[1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
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
        obs, dones, avail_actions = x
        embedding = nn.Dense(
            self.config.fc_dim_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
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

        return hidden, pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    avail_actions: jnp.ndarray
    train_mask: jnp.ndarray


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def bc_annealing_schedule(config):
    start_at_update_step = config.anneal_start // config.num_steps // config.num_envs
    end_at_update_step = config.anneal_horizon // config.num_steps // config.num_envs
    return optax.linear_schedule(
        init_value=0.0,
        end_value=1.0,
        transition_begin=start_at_update_step,
        transition_steps=end_at_update_step - start_at_update_step
    )


def make_train(config):
    env = jaxmarl.make(config.env_name, **config.env_kwargs)
    config.num_actors = env.num_agents * config.num_envs
    config.num_updates = (
            config.total_timesteps // config.num_steps // config.num_envs
    )
    config.minibatch_size = (
            config.num_actors * config.num_steps // config.num_minibatches
    )

    env = LogWrapper(env)

    def linear_schedule(count):
        frac = 1.0 - (count // (config.num_minibatches * config.update_epochs)) / config.num_updates
        return config.lr * frac

    def train(rng):
        # INIT NETWORK
        network = ActorCriticRNN(env.action_space(env.agents[0]).n, config=config)
        bc_network = ActorCriticLSTM(
            actor=ActorLstm(
                preprocessing_features=config.bc_policy.preprocessing_features,
                lstm_features=config.bc_policy.lstm_features,
                postprocessing_features=config.bc_policy.postprocessing_features,
                action_dim=config.bc_policy.action_dim,
                dropout_rate=config.bc_policy.dropout,
                activation_fn_name=config.bc_policy.act_fn
            ),
            critic_features=config.bc_policy.critic_features
        )

        rng, _rng = jax.random.split(rng)
        init_x = (
            jnp.zeros(
                (1, config.num_envs, env.observation_space(env.agents[0]).n)
            ),
            jnp.zeros((1, config.num_envs)),
            jnp.zeros((1, config.num_envs, env.action_space(env.agents[0]).n))
        )

        init_hstate = ScannedRNN.initialize_carry(config.num_envs, config.gru_hidden_dim)
        network_params = network.init(_rng, init_hstate, init_x)

        bc_dummy_init_hstate = bc_network.initialize_carry(batch_size=config.num_envs)
        bc_params = bc_network.init(_rng, bc_dummy_init_hstate, init_x)
        bc_params["params"]["actor"] = jax.lax.stop_gradient(load_params(config.bc_policy.weights_path))

        lr = config.lr if not config.anneal_lr else linear_schedule
        tx = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(learning_rate=lr, eps=1e-5),
        )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        bc_anneal_schedule = bc_annealing_schedule(config)

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config.num_envs)
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        init_hstate = ScannedRNN.initialize_carry(config.num_actors, config.gru_hidden_dim)
        bc_init_hstate = bc_network.initialize_carry(batch_size=config.num_actors)

        # BC Policy annealing masking
        position_idxs = jnp.linspace(
            0,
            env.num_agents,
            config.num_envs,
            dtype=jnp.int32,
            endpoint=False
        )
        train_mask_dict = {
            a: position_idxs == i
            for i, a
            in enumerate(env.agents)
        }
        train_idxs_mask = batchify(
            train_mask_dict, env.agents, config.num_actors
        ).squeeze()
        init_bc_annealing_mask = jax.random.uniform(_rng, (config.num_envs,)) < bc_anneal_schedule(0)

        def _make_train_mask(annealing_mask):
            full_anneal_mask = jnp.tile(annealing_mask, env.num_agents)
            return jnp.where(full_anneal_mask, train_idxs_mask, True)

        # TRAIN LOOP
        def _update_step(update_runner_state, unused):
            # COLLECT TRAJECTORIES
            runner_state, update_steps = update_runner_state

            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, last_done, hstate, bc_hstate, bc_anneal_mask, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                avail_actions = jax.vmap(env.get_legal_moves)(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(
                    batchify(avail_actions, env.agents, config.num_actors)
                )

                obs_batch = batchify(last_obs, env.agents, config.num_actors)
                ac_in = (
                    obs_batch[np.newaxis, :],
                    last_done[np.newaxis, :],
                    avail_actions[np.newaxis, :]
                )

                # pi apply:
                hstate, pi, value = network.apply(train_state.params, hstate, ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                # bc apply:
                new_bc_hstate, bc_pi, bc_value = bc_network.apply(bc_params, bc_hstate, ac_in)
                bc_action = bc_pi.sample(seed=_rng)

                action_pick_mask = _make_train_mask(bc_anneal_mask)
                action = jnp.where(action_pick_mask, action, bc_action)

                # SELECT ACTION
                env_act = unbatchify(action, env.agents, config.num_envs, env.num_agents)
                env_act = jax.tree.map(lambda x: x.squeeze(), env_act)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config.num_envs)
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                    rng_step, env_state, env_act
                )
                info = jax.tree.map(lambda x: x.reshape((config.num_actors)), info)
                done_batch = batchify(done, env.agents, config.num_actors).squeeze()

                new_bc_anneal_mask = jnp.where(
                    done["__all__"],
                    jax.random.uniform(_rng, (config.num_envs,)) < bc_anneal_schedule(update_steps),
                    bc_anneal_mask
                )

                transition = Transition(
                    jnp.tile(done["__all__"], env.num_agents),
                    last_done,
                    action.squeeze(),
                    value.squeeze(),
                    batchify(reward, env.agents, config.num_actors).squeeze(),
                    log_prob.squeeze(),
                    obs_batch,
                    info,
                    avail_actions,
                    action_pick_mask
                )
                runner_state = (train_state, env_state, obsv, done_batch, hstate, new_bc_hstate, new_bc_anneal_mask, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config.num_steps
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, last_done, hstate, bc_hstate, bc_anneal_mask, rng = runner_state
            last_obs_batch = batchify(last_obs, env.agents, config.num_actors)
            avail_actions = jnp.ones(
                (config.num_actors, env.action_space(env.agents[0]).n)
            )
            ac_in = (last_obs_batch[np.newaxis, :], last_done[np.newaxis, :], avail_actions)
            _, _, last_val = network.apply(train_state.params, hstate, ac_in)
            last_val = last_val.squeeze()

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.global_done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config.gamma * next_value * (1 - done) - value
                    gae = (
                            delta
                            + config.gamma * config.gae_lambda * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, _):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        # RERUN NETWORK
                        train_mask = jax.lax.stop_gradient(traj_batch.train_mask)

                        _, pi, value = network.apply(
                            params,
                            init_hstate.squeeze(),
                            (traj_batch.obs, traj_batch.done, traj_batch.avail_actions)
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                                value - traj_batch.value
                        ).clip(-config.clip_eps, config.clip_eps)
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                                0.5 * jnp.maximum(value_losses, value_losses_clipped).mean(where=train_mask)
                        )

                        # CALCULATE ACTOR LOSS
                        logratio = log_prob - traj_batch.log_prob
                        ratio = jnp.exp(logratio)
                        gae = (gae - gae.mean(where=train_mask)) / (gae.std(where=train_mask) + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                                jnp.clip(
                                    ratio,
                                    1.0 - config.clip_eps,
                                    1.0 + config.clip_eps,
                                )
                                * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean(where=train_mask)
                        entropy = pi.entropy().mean(where=train_mask)

                        # debug
                        approx_kl = ((ratio - 1) - logratio).mean(where=train_mask)
                        clip_frac = jnp.mean(jnp.abs(ratio - 1) > config.clip_eps, where=train_mask)

                        total_loss = (
                                loss_actor
                                + config.vf_coef * value_loss
                                - config.ent_coef * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy, ratio, approx_kl, clip_frac)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, init_hstate, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, init_hstate, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)

                init_hstate = jnp.reshape(
                    init_hstate, (1, config.num_actors, -1)
                )
                batch = (init_hstate, traj_batch, advantages.squeeze(), targets.squeeze())
                permutation = jax.random.permutation(_rng, config.num_actors)

                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )

                minibatches = jax.tree.map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config.num_minibatches, -1]
                            + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )

                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, init_hstate.squeeze(), traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, init_hstate, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config.update_epochs
            )
            train_state = update_state[0]
            metric = traj_batch.info
            ratio_0 = loss_info[1][3].at[0, 0].get().mean()
            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)
            metric["loss"] = {
                "total_loss": loss_info[0],
                "value_loss": loss_info[1][0],
                "actor_loss": loss_info[1][1],
                "entropy": loss_info[1][2],
                "ratio": loss_info[1][3],
                "ratio_0": ratio_0,
                "approx_kl": loss_info[1][4],
                "clip_frac": loss_info[1][5],
            }
            metric["num_bc_envs"] = bc_anneal_mask.sum()
            metric["update_steps"] = update_steps
            metric["bc_anneal_rate"] = bc_anneal_schedule(update_steps)

            rng = update_state[-1]

            def callback(metric):
                wandb.log(
                    {
                        "returns": metric["returned_episode_returns"][-1, :].mean(),
                        "env_step": metric["update_steps"]
                                    * config.num_envs
                                    * config.num_steps,
                        **metric["loss"],
                        "num_bc_envs": metric["num_bc_envs"],
                        "bc_anneal_rate": metric["bc_anneal_rate"]
                    },
                    step=metric["update_steps"],
                )

            metric["update_steps"] = update_steps
            jax.experimental.io_callback(callback, None, metric)
            jax.debug.print("Returns: {x}", x=metric["returned_episode_returns"][-1, :].mean())
            jax.debug.print("Num BC envs: {x}", x=metric["num_bc_envs"])

            def save_last_policy(params, update_step):
                if config.save_path is None:
                    return

                # Delete the old best score files.
                delete_path_pattern = os.path.join(
                    config.save_path,
                    f"seed{config.seed}_step*_{config.num_players}p.*",
                )
                old_file = glob.glob(delete_path_pattern)
                if old_file:
                    os.remove(old_file[0])
                    os.remove(old_file[1])

                # Save new best score files.
                save_path_params = os.path.join(
                    config.save_path,
                    f"seed{config.seed}_step{update_step}_{config.num_players}p.safetensors",
                )
                save_params(params, save_path_params)
                # NOTE: We save config every time to have conventional loading of params & configs.
                save_path_config = os.path.join(
                    config.save_path,
                    f"seed{config.seed}_step{update_step}_{config.num_players}p.yaml",
                )
                OmegaConf.save(OmegaConf.create(config.to_dict()), save_path_config)

            jax.experimental.io_callback(save_last_policy, None, train_state.params, update_steps)

            update_steps = update_steps + 1
            runner_state = (train_state, env_state, last_obs, last_done, hstate, bc_hstate, bc_anneal_mask, rng)
            return (runner_state, update_steps), None

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, jnp.zeros((config.num_actors), dtype=bool), init_hstate, bc_init_hstate, init_bc_annealing_mask, _rng)
        runner_state, _ = jax.lax.scan(
            _update_step, (runner_state, 0), None, config.num_updates
        )
        return {"runner_state": runner_state}

    return train


@hydra.main(version_base=None, config_path="config", config_name="br_bc_lstm_2p")
def main(config):
    OmegaConf.set_struct(config, False)

    bc_policy_config = {"bc_policy": OmegaConf.load(config.bc_policy.config_path)}
    config = OmegaConf.merge(config, bc_policy_config)
    config.seed = bc_policy_config["bc_policy"].seed

    config = OmegaConf.to_container(config)
    config = ml_collections.ConfigDict(config)

    wandb.init(
        name=f"seed{config.seed}_{config.wandb_name}",
        project=config.project,
        config=config.to_dict(),
        mode=config.wandb_mode,
    )

    rng = jax.random.PRNGKey(config.seed)
    train_jit = jax.jit(make_train(config))
    _ = train_jit(rng)

if __name__ == "__main__":
    main()