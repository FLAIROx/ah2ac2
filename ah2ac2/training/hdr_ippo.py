import distrax
import hydra
import jax
import flax.linen as nn
import jax.numpy as jnp
import ml_collections
import optax
import wandb
import numpy as np
import os
import glob

from jaxmarl.wrappers.baselines import save_params
from typing import List, NamedTuple
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from jaxmarl import make
from jaxmarl.wrappers.baselines import LogWrapper, load_params
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


class ActorCriticRNN(nn.Module):
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


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def make_train(config):
    env = make("hanabi", num_agents=config.num_players)

    config.num_actors = env.num_agents * config.num_envs
    config.num_updates = config.total_timesteps // config.num_steps // config.num_envs
    config.minibatch_size = config.num_actors * config.num_steps // config.num_minibatches
    config.num_test_actors = env.num_agents * config.num_test_envs

    env = LogWrapper(env)

    def linear_schedule(count):
        frac = (
            1.0 - (count // (config.num_minibatches * config.update_epochs)) / config.num_updates
        )
        return config.lr * frac

    def train(rng):
        # INIT NETWORK
        actor_params = load_params(config.network.weights_checkpoint_path)
        actor_module = ActorLstm(
            preprocessing_features=config.network.preprocessing_features,
            lstm_features=config.network.lstm_features,
            postprocessing_features=config.network.postprocessing_features,
            action_dim=config.network.action_dim,
            dropout_rate=config.network.dropout,
            activation_fn_name=config.network.act_fn,
        )
        network = ActorCriticRNN(
            actor=actor_module, critic_features=config.network.critic_features
        )

        init_x = (
            jnp.zeros((1, config.num_envs, env.observation_space(env.agents[0]).n)),
            jnp.zeros((1, config.num_envs)),
            jnp.zeros((1, config.num_envs, env.action_space(env.agents[0]).n)),
        )
        init_hstate = network.initialize_carry(batch_size=config.num_envs)
        rng, _rng = jax.random.split(rng)
        network_params = network.init(_rng, init_hstate, init_x)
        network_params["params"]["actor"] = actor_params
        # NOTE: Load the params again for the BC policy we use for regularization.
        behavioral_cloning_params = network.init(_rng, init_hstate, init_x)
        behavioral_cloning_params["params"]["actor"] = load_params(
            config.network.weights_checkpoint_path
        )

        lr = config.lr if not config.lr_linear_schedule else linear_schedule
        tx = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm), optax.adam(learning_rate=lr, eps=1e-5)
        )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config.num_envs)
        obsv, env_state = jax.vmap(env.reset, in_axes=0)(reset_rng)
        init_hstate = network.initialize_carry(config.num_actors)

        # TEST WITH GREEDY POLICY
        def get_greedy_metrics(rng, train_state):
            def _greedy_env_step(runner_state, _):
                train_state, env_state, last_obs, last_done, hstate, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                legal_actions = jax.vmap(env.get_legal_moves)(env_state.env_state)
                legal_actions = jax.lax.stop_gradient(
                    batchify(legal_actions, env.agents, config.num_test_actors)
                )

                obs_batch = batchify(last_obs, env.agents, config.num_test_actors)
                ac_in = (
                    obs_batch[np.newaxis, :],
                    last_done[np.newaxis, :],
                    legal_actions[np.newaxis, :],
                )
                hstate, pi, value = network.apply(train_state.params, hstate, ac_in)

                action = jnp.argmax(pi.probs, axis=-1)
                env_act = unbatchify(action, env.agents, config.num_test_envs, env.num_agents)
                env_act = jax.tree_map(lambda x: x.squeeze(), env_act)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config.num_test_envs)
                obsv, env_state, step_rewards, step_dones, step_infos = jax.vmap(
                    env.step, in_axes=0
                )(rng_step, env_state, env_act)

                done_batch = batchify(step_dones, env.agents, config.num_test_actors).squeeze()
                runner_state = (train_state, env_state, obsv, done_batch, hstate, rng)
                return runner_state, (step_infos, step_rewards, step_dones)

            rng, _rng = jax.random.split(rng)
            test_reset_rng = jax.random.split(_rng, config.num_test_envs)
            test_obsv, test_env_state = jax.vmap(env.reset, in_axes=0)(test_reset_rng)
            test_init_hstate = network.initialize_carry(config.num_test_actors)

            test_runner_state = (
                train_state,
                test_env_state,
                test_obsv,
                jnp.zeros((config.num_test_actors), dtype=bool),
                test_init_hstate,
                _rng,
            )
            test_runner_state, (final_infos, final_rewards, final_dones) = jax.lax.scan(
                _greedy_env_step, test_runner_state, None, 89
            )

            def first_episode_returns(rewards, dones):
                first_done = jax.lax.select(
                    jnp.argmax(dones) == 0.0, dones.size, jnp.argmax(dones)
                )
                first_episode_mask = jnp.where(jnp.arange(dones.size) <= first_done, True, False)
                first_episode_rewards = jnp.where(first_episode_mask, rewards, 0.0)
                return first_episode_rewards.sum()

            first_returns = jax.vmap(first_episode_returns, in_axes=1)(
                final_rewards["__all__"], final_dones["__all__"]
            )
            return {
                "mean": jnp.nanmean(first_returns),
                "max": jnp.nanmax(first_returns),
                "min": jnp.nanmin(first_returns),
                "std": jnp.nanstd(first_returns),
                "perfect": jnp.nansum(first_returns == 25),
                "zero": jnp.nansum(first_returns == 0),
            }

        # TRAIN LOOP
        def _update_step(update_runner_state, _):
            # COLLECT TRAJECTORIES
            runner_state, (update_steps, best_greedy_score) = update_runner_state

            def _env_step(runner_state, _):
                train_state, env_state, last_obs, last_done, hstate, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                legal_actions = jax.vmap(env.get_legal_moves)(env_state.env_state)
                legal_actions = jax.lax.stop_gradient(
                    batchify(legal_actions, env.agents, config.num_actors)
                )

                obs_batch = batchify(last_obs, env.agents, config.num_actors)
                ac_in = (
                    obs_batch[np.newaxis, :],
                    last_done[np.newaxis, :],
                    legal_actions[np.newaxis, :],
                )
                hstate, pi, value = network.apply(train_state.params, hstate, ac_in)

                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(action, env.agents, config.num_envs, env.num_agents)
                env_act = jax.tree_map(lambda x: x.squeeze(), env_act)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config.num_envs)
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=0)(
                    rng_step, env_state, env_act
                )
                info = jax.tree_map(lambda x: x.reshape((config.num_actors)), info)
                done_batch = batchify(done, env.agents, config.num_actors).squeeze()
                transition = Transition(
                    global_done=jnp.tile(done["__all__"], env.num_agents),
                    done=last_done,
                    action=action.squeeze(),
                    value=value.squeeze(),
                    reward=batchify(reward, env.agents, config.num_actors).squeeze(),
                    log_prob=log_prob.squeeze(),
                    obs=obs_batch,
                    info=info,
                    avail_actions=legal_actions,
                )
                runner_state = (train_state, env_state, obsv, done_batch, hstate, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config.num_steps
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, last_done, hstate, rng = runner_state
            last_obs_batch = batchify(last_obs, env.agents, config.num_actors)
            legal_actions = jnp.ones((config.num_actors, env.action_space(env.agents[0]).n))
            ac_in = (last_obs_batch[np.newaxis, :], last_done[np.newaxis, :], legal_actions)
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
                    delta = reward + config.gamma + next_value * (1 - done) - value
                    gae = delta + config.gamma * config.gae_lambda * (1 - done) * gae
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

            # TEST GREEDY POLICY BEFORE UPDATING THE NETWORK
            rng, _rng = jax.random.split(rng)
            greedy_metrics = get_greedy_metrics(_rng, train_state)

            # CHECKPOINT
            def save_best_policy(best_score, current_score, params):
                if config.save_path is None or best_score >= current_score:
                    return

                # Delete the old best score files.
                delete_path_pattern = os.path.join(
                    config.save_path,
                    f"seed{config.seed}_score{best_score:.3f}_hrkl{config.bc_policy_kl_weight}_{config.num_players}p.*",
                )
                old_file = glob.glob(delete_path_pattern)
                if old_file:
                    os.remove(old_file[0])
                    os.remove(old_file[1])

                # Save new best score files.
                score = current_score
                save_path_params = os.path.join(
                    config.save_path,
                    f"seed{config.seed}_score{score:.3f}_hrkl{config.bc_policy_kl_weight}_{config.num_players}p.safetensors",
                )
                save_params(params, save_path_params)
                # NOTE: We save config every time to have conventional loading of params & configs.
                save_path_config = os.path.join(
                    config.save_path,
                    f"seed{config.seed}_score{score:.3f}_hrkl{config.bc_policy_kl_weight}_{config.num_players}p.yaml",
                )
                OmegaConf.save(OmegaConf.create(config.to_dict()), save_path_config)

            jax.experimental.io_callback(
                save_best_policy,
                None,
                best_greedy_score,
                greedy_metrics["mean"],
                train_state.params,
            )

            def save_last_policy(current_score, params):
                if (
                    config.save_path is None
                    or "save_last_policy" not in config
                    or not config.save_last_policy
                ):
                    return

                # Delete the old best score files.
                delete_path_pattern = os.path.join(
                    config.save_path,
                    f"recent_seed{config.seed}_score*_hrkl{config.bc_policy_kl_weight}_{config.num_players}p.*",
                )
                old_file = glob.glob(delete_path_pattern)
                if old_file:
                    os.remove(old_file[0])
                    os.remove(old_file[1])

                # Save new best score files.
                score = current_score
                save_path_params = os.path.join(
                    config.save_path,
                    f"recent_seed{config.seed}_score{score:.3f}_hrkl{config.bc_policy_kl_weight}_{config.num_players}p.safetensors",
                )
                save_params(params, save_path_params)
                # NOTE: We save config every time to have conventional loading of params & configs.
                save_path_config = os.path.join(
                    config.save_path,
                    f"recent_seed{config.seed}_score{score:.3f}_hrkl{config.bc_policy_kl_weight}_{config.num_players}p.yaml",
                )
                OmegaConf.save(OmegaConf.create(config.to_dict()), save_path_config)

            jax.experimental.io_callback(
                save_last_policy,
                None,
                greedy_metrics["mean"],
                train_state.params,
            )

            # UPDATE NETWORK
            def _update_epoch(update_state, _):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        # RERUN NETWORK
                        ac_ins = (traj_batch.obs, traj_batch.done, traj_batch.avail_actions)
                        _, pi, value = network.apply(params, init_hstate, ac_ins)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                            -config.clip_eps, config.clip_eps
                        )
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                        # CALCULATE ACTOR LOSS
                        logratio = log_prob - traj_batch.log_prob
                        ratio = jnp.exp(logratio)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
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
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        # DEBUG
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clip_frac = jnp.mean(jnp.abs(ratio - 1) > config.clip_eps)

                        # REGULARIZATION TERM
                        _, pi_bc, _ = network.apply(behavioral_cloning_params, init_hstate, ac_ins)
                        pi_bc_kl_term = pi_bc.kl_divergence(pi).mean()
                        # total_loss = (1 - lambda) * IPPO_loss + lambda * BC_KL_term
                        ippo_loss = (
                            loss_actor + config.vf_coef * value_loss - config.ent_coef * entropy
                        ) * (1.0 - config.bc_policy_kl_weight)
                        regularization_loss = config.bc_policy_kl_weight * pi_bc_kl_term
                        total_loss = ippo_loss + regularization_loss

                        return total_loss, (
                            value_loss,
                            loss_actor,
                            entropy,
                            ratio,
                            approx_kl,
                            clip_frac,
                            pi_bc_kl_term,
                        )

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, init_hstate, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, init_hstate, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)

                batch = (init_hstate, traj_batch, advantages.squeeze(), targets.squeeze())
                permutation = jax.random.permutation(_rng, config.num_actors)

                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )

                minibatches = jax.tree.map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config.num_minibatches, -1] + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )

                train_state, total_loss = jax.lax.scan(_update_minbatch, train_state, minibatches)
                update_state = (train_state, init_hstate, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, init_hstate, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config.update_epochs
            )
            train_state = update_state[0]
            metric = traj_batch.info
            ratio_0 = loss_info[1][3].at[0, 0].get().mean()
            loss_info = jax.tree_map(lambda x: x.mean(), loss_info)
            metric["loss"] = {
                "total_loss": loss_info[0],
                "value_loss": loss_info[1][0],
                "actor_loss": loss_info[1][1],
                "entropy": loss_info[1][2],
                "ratio": loss_info[1][3],
                "ratio_0": ratio_0,
                "approx_kl": loss_info[1][4],
                "clip_frac": loss_info[1][5],
                "human_policy_kl": loss_info[1][6],
            }
            metric["greedy_metrics"] = greedy_metrics
            metric["update_steps"] = update_steps
            rng = update_state[-1]

            def callback(metric):
                wandb.log(
                    {
                        "returns": metric["returned_episode_returns"][-1, :].mean(),
                        "env_step": metric["update_steps"] * config.num_envs * config.num_steps,
                        **metric["loss"],
                        "greedy_metrics": metric["greedy_metrics"],
                    }
                )

            jax.experimental.io_callback(callback, None, metric)

            update_steps = update_steps + 1
            new_best_greedy_score = jnp.maximum(best_greedy_score, greedy_metrics["mean"])
            runner_state = (train_state, env_state, last_obs, last_done, hstate, rng)
            return (runner_state, (update_steps, new_best_greedy_score)), None

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obsv,
            jnp.zeros(config.num_actors, dtype=bool),
            init_hstate,
            _rng,
        )
        jax.lax.scan(_update_step, (runner_state, (0, 0.0)), None, config.num_updates)
        return {"runner_state": runner_state}

    return train


@hydra.main(version_base=None, config_path="config", config_name="hdr_ippo_2p")
def main(config):
    OmegaConf.set_struct(config, False)
    network_config = {"network": OmegaConf.load(config.network.config_checkpoint_path)}
    config = OmegaConf.merge(config, network_config)
    config.seed = network_config["network"].seed
    config = OmegaConf.to_container(config)
    config = ml_collections.ConfigDict(config)

    wandb.init(
        name=f"seed{config.seed}_{config.wandb_name}",
        project=config.wandb_project,
        config=config.to_dict(),
        mode=config.wandb_mode,
    )

    rng = jax.random.PRNGKey(config.seed)
    train_jit = jax.jit(make_train(config))

    _ = train_jit(rng)


if __name__ == "__main__":
    main()
