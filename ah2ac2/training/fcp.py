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
import os
import glob
from typing import List, Optional
from omegaconf import OmegaConf


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, x):
        obs, dones, avail_actions = x
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
        unavail_actions = 1 - avail_actions
        action_logits = actor_mean - (unavail_actions * 1e10)
        pi = distrax.Categorical(logits=action_logits)

        critic = nn.Dense(512, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


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
    init_value = 1.0 if config.anneal_horizon == 0 else 0.0
    start_at_update_step = config.anneal_start // config.num_steps // config.num_envs
    end_at_update_step = config.anneal_horizon // config.num_steps // config.num_envs
    return optax.linear_schedule(
        init_value=init_value,
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

    dummy_init_x = (
        jnp.zeros((1, config.num_envs, env.observation_space(env.agents[0]).n)),
        jnp.zeros((1, config.num_envs)),
        jnp.zeros((1, config.num_envs, env.action_space(env.agents[0]).n))
    )
    dummy_partner_network = ActorCritic(env.action_space(env.agents[0]).n, config=config.fcp_population)
    rng, _rng = jax.random.split(jax.random.PRNGKey(0))
    partner_param_collection = []
    for fcp_partner_agent_weights_path in config.fcp_population.weights_paths:
        partner_params = dummy_partner_network.init(_rng, dummy_init_x)
        loaded_params = load_params(fcp_partner_agent_weights_path)
        partner_params["params"] = jax.lax.stop_gradient(loaded_params["params"])
        partner_param_collection.append(partner_params)
    config.fcp_population_size = len(partner_param_collection)

    env = LogWrapper(env)

    def linear_schedule(count):
        frac = 1.0 - (count // (config.num_minibatches * config.update_epochs)) / config.num_updates
        return config.lr * frac

    def train(rng):
        # INIT NETWORK
        network = ActorCriticRNN(env.action_space(env.agents[0]).n, config=config)
        partner_network = ActorCritic(env.action_space(env.agents[0]).n, config=config.fcp_population)

        rng, _rng = jax.random.split(rng)
        init_x = (
            jnp.zeros((1, config.num_envs, env.observation_space(env.agents[0]).n)),
            jnp.zeros((1, config.num_envs)),
            jnp.zeros((1, config.num_envs, env.action_space(env.agents[0]).n))
        )

        init_hstate = ScannedRNN.initialize_carry(config.num_envs, config.gru_hidden_dim)
        network_params = network.init(_rng, init_hstate, init_x)

        fcp_population_params = jax.tree.map(lambda *v: jnp.stack(v), *partner_param_collection)

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

            def _env_step(runner_state, _):
                train_state, env_state, last_obs, last_done, hstate, bc_anneal_mask, fcp_pop_agent_idxs, rng = runner_state

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

                def _compute_population_actions(policy_idx, obs_pop, obs_ld, obs_avail_act):
                    current_p = jax.tree.map(lambda x: x[policy_idx], fcp_population_params)
                    current_ac_in = (
                        obs_pop[np.newaxis, np.newaxis, :],
                        jnp.array([obs_ld])[np.newaxis, :],
                        obs_avail_act[np.newaxis, np.newaxis, :]
                    )
                    fcp_pi, _ = partner_network.apply(current_p, current_ac_in)
                    fcp_action = fcp_pi.sample(seed=_rng)
                    return fcp_action.squeeze()

                computed_fcp_actions = jax.vmap(_compute_population_actions)(
                    fcp_pop_agent_idxs,
                    obs_batch,
                    last_done,
                    avail_actions
                )
                action_pick_mask = _make_train_mask(bc_anneal_mask)
                action = jnp.where(action_pick_mask, action, computed_fcp_actions[np.newaxis, :])

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
                rng, _rng = jax.random.split(rng)
                new_fcp_pop_agent_idxs = jnp.where(
                    jnp.tile(done["__all__"], env.num_agents),
                    jax.random.randint(_rng, (config.num_actors,), 0, config.fcp_population_size),
                    fcp_pop_agent_idxs
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
                runner_state = (train_state, env_state, obsv, done_batch, hstate, new_bc_anneal_mask, new_fcp_pop_agent_idxs, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config.num_steps
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, last_done, hstate, bc_anneal_mask, fcp_agent_idxs, rng = runner_state
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

            jax.experimental.io_callback(callback, None, metric)

            def save_last_policy(params, update_step, returns):
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
                    f"seed{config.seed}_step{update_step}_ret{returns}_{config.num_players}p.safetensors",
                )
                save_params(params, save_path_params)
                # NOTE: We save config every time to have conventional loading of params & configs.
                save_path_config = os.path.join(
                    config.save_path,
                    f"seed{config.seed}_step{update_step}_ret{returns}_{config.num_players}p.yaml",
                )
                OmegaConf.save(OmegaConf.create(config.to_dict()), save_path_config)

            jax.experimental.io_callback(
                save_last_policy,
                None,
                train_state.params,
                update_steps,
                metric["returned_episode_returns"][-1, :].mean()
            )

            jax.debug.print("Returns: {x}", x=metric["returned_episode_returns"][-1, :].mean())

            update_steps = update_steps + 1
            runner_state = (train_state, env_state, last_obs, last_done, hstate, bc_anneal_mask, fcp_agent_idxs, rng)
            return (runner_state, update_steps), None

        rng, _rng = jax.random.split(rng)
        init_fcp_pop_idxs = jax.random.randint(_rng, (config.num_actors,), 0, config.fcp_population_size)
        runner_state = (train_state, env_state, obsv, jnp.zeros((config.num_actors), dtype=bool), init_hstate, init_bc_annealing_mask, init_fcp_pop_idxs, _rng)
        runner_state, _ = jax.lax.scan(
            _update_step, (runner_state, 0), None, config.num_updates
        )
        return {"runner_state": runner_state}

    return train



REQUIRED_SUBSTRINGS = ["_0_", "_100_", "_10000_", "_70000_"]
def has_required_substring(filename: str) -> bool:
    return any(substring in filename for substring in REQUIRED_SUBSTRINGS)


def find_first_yaml_file(directory_path: str) -> Optional[str]:
    pattern = os.path.join(directory_path, '*.yaml')
    yaml_files = glob.glob(pattern)

    # Filter files based on required substrings
    for file_path in yaml_files:
        filename = os.path.basename(file_path)
        if has_required_substring(filename):
            return file_path

    return None


def find_all_safetensors(directory_path: str) -> List[str]:
    pattern = os.path.join(directory_path, '*.safetensors')
    safetensor_files = glob.glob(pattern)
    filtered_files = [
        file_path for file_path in safetensor_files
        if has_required_substring(os.path.basename(file_path))
    ]
    return filtered_files


@hydra.main(version_base=None, config_path="config", config_name="fcp_2p")
def main(config):
    OmegaConf.set_struct(config, False)
    fcp_population_dir = config.fcp_population.population_directory

    yaml_config_path = find_first_yaml_file(fcp_population_dir)
    fcp_config = {"fcp_population": OmegaConf.load(yaml_config_path)}

    safetensors_files = find_all_safetensors(fcp_population_dir)
    fcp_config["fcp_population"]["weights_paths"] = safetensors_files

    print(f"Number of safetensors files: {len(safetensors_files)}")

    config = OmegaConf.merge(config, fcp_config)

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