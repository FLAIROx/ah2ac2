import os
import copy
import hydra
import jax
import chex
import ml_collections
import optax
import wandb
import glob
import pprint
import jax.numpy as jnp
import flax.training.train_state
import flax.linen as nn

from flax import struct
from omegaconf import OmegaConf
from tqdm import tqdm
from typing import NamedTuple, TypedDict
from jaxmarl import make
from jaxmarl.environments.hanabi import hanabi_game
from jaxmarl.environments.hanabi.hanabi import HanabiEnv
from jaxmarl.wrappers.baselines import save_params
from ah2ac2.nn.multi_layer_lstm import MultiLayerLstm
from ah2ac2.datasets.dataset import HanabiLiveGamesDataset
from ah2ac2.datasets.dataloader import HanabiLiveGamesDataloader


def batchify(x, agent_list):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((len(agent_list), -1))


def unbatchify(x, agent_list):
    return {a: x[i] for i, a in enumerate(agent_list)}


"""Trajectory factory:"""


class StepCarry(NamedTuple):
    rng: chex.PRNGKey  # Random number generator.
    prev_obs: dict  # Observation at previous timestep.
    prev_env_state: hanabi_game.State  # Current state of the environment.


class Transition(NamedTuple):
    observations: chex.Array
    actions: chex.Array
    legal_actions: chex.Array


def make_trajectory_batcher(num_players: int):
    hand_size = 5 if num_players <= 3 else 4
    env: HanabiEnv = make("hanabi", hand_size=hand_size, num_agents=num_players)

    def get_trajectory_batch(
        rng: chex.PRNGKey,
        deck: chex.Array,
        actions: chex.Array,
    ):
        def _step(carry: StepCarry, step_actions: chex.Array):
            step_rng, _step_rng = jax.random.split(carry.rng)

            # Step env with selected actions.
            env_act = unbatchify(step_actions, env.agents)
            obs, env_state, reward, dones, infos = env.step_env(
                _step_rng, carry.prev_env_state, env_act
            )

            # Stay in terminal state if one is reached.
            reached_terminal = carry.prev_env_state.terminal
            env_state = jax.lax.cond(
                reached_terminal, lambda: carry.prev_env_state, lambda: env_state
            )
            obs = jax.lax.cond(reached_terminal, lambda: carry.prev_obs, lambda: obs)

            # Batchify observations, actions and legal actions.
            legal_moves = env.get_legal_moves(carry.prev_env_state)
            legal_moves_batched = batchify(legal_moves, env.agents)
            prev_obs_batched = batchify(carry.prev_obs, env.agents)
            act_batched = batchify(env_act, env.agents)

            # Create the carry/transition.
            new_carry = StepCarry(prev_obs=obs, prev_env_state=env_state, rng=step_rng)
            new_transition = Transition(
                observations=prev_obs_batched,  # (num_agents, obs_dim)
                actions=act_batched,  # (num_agents, 1)
                legal_actions=legal_moves_batched,  # (num_agents, num_moves)
            )
            return new_carry, new_transition

        rng, _rng = jax.random.split(rng)
        init_obs, init_env_state = env.reset_from_deck_of_pairs(deck)
        init_carry = StepCarry(prev_obs=init_obs, prev_env_state=init_env_state, rng=_rng)
        _, trajectory_batch = jax.lax.scan(_step, init_carry, actions)
        return trajectory_batch  # (num_steps, num_players, -1)

    return get_trajectory_batch


@jax.jit
def transform_trajectory_batch(trajectory_batch, game_len_mask):
    """
    Takes in a trajectory batch and normalizes it to be used for training/evaluation.

    Trajectory batch is expected to be a PyTree of arrays with the following structure:
    Objects have shape: (batch_size, max_game_len, num_players, features)
    We transform to: (batch_size * num_players, max_game_len, features)

    Game len mask is expected to be an array:
    Shape: (batch_size, max_game_len)
    We transform to: (batch_size * num_players, max_game_len)

    Essentially output is a batch of sequences where each sequence is a player's trajectory.
    For transformed trajectory batch:
    - First dimension is the batch size.
    - Second dimension is time dimension.
    - Third dimension is the feature size dimension.
    For game len mask:
    - First dimension is the batch size.
    - Second dimension is time dimension.
    """
    # Transform the trajectory batch data.
    obs_batch, act_batch, legal_moves_batch = jax.tree.map(
        lambda x: jnp.swapaxes(
            x,
            1,
            2,  # Swap max_game_len and num_players axes.
        ).reshape(-1, x.shape[1], x.shape[-1]),
        trajectory_batch,
    )
    # Repeat game len mask for each player.
    game_len_mask_batch = jnp.repeat(game_len_mask, trajectory_batch[0].shape[2], axis=0)
    return obs_batch, act_batch, legal_moves_batch, game_len_mask_batch


"""Training utilities:"""


class TrainState(flax.training.train_state.TrainState):
    key: jax.Array = struct.field(pytree_node=False)


class EvaluationMetrics(TypedDict):
    loss: float
    accuracy: float
    non_noop_accuracy: float


def learning_rate_schedule_fn(config):
    config.num_steps = config.num_epochs * config.num_steps_per_epoch
    return optax.linear_schedule(
        init_value=config.lr_initial,
        end_value=config.lr_final,
        transition_steps=config.num_steps
        * 0.9,  # Do 10% of the steps with the final learning rate.
    )


def make_model(config):
    return MultiLayerLstm(
        preprocessing_features=config.preprocessing_features,
        lstm_features=config.lstm_features,
        postprocessing_features=config.postprocessing_features,
        action_dim=config.action_dim,
        dropout_rate=config.dropout,
        activation_fn_name=config.act_fn,
    )


def get_train_state(rng, input_shape, config):
    rng, _rng = jax.random.split(rng)

    model = make_model(config)
    init_carry = model.initialize_carry(batch_size=input_shape[0])
    variables = model.init(_rng, x=jnp.ones(input_shape), carry=init_carry, training=False)
    params = variables["params"]

    schedule = learning_rate_schedule_fn(config)
    tx = optax.adamw(learning_rate=schedule)

    rng, dropout_key = jax.random.split(rng)
    state = TrainState.create(apply_fn=model.apply, params=params, key=dropout_key, tx=tx)
    return state, model


def masked_cross_entropy_loss(logits, act, mask):
    act = act.squeeze(axis=-1)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, act)
    masked_loss = loss * mask
    return masked_loss.sum() / mask.sum()


def compute_metrics(logits, actions, mask):
    actions_squeezed = actions.squeeze(axis=-1)
    correct = jnp.argmax(logits, -1) == actions_squeezed
    # Full accuracy
    correct_masked = correct * mask
    accuracy = correct_masked.sum() / mask.sum()
    # Non-noop accuracy
    non_noop_mask = (actions_squeezed != actions.max()) * mask
    correct_non_noop_masked = correct * non_noop_mask
    non_noop_accuracy = correct_non_noop_masked.sum() / non_noop_mask.sum()

    return EvaluationMetrics(
        loss=masked_cross_entropy_loss(logits, actions, mask),
        accuracy=accuracy,
        non_noop_accuracy=non_noop_accuracy,
    )


@jax.jit
def train_step(state, obs, act, legal_moves, game_len_mask, network_carry):
    def _loss_fn(params):
        _, output = state.apply_fn(
            {"params": params},
            x=obs,
            carry=network_carry,
            training=True,
            rngs={"dropout": jax.random.fold_in(state.key, state.step)},
        )
        output = output - (1 - legal_moves) * 1e10
        mean_loss = masked_cross_entropy_loss(output, act, game_len_mask)
        return mean_loss, output

    # Update parameters.
    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
    (loss, logits), grad = grad_fn(state.params)
    new_train_state = state.apply_gradients(grads=grad)

    # Compute metrics.
    metrics = compute_metrics(logits, act, game_len_mask)
    return new_train_state, metrics


"""Evaluation scripts"""


@jax.jit
def evaluate(network_carry, train_state, obs, act, game_len_mask, legal_moves):
    _, logits = train_state.apply_fn(
        {"params": jax.lax.stop_gradient(train_state.params)},
        x=obs,
        carry=network_carry,
        training=False,
    )
    logits = logits - (1 - legal_moves) * 1e10
    return compute_metrics(logits, act, game_len_mask)


def make_evaluate_in_self_play(num_players: int, config):
    hand_size = 5 if num_players <= 3 else 4
    env: HanabiEnv = make("hanabi", hand_size=hand_size, num_agents=num_players)

    def evaluate_in_self_play(key, agent_state):
        def _env_step(carry, _):
            step_rng, prev_env_state, prev_obs, agent_carry = carry

            # Get logits:
            batch_obs = batchify(prev_obs, env.agents)
            batch_obs = jnp.expand_dims(batch_obs, axis=1)
            new_agent_carry, batch_logits = agent_state.apply_fn(
                {"params": jax.lax.stop_gradient(agent_state.params)},
                x=batch_obs,
                carry=agent_carry,
                training=False,
            )
            batch_logits = batch_logits.squeeze()

            # Choose move from logits.
            batch_legal_moves = batchify(env.get_legal_moves(prev_env_state), env.agents)
            legal_logits = batch_logits - (1 - batch_legal_moves) * 1e10
            greedy_actions = jnp.argmax(legal_logits, axis=-1)
            actions = unbatchify(greedy_actions, env.agents)

            # Step env.
            new_rng, _rng = jax.random.split(step_rng)
            new_obs, new_env_state, step_rewards, step_dones, step_infos = env.step(
                _rng, prev_env_state, actions
            )

            return (new_rng, new_env_state, new_obs, new_agent_carry), (
                step_infos,
                step_rewards,
                step_dones,
            )

        initial_agent_carry = make_model(config).initialize_carry(env.num_agents)

        rng, reset_rng = jax.random.split(key)
        init_obs, init_env_state = env.reset(reset_rng)

        rng, scan_rng = jax.random.split(rng)
        _, (final_infos, final_rewards, final_dones) = jax.lax.scan(
            _env_step,
            (scan_rng, init_env_state, init_obs, initial_agent_carry),
            None,
            89,
        )

        def first_episode_returns(rewards, dones):
            first_done = jax.lax.select(jnp.argmax(dones) == 0.0, dones.size, jnp.argmax(dones))
            first_episode_mask = jnp.where(jnp.arange(dones.size) <= first_done, True, False)
            first_episode_rewards = jnp.where(first_episode_mask, rewards, 0.0)
            return first_episode_rewards.sum()

        def first_episode_returns_bomb_one(rewards, dones):
            first_done = jax.lax.select(jnp.argmax(dones) == 0.0, dones.size, jnp.argmax(dones))
            first_episode_mask = jnp.where(jnp.arange(dones.size) <= first_done, True, False)
            first_episode_rewards = jnp.where(first_episode_mask, rewards, 0.0)
            first_episode_rewards = nn.relu(first_episode_rewards)
            return first_episode_rewards.sum()

        def bombed_step(rewards, dones):
            first_done = jax.lax.select(jnp.argmax(dones) == 0.0, dones.size, jnp.argmax(dones))
            first_episode_mask = jnp.where(jnp.arange(dones.size) <= first_done, True, False)
            first_episode_rewards = jnp.where(first_episode_mask, rewards, 0.0)
            step_bombed = jnp.where(first_episode_rewards < 0.0, jnp.arange(dones.size), 0.0).sum()
            return step_bombed

        step_bombed = bombed_step(final_rewards["__all__"], final_dones["__all__"])
        bomb_one_returns = first_episode_returns_bomb_one(
            final_rewards["__all__"], final_dones["__all__"]
        )
        bomb_zero_returns = first_episode_returns(final_rewards["__all__"], final_dones["__all__"])
        return bomb_zero_returns, bomb_one_returns, step_bombed

    return evaluate_in_self_play


def compute_self_play_metrics(scores):
    return {
        "mean": jnp.mean(scores),
        "max": jnp.max(scores),
        "min": jnp.min(scores),
        "std": jnp.std(scores),
        "perfect": jnp.sum(scores == 25),
        "zero": jnp.sum(scores == 0),
    }


def compute_bomb_metrics(bombed_at_steps):
    return {
        "count": bombed_at_steps.size,
        "mean": jnp.mean(bombed_at_steps),
        "median": jnp.median(bombed_at_steps),
        "min": jnp.min(bombed_at_steps),
        "max": jnp.max(bombed_at_steps),
    }


"""Run scripts:"""


def train_and_evaluate(config: ml_collections.ConfigDict):
    key = jax.random.PRNGKey(config.seed)
    rng, _rng = jax.random.split(key)

    """Create all datasets and loaders."""
    data_aug_key = jax.random.fold_in(key, config.seed) if config.should_augment_data else None
    limited_data_regime_key = (
        jax.random.fold_in(key, config.seed)
        if config.limited_data_ratio or config.limited_data_regime_games
        else None
    )
    train_dataset = HanabiLiveGamesDataset(
        file=config.train_data_path,
        color_shuffle_key=data_aug_key,
        limited_data_regime_ratio=config.limited_data_ratio,
        limited_data_regime_key=limited_data_regime_key,
        limited_data_regime_games=config.limited_data_regime_games,
    )
    val_dataset = HanabiLiveGamesDataset(config.val_data_path)
    rng, _rng = jax.random.split(rng)
    train_loader = HanabiLiveGamesDataloader(train_dataset, config.batch_size, _rng)
    val_loader = HanabiLiveGamesDataloader(val_dataset)

    config.num_steps_per_epoch = len(train_loader)

    """JIT & VMAP everything."""
    get_trajectory_batch_vjit = jax.jit(
        jax.vmap(make_trajectory_batcher(train_dataset.num_players), in_axes=0)
    )
    evaluate_self_play_vjit = jax.jit(
        jax.vmap(
            make_evaluate_in_self_play(train_dataset.num_players, config),
            in_axes=(0, None),
        )
    )

    """Cache data for evaluation - we load all the data at once since it's small."""
    val_games_batch = next(iter(val_loader))  # This gets all the data at once.
    rng, _rng = jax.random.split(rng)
    val_trajectory_batch = get_trajectory_batch_vjit(
        jax.random.split(_rng, len(val_dataset)),
        val_games_batch.decks,
        val_games_batch.actions,
    )
    val_obs, val_act, val_legal_moves, val_game_len_mask = transform_trajectory_batch(
        val_trajectory_batch, val_games_batch.game_len_masks
    )

    """Create the network."""
    dummy_batch = next(iter(train_loader))
    rng, _rng_dummy_data, _rng_init_train_state = jax.random.split(rng, 3)
    dummy_trajectory_keys = jax.random.split(_rng_dummy_data, config.batch_size)
    dummy_trajectory_batch = get_trajectory_batch_vjit(
        dummy_trajectory_keys, dummy_batch.decks, dummy_batch.actions
    )
    dummy_obs = transform_trajectory_batch(dummy_trajectory_batch, dummy_batch.game_len_masks)[0]
    train_state, model = get_train_state(_rng_init_train_state, dummy_obs.shape, config)

    """Training loop."""
    complete_epoch_metrics = None
    for epochs in range(config.num_epochs):
        epoch_metrics = EvaluationMetrics(loss=0.0, accuracy=0.0, non_noop_accuracy=0.0)
        for batch in tqdm(train_loader):
            # Unpack the batch of games.
            decks = batch.decks
            actions = batch.actions
            game_len_masks = batch.game_len_masks

            # Get a batch of trajectories from decks and actions.
            rng, _rng = jax.random.split(rng)
            trajectory_rngs = jax.random.split(_rng, actions.shape[0])
            trajectory_batch = get_trajectory_batch_vjit(
                trajectory_rngs, decks, actions
            )  # (batch_size, num_steps, num_players, -1)
            obs, act, legal_moves, game_len_mask = transform_trajectory_batch(
                trajectory_batch, game_len_masks
            )  # (batch_size * num_players, num_steps, -1)

            # Update the train state.
            network_carry = model.initialize_carry(batch_size=obs.shape[0])
            train_state, metrics = train_step(
                state=train_state,
                obs=obs,
                act=act,
                legal_moves=legal_moves,
                game_len_mask=game_len_mask,
                network_carry=network_carry,
            )

            # Update epoch metrics.
            for k, v in metrics.items():
                epoch_metrics[k] += v

        # Normalize epoch train metrics.
        for k in epoch_metrics.keys():
            epoch_metrics[k] /= len(train_loader)

        # Evaluate on validation set.
        val_network_carry = model.initialize_carry(batch_size=val_obs.shape[0])
        val_metrics = evaluate(
            network_carry=val_network_carry,
            train_state=train_state,
            obs=val_obs,
            act=val_act,
            game_len_mask=val_game_len_mask,
            legal_moves=val_legal_moves,
        )

        # Evaluate on self play on fixed decks.
        fixed_decks_keys = jax.random.split(
            jax.random.PRNGKey(config.seed), config.self_play_eval_num_games
        )  # We are doing some redundant work here, but readability is better.
        self_play_scores_fixed_decks, self_play_bomb_one, bombed_info = evaluate_self_play_vjit(
            fixed_decks_keys, train_state
        )
        self_play_fixed_decks_metrics = compute_self_play_metrics(self_play_scores_fixed_decks)
        self_play_bomb_one_scores_fixed_decks_metrics = compute_self_play_metrics(
            self_play_bomb_one
        )

        # Compute some metrics for bombed games.
        bombed_at_steps = jnp.array([b for b in bombed_info.ravel() if b > 0.0])
        bomb_metrics = compute_bomb_metrics(bombed_at_steps)

        # Evaluate on self play on random decks, ignore bombing stats.
        # This is more like a confirmation that fixed decks are valid and relatively fine.
        # We expect the scores to move in a similar fashion as for fixed decks.
        rng, _rng = jax.random.split(rng)
        random_decks_key = jax.random.split(_rng, config.self_play_eval_num_games)
        self_play_scores_random_decks, _, _ = evaluate_self_play_vjit(
            random_decks_key, train_state
        )  # We ignore additional metrics for random decks.
        self_play_random_decks_metrics = compute_self_play_metrics(self_play_scores_random_decks)

        # Log metrics.
        if complete_epoch_metrics is None:
            prev_best_self_play_score = -1.0
            prev_best_val_acc = -1.0
        else:
            prev_best_self_play_score = complete_epoch_metrics["best_self_play_fixed_decks"]
            prev_best_val_acc = complete_epoch_metrics["best_val_acc"]
        best_self_play_fixed_decks = max(
            self_play_fixed_decks_metrics["mean"], prev_best_self_play_score
        )  # Either update the best score or keep the previous kept in logged metrics.
        best_val_acc = max(val_metrics["non_noop_accuracy"], prev_best_val_acc)
        complete_epoch_metrics = {
            "train": epoch_metrics,
            "val": val_metrics,
            "self_play_fixed_decks": self_play_fixed_decks_metrics,
            "self_play_fixed_decks_bomb_one": self_play_bomb_one_scores_fixed_decks_metrics,
            "self_play_fixed_decks_bomb_metrics": bomb_metrics,
            "self_play_random_decks": self_play_random_decks_metrics,
            "lr": learning_rate_schedule_fn(config)(train_state.step),
            "best_self_play_fixed_decks": best_self_play_fixed_decks,
            "best_val_acc": best_val_acc,
        }

        if config.use_wandb:
            wandb.log(complete_epoch_metrics)
        print(f"Epoch {epochs + 1}:")
        pprint.pprint(complete_epoch_metrics)

        # Save the model.
        if (
            config.save_path is not None
            and self_play_fixed_decks_metrics["mean"] > prev_best_self_play_score
        ):
            n_players = train_dataset.num_players

            # Delete the old best score files.
            delete_path_pattern = os.path.join(
                config.save_path,
                f"epoch*_seed{config.seed}_score{prev_best_self_play_score:.3f}_{n_players}p.*",
            )
            old_file = glob.glob(delete_path_pattern)
            if old_file:
                os.remove(old_file[0])
                os.remove(old_file[1])

            score = complete_epoch_metrics["best_self_play_fixed_decks"]
            save_path_params = os.path.join(
                config.save_path,
                f"epoch{epochs}_seed{config.seed}_score{score:.3f}_{n_players}p.safetensors",
            )
            save_params(train_state.params, save_path_params)
            # NOTE: We save config every time to have conventional loading of params & configs.
            save_path_config = os.path.join(
                config.save_path,
                f"epoch{epochs}_seed{config.seed}_score{score:.3f}_{n_players}p.yaml",
            )
            OmegaConf.save(OmegaConf.create(config.to_dict()), save_path_config)

        if (
            config.save_best_val_path is not None
            and val_metrics["non_noop_accuracy"] > prev_best_val_acc
        ):
            n_players = train_dataset.num_players

            # Delete the old best score files.
            delete_path_pattern = os.path.join(
                config.save_path,
                f"epoch*_seed{config.seed}_valacc{prev_best_val_acc:.3f}_{n_players}p.*",
            )
            old_file = glob.glob(delete_path_pattern)
            if old_file:
                os.remove(old_file[0])
                os.remove(old_file[1])

            val_acc = complete_epoch_metrics["best_val_acc"]
            save_path_params = os.path.join(
                config.save_best_val_path,
                f"epoch{epochs}_seed{config.seed}_valacc{val_acc:.3f}_{n_players}p.safetensors",
            )
            save_params(train_state.params, save_path_params)
            # NOTE: We save config every time to have conventional loading of params & configs.
            save_path_config = os.path.join(
                config.save_best_val_path,
                f"epoch{epochs}_seed{config.seed}_valacc{val_acc:.3f}_{n_players}p.yaml",
            )
            OmegaConf.save(OmegaConf.create(config.to_dict()), save_path_config)

    return complete_epoch_metrics


def tune(default_config):
    sweep_config = {
        "name": "LSTM-BC-Sweep",
        "method": "grid",
        "metric": {"name": "best_self_play_fixed_decks", "goal": "maximize"},
        "parameters": {
            "preprocessing_features": {"values": [[256], [512], [1024]]},
            "lstm_features": {"values": [[256], [512], [128, 128], [256, 256], [512, 512]]},
            "postprocessing_features": {"values": [[128], [256], [512], [256, 256]]},
            "batch_size": {"values": [128, 256]},
        },
    }

    def wrapped_make_train():
        wandb.init()
        config = copy.deepcopy(default_config)

        for k, v in dict(wandb.config).items():
            config[k] = v

        print(f"Running with parameters:\n{config}")
        train_and_evaluate(config)

    sweep_id = wandb.sweep(sweep_config, project=default_config.project_name)
    wandb.agent(sweep_id, wrapped_make_train, count=120)


def single_run(config: ml_collections.ConfigDict):
    if config.use_wandb:
        wandb.init(
            name=f"seed{config.seed}_{config.run_name}",
            config=config.to_dict(),
            project=config.project_name,
        )
    print(f"Running with parameters:\n{config}")
    train_and_evaluate(config)
    if config.use_wandb:
        wandb.finish()


def run_multiple_seeds(config: ml_collections.ConfigDict):
    run_config = copy.deepcopy(config)
    for i in range(config.num_seeds):
        # We will make the first run with the original config and initial seed.
        single_run(run_config)
        # Get the new seed and update the config.
        run_config = copy.deepcopy(run_config)
        run_config.seed = run_config.seed + 1


@hydra.main(version_base=None, config_path="config", config_name="bc_2p")
def main(config):
    config = OmegaConf.to_container(config)
    config = ml_collections.ConfigDict(config)
    if config.tune:
        tune(config)
    elif config.num_seeds > 1:
        run_multiple_seeds(config)
    else:
        single_run(config)


if __name__ == "__main__":
    main()
