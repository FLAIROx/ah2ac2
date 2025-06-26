# :gear: Evaluation API Guide: `EvaluationSpace` & `EvaluationEnvironment`

This document provides a guide to the core classes you'll use to interact with the AH2AC2 evaluation system: `EvaluationSpace` and `EvaluationEnvironment`. For exhaustive details on all methods and attributes, please refer to the full [API Reference documentation](../api_reference/evaluation_space.md).

Our evaluation API is asynchronous, meaning interactions with the game server (like resetting an environment or taking a step) will use `async` and `await`.

```python
import asyncio
from ah2ac2.evaluation.evaluation_space import EvaluationSpace, EvaluationInfo
from ah2ac2.evaluation.evaluation_environment import EvaluationEnvironment, EnvironmentInfo

# You will need your API keys
_TEST_API_KEY = "YOUR_TEST_API_KEY_HERE" 
_EVALUATION_API_KEY = "YOUR_EVALUATION_API_KEY_HERE"
```

## `EvaluationSpace`

The `EvaluationSpace` class is your main entry point for an entire evaluation session associated with one of your API keys. It manages requesting game environments from the server and provides overall status information for your submission.

### Initialization

You initialize an `EvaluationSpace` by providing your submission API key.

```python
eval_space_test = EvaluationSpace(api_key=_TEST_API_KEY)
eval_space_official = EvaluationSpace(api_key=_EVALUATION_API_KEY)
```

/// danger | API Key Usage
- Use your **Test API Key** (`_TEST_API_KEY`) for all development and testing. It allows unlimited game environment creations against random opponents.
- Use your **Evaluation API Key** (`_EVALUATION_API_KEY`) **only** for official evaluation runs. This key has a limited number of uses.
///

### Key Methods and Properties

1.  **`new_test_environment(num_players: int, candidate_position: list[int]) -> EvaluationEnvironment`**
    *   Requests a new **test** game environment from the server.
    *   `num_players`: The total number of players in this test game (e.g., 2 or 3).
    *   `candidate_position`: A list of 0-indexed integers specifying which player positions your submission's agent(s) will control in this test game.
    *   Returns an `EvaluationEnvironment` instance for the created test game.

    ```python
    # Example: Create a 2-player test game where your agent is player 0
    test_env_2p: EvaluationEnvironment = eval_space_test.new_test_environment(
        num_players=2, 
        candidate_position=[0]
    )

    # Example: Create a 3-player test game where your agents are players 0 and 2
    test_env_3p: EvaluationEnvironment = eval_space_test.new_test_environment(
        num_players=3, 
        candidate_position=[0, 2]
    )
    ```

2.  **`next_environment() -> EvaluationEnvironment`**
    *   Requests the **next official** game environment from the server for your `_EVALUATION_API_KEY`.
    *   The server determines all game parameters (number of players, your agent's role, etc.) based on the official evaluation schedule for your submission.
    *   Returns an `EvaluationEnvironment` instance for the official game.
    *   **Important**: This method will raise an exception if the previous environment obtained from *this specific `eval_space` instance* is still considered active (i.e., its game is not `done` or its connection wasn't properly closed).

    ```python
    # Used with an EvaluationSpace initialized with the Evaluation API Key
    official_env: EvaluationEnvironment = eval_space_official.next_environment()
    ```

3.  **`info` (Property) `-> EvaluationInfo`**
    *   Retrieves overall status information for the submission key associated with this `EvaluationSpace`.
    *   Returns an `EvaluationInfo` object which contains:
        *   `current_env`: Information about the currently active environment (if any).
        *   `all_envs`: A list of `EvaluationEnvironmentInfo` for all environments (past and current).
        *   `human_ai_eval_done`: A boolean flag. If `True`, all official evaluation games for your submission key have been completed.

    ```python
    if eval_space_official.info.human_ai_eval_done:
        print("All official evaluation games are complete!")
    else:
        print("More official games to play.")
    ```

For more details, see the [EvaluationSpace API Reference](../api_reference/evaluation_space.md).

## `EvaluationEnvironment`

An `EvaluationEnvironment` instance represents a single, specific game. You obtain it from an `EvaluationSpace` and use it to interact directly with the game server (resetting the game, sending actions, receiving observations).

### Obtaining an Instance

Instances are returned by `eval_space.new_test_environment(...)` or `eval_space.next_environment()`.

### Key Methods and Properties

1.  **`await env.reset() -> tuple[dict, dict]`**
    *   Establishes the WebSocket connection to the specific game instance on the server.
    *   Resets the game to its initial state.
    *   Returns a tuple `(observations, legal_moves)`:
        *   `observations` (dict): `{agent_id_str: observation_array}`. Contains the initial observation for each agent ID your submission controls in *this* game.
        *   `legal_moves` (dict): `{agent_id_str: legal_moves_array}`. Contains the initial legal actions for each controlled agent.
    *   This method **must be called and awaited** before any other interaction with the environment (like `env.step()` or accessing `env.info`).

2.  **`info` (Property) `-> EnvironmentInfo`**
    *   Accessible **after** `await env.reset()` has successfully completed.
    *   Provides static information about *this specific* game environment.
    *   Returns an `EnvironmentInfo` (a NamedTuple) containing:
        *   `game_id` (int): A unique identifier for this game instance.
        *   `num_players` (int): The total number of players in this game.
        *   `candidate_controlling` (list[str]): A list of agent IDs (e.g., `['agent_0']` or `['agent_0', 'agent_1']`) that your submission is responsible for controlling in this game.
        *   `is_debug` (bool): Indicates if this is a debug/test environment.

    ```python
    obs_dict, legal_moves_dict = await env.reset()
    logging.info(f"Game ID: {env.info.game_id}")
    logging.info(f"My submission controls: {env.info.candidate_controlling}")
    logging.info(f"Total players in this game: {env.info.num_players}")
    ```

3.  **`await env.step(actions: dict) -> tuple[dict, float, bool, dict]`**
    *   Sends the chosen actions for your controlled agent(s) to the game server.
    *   `actions` (dict): `{agent_id_str: action_int}`. A dictionary mapping each agent ID your submission controls (from `env.info.candidate_controlling`) to the integer representing its chosen action.
    *   Advances the game by one step based on the actions of all players.
    *   Returns a tuple `(observations, score, done, legal_moves)`:
        *   `observations` (dict): New observations for your controlled agents.
        *   `score` (float): The current score of the game from the perspective of your submission.
        *   `done` (bool): `True` if the game has ended, `False` otherwise.
        *   `legal_moves` (dict): New legal actions for your controlled agents.
    *   If `done` is `True`, the server will typically close the WebSocket connection for this environment shortly after this payload is sent. `EvaluationEnvironment` also handles closing the connection internally when `done` is true.

### Core Interaction Loop

A typical game interaction sequence looks like this:

```python
async def play_single_game(env: EvaluationEnvironment):
    try:
        observations, legal_moves = await env.reset()
        print(f"Game started. ID: {env.info.game_id}, Controlling: {env.info.candidate_controlling}")

        # Initialize your agent(s) for the roles defined in env.info.candidate_controlling
        my_agents = {}
        for agent_id in env.info.candidate_controlling:
            my_agents[agent_id] = YourAgent.create(config_for_agent_id) # Or however you init

        done = False
        current_score = 0.0

        while not done:
            actions_to_send = {}
            for agent_id in env.info.candidate_controlling:
                obs_for_agent = observations[agent_id]
                legal_moves_for_agent = legal_moves[agent_id]
                
                # Get action from your specific agent instance
                action, my_agents[agent_id] = my_agents[agent_id].act(obs_for_agent, legal_moves_for_agent)
                actions_to_send[agent_id] = action
            
            # Send actions and get next state
            observations, current_score, done, legal_moves = await env.step(actions_to_send)
            print(f"Step taken. Score: {current_score}, Done: {done}")

        print(f"Game ID {env.info.game_id} finished. Final score: {current_score}")

    except Exception as e:
        print(f"Error during game {env.info.game_id if env._info else 'Unknown'}: {e}")

```

For more details, see the [EvaluationEnvironment API Reference](../api_reference/evaluation_environment.md).

## Summary

Together, `EvaluationSpace` and `EvaluationEnvironment` provide the necessary tools to:

1.  Connect to the evaluation system using your API key (`EvaluationSpace`).
2.  Request either test games with custom parameters or official evaluation games (`EvaluationSpace`).
3.  Manage individual game sessions: reset, get game info, send actions, and receive game state updates (`EvaluationEnvironment`).
4.  Determine when your official evaluation is complete (`EvaluationSpace.info`).

Refer to the [Evaluation Guide](./guide.md) for a complete, runnable example script (`bc_eval.py`) that demonstrates these classes in action.
