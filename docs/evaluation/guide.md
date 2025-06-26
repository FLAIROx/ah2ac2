# :joystick: Evaluation Usage Example: A Walkthrough of `bc_eval.py`

This guide provides a practical, step-by-step walkthrough of the `bc_eval.py` script. This script serves as a baseline and an example for interacting with the AH2AC2 evaluation system. We will break down its components and explain how it uses the `EvaluationSpace` and `EvaluationEnvironment` classes.

/// info | API Asynchronicity
Our evaluation API is asynchronous, so you'll see `async` and `await` used throughout `bc_eval.py` and your own interaction code.
///

## 1. Setup and Imports

The script begins by importing necessary libraries and modules:

```python
from __future__ import annotations # For type hinting with forward references

import asyncio
import logging
import time
import jax.numpy as jnp # For numerical operations

from typing import Callable, Any
from flax import core, struct # For creating structured PyTree-based classes
from jaxmarl.wrappers.baselines import load_params # Utility for loading model parameters
from omegaconf import OmegaConf # For configuration management

from ah2ac2.nn.multi_layer_lstm import MultiLayerLstm # The neural network model for the BC agent
# Key classes for AH2AC2 Evaluation
from ah2ac2.evaluation.evaluation_space import EvaluationSpace
from ah2ac2.evaluation.evaluation_environment import EvaluationEnvironment

logging.basicConfig(level=logging.INFO)

# --- IMPORTANT API Keys & Configuration ---
_TEST_API_KEY = "<API_KEY>" # Replace with your Test API Key
_EVALUATION_API_KEY = "<API_KEY>" # Replace with your Evaluation API Key
_NUM_CONCURRENT_EVAL_INSTANCES = 3 # How many evaluation loops to run in parallel
```

**Key Takeaways:**

*   Standard Python libraries for `asyncio`, `logging`, and `time` are used.
*   JAX and Flax are used for the example agent's neural network and parameter handling.
*   `OmegaConf` is used for managing agent configurations in the example.
*   Crucially, `EvaluationSpace` and `EvaluationEnvironment` are imported from the `ah2ac2` framework.

## 2. Agent Definition: `LstmBcAgent` and `AgentSpecification`

`bc_eval.py` defines an agent that uses a pre-trained LSTM model (Behavioral Cloning - BC). You will replace this with your own agent logic.

### Model: `LstmBcAgent`

This class encapsulates the example agent's state (parameters, hidden state) and its acting logic.

```python
class LstmBcAgent(struct.PyTreeNode):
    params: core.FrozenDict[str, Any]  # Model parameters
    hidden_state: Any                  # LSTM hidden state
    apply_fn: Callable = struct.field(pytree_node=False) # Function to apply the model

    def act(self, obs, legal_actions) -> tuple[int, LstmBcAgent]:
        # Preprocess observation, add batch & time dimensions
        obs = jnp.expand_dims(obs, axis=(0, 1))
        # Get new hidden state and action logits from the network
        new_hidden_state, batch_logits = self.apply_fn(
            {"params": self.params},
            x=obs,
            carry=self.hidden_state,
            training=False,
        )
        batch_logits = batch_logits.squeeze() # Remove batch & time dimensions

        # Mask illegal actions and choose the greedy action
        legal_logits = batch_logits - (1 - legal_actions) * 1e10
        greedy_action = jnp.argmax(legal_logits, axis=-1)
        # Return the chosen action and the updated agent (with new hidden state)
        return int(greedy_action), self.replace(hidden_state=new_hidden_state)

    @classmethod
    def create(cls, agent_config, params):
        # Initialize the MultiLayerLstm network
        network = MultiLayerLstm(
            preprocessing_features=agent_config.preprocessing_features,
            lstm_features=agent_config.lstm_features,
            postprocessing_features=agent_config.postprocessing_features,
            action_dim=agent_config.action_dim,
            dropout_rate=agent_config.dropout,
            activation_fn_name=agent_config.act_fn,
        )
        # Get the initial hidden state for the LSTM
        hidden_state = network.initialize_carry(batch_size=1)
        return cls(apply_fn=network.apply, params=params, hidden_state=hidden_state)
```

**Key Takeaways for Your Agent:**

*   Your agent needs an `act` method (or similar). This method should accept the observation and legal actions for the agent(s) it controls. It should return the chosen integer action(s) and its own updated state (if stateful).
*   It needs a way to be initialized (like the `create` method), potentially loading configurations or model parameters.
*   **You will replace `LstmBcAgent` with your own agent implementation.**

### Agent Configuration: `AgentSpecification`

This is a helper class in `bc_eval.py` to load the `LstmBcAgent`.

```python
class AgentSpecification:
    def __init__(self, name: str, path: str):
        self.name = name
        self.path = path # Path to model config and parameters

    def init_agent(self):
        agent_config = OmegaConf.load(f"{self.path}.yaml")
        agent_params = load_params(f"{self.path}.safetensors")
        return LstmBcAgent.create(agent_config=agent_config, params=agent_params)
```

/// tip | Adapting Agent Loading
**You should adapt this `AgentSpecification` class or use your own system for loading your agent(s), especially if your agent initialization or configuration management differs from the example.**
///

## 3. Test Environment Interaction: `interaction_test_2p` and `interaction_test_3p`

These functions are crucial for testing your integration using the **Test API Key**. They allow you to debug your agent against opponents making random moves before using your limited official evaluation attempts.

Let's analyze `interaction_test_2p` (`interaction_test_3p` is similar but tailored for 3-player scenarios):

```python
async def interaction_test_2p():
    # 1. Agent Setup: Define which pre-trained agent model to use for this test.
    agent_spec = AgentSpecification(
        "BC-1k-2p",
        "../models/bc_1k/epoch18_seed1_valacc0.467_2p" # Path to a 2-player model
    )
    num_players = 2 # Specify a 2-player game for this test.
    candidate_positions = [0] # Your submission will control the agent at index 0.

    # 2. Initialize EvaluationSpace (with Test API Key!)
    # This object manages your session with the evaluation server.
    eval_space = EvaluationSpace(_TEST_API_KEY)

    # 3. Get a Test Environment from EvaluationSpace
    # For test environments, you specify the number of players and your agent's position(s).
    env: EvaluationEnvironment = eval_space.new_test_environment(num_players, candidate_positions)
    # Note: env.info is not fully populated until after `await env.reset()`.

    # 4. Reset Environment (Crucial First Call to EvaluationEnvironment)
    # This establishes the WebSocket connection to the game instance and gets the initial state.
    obs_dict, legal_moves_dict = await env.reset()
    # Now, env.info is populated. This object contains details about the specific game instance.
    logging.info(f"Test (2p) - Game ID: {env.info.game_id}, Candidate controlling: {env.info.candidate_controlling}")

    # 5. Initialize Your Agent(s)
    # `env.info.candidate_controlling` is a list of agent_ids (e.g., ['agent_0']) 
    # that your submission is responsible for controlling in this game instance.
    my_agents = {agent_id: agent_spec.init_agent() for agent_id in env.info.candidate_controlling}

    done, score = False, None
    # 6. Main Interaction Loop: Continues until the game is done.
    while not done:
        actions_to_send = {}
        # For each agent your submission controls...
        for agent_id in env.info.candidate_controlling:
            agent_obs = obs_dict[agent_id]
            agent_legal_moves = legal_moves_dict[agent_id]
            
            # Get an action from your agent.
            action, my_agents[agent_id] = my_agents[agent_id].act(agent_obs, agent_legal_moves)
            actions_to_send[agent_id] = action # Store action for this agent_id.

        # 7. Step the Environment: Send all actions and get the next state.
        # This returns new observations, score, done status, and new legal moves.
        obs_dict, score, done, legal_moves_dict = await env.step(actions_to_send)
        logging.info(f"Test (2p) - Score: {score}, Done: {done}")
    
    logging.info(f"Test (2p) - Final Score: {score}")
    # The environment connection closes automatically when `done` is true, as handled by `env.step()`.
```

**Key Steps for Your Adaptation:**

*   **Initialize `EvaluationSpace`**: Use `EvaluationSpace(_TEST_API_KEY)`.
*   **Get Test Environment**: Call `eval_space.new_test_environment(num_players, candidate_positions)`.
    *   `num_players`: Total players in the test game (e.g., 2 or 3).
    *   `candidate_positions`: A `list[int]` of 0-indexed player positions your agent(s) will control (e.g., `[0]` or `[0, 1]`).
*   **`await env.reset()`**: This is the first interaction with `EvaluationEnvironment`. It returns:
    *   `obs_dict`: A dictionary mapping `agent_id_str` (e.g., 'agent_0') to its observation array for agents you control.
    *   `legal_moves_dict`: A dictionary mapping `agent_id_str` to a binary array indicating its legal actions.
*   **`env.info`**: After `reset()`, `env.info` (an `EnvironmentInfo` object) is available. Check `env.info.candidate_controlling` to see which specific agent IDs (e.g., `['agent_0']`, or `['agent_1', 'agent_2']`) your code needs to provide actions for.
*   **Agent Logic**: Initialize your agent(s) (one per ID in `env.info.candidate_controlling`). In the loop, use their `act` methods.
*   **`await env.step(actions_to_send)`**: Send a dictionary `actions_to_send = {agent_id_str: action_int}`.
*   **Loop until `done`**.

/// note
`interaction_test_3p` in `bc_eval.py` follows the same pattern but demonstrates iterating through different `candidate_positions` setups for 3-player games (e.g., controlling only player 0, or players 1 and 2). This is useful for testing team coordination if your agent controls multiple entities.
///

## 4. Official Evaluation: The `main()` function

This is the core loop for running official evaluation games using the **Evaluation API Key**.

```python
async def main():
    # 1. Setup Agent Specifications (e.g., for 2P and 3P games)
    # Your agent might need different configurations or models for different player counts.
    agent_spec_2p = AgentSpecification(
        "BC-1k-2p", "../models/bc_1k/epoch18_seed1_valacc0.467_2p"
    )
    agent_spec_3p = AgentSpecification(
        "BC-1k-3p", "../models/bc_1k/epoch22_seed0_valacc0.397_3p"
    )

    # 2. Initialize EvaluationSpace (with Evaluation API Key!)
    eval_space = EvaluationSpace(_EVALUATION_API_KEY)

    game_count = 1
    # 3. Loop While Evaluation is Not Done for Your Submission Key
    # `eval_space.info` fetches the overall status for your `_EVALUATION_API_KEY`.
    # The loop continues as long as the server indicates more games are pending.
    while not eval_space.info.human_ai_eval_done:
        start_time = time.time()

        # 4. Get Next Official Environment from Server
        # The server decides all game settings (player count, your role, etc.).
        # This call will raise an Exception if the previous environment obtained 
        # from *this specific eval_space instance* is still active (i.e., its game is 
        # not 'done' or its connection was not properly closed).
        env: EvaluationEnvironment = eval_space.next_environment()
        
        # 5. Reset Environment (same procedure as in test functions)
        obs_dict, legal_moves_dict = await env.reset()
        logging.info(f"Official Game ID: {env.info.game_id}, Players: {env.info.num_players}, Candidate controlling: {env.info.candidate_controlling}")

        # 6. Initialize Correct Agent(s) Based on `env.info`
        # You must adapt to the specific game parameters provided by the server for this game.
        if env.info.num_players == 2:
            agents = {c: agent_spec_2p.init_agent() for c in env.info.candidate_controlling}
        elif env.info.num_players == 3:
            agents = {c: agent_spec_3p.init_agent() for c in env.info.candidate_controlling}
        else:
            # This case should ideally not occur with official environments from the server.
            raise ValueError(f"Unexpected num_players ({env.info.num_players}) in official game. Game ID: {env.info.game_id}")

        # 7. Interaction Loop (identical to test functions)
        done, score = False, None
        while not done:
            actions_to_send = {}
            for agent_id in env.info.candidate_controlling:
                action, agents[agent_id] = agents[agent_id].act(obs_dict[agent_id], legal_moves_dict[agent_id])
                actions_to_send[agent_id] = action
            obs_dict, score, done, legal_moves_dict = await env.step(actions_to_send)

        logging.info(f"Official Game Score: {score}, Duration: {time.time() - start_time:.2f}s, Games in this session: {game_count}")
        game_count += 1
    
    logging.info("All official evaluation games for this submission key are complete!")
```

**Key Differences & Adaptations for Official Evaluation:**

*   **`EvaluationSpace(_EVALUATION_API_KEY)`**: Crucially, this uses your official, limited-use Evaluation API Key.
*   **`while not eval_space.info.human_ai_eval_done`**: This is the primary condition for the evaluation loop. `eval_space.info` (an `EvaluationInfo` object) checks the overall completion status for your submission key.
*   **`env = eval_space.next_environment()`**: This fetches the *next pre-configured official game* from the server. You do **not** specify player numbers or your agent's positions here; these are determined by the evaluation server based on the official schedule for your submission.
*   **Dynamic Agent Setup**: You **must** inspect `env.info` (after `reset()`) to understand the current game's configuration (e.g., `env.info.num_players`, `env.info.candidate_controlling`) and initialize or configure your agent(s) accordingly for that specific game.

/// warning | `eval_space.next_environment()` Safety
If the previous `EvaluationEnvironment` instance obtained from *this specific `eval_space` instance* was not properly concluded (i.e., its game loop didn't finish, `done` wasn't true, or its connection wasn't closed), `next_environment()` will raise an error. The `EvaluationEnvironment.step` method is designed to handle closing the connection when `done` is true. If you add custom error handling, ensure connections are closed in `finally` blocks to prevent issues.
///

## 5. Running Concurrently: `_run_all` and `_run_one_instance`

`bc_eval.py` demonstrates how to run multiple `main()` loops concurrently, which can speed up the evaluation process.

```python
async def _run_all() -> None:
    # It's good practice to run test interactions first.
    logging.info("Running 2P interaction test.")
    await interaction_test_2p()
    logging.info("Running 3P interaction test.")
    await interaction_test_3p()

    logging.info(f"Running official evaluation with {_NUM_CONCURRENT_EVAL_INSTANCES} concurrent instances.")

    async def _run_one_instance(idx: int):
        """Wrapper for one complete evaluation process (i.e., one `main()` loop)."""
        try:
            logging.info(f"Starting main instance {idx}.")
            # Each instance calls main(), which will create its own EvaluationSpace.
            await main()
            logging.info(f"Main instance {idx} finished successfully.")
        except Exception as exc:
            # Log errors from individual instances without stopping others.
            logging.error(f"Main instance {idx} crashed: {exc}", exc_info=True)

    # Create and run N concurrent tasks using asyncio.
    evaluation_tasks = [
        asyncio.create_task(_run_one_instance(i + 1), name=f"main_instance_{i + 1}")
        for i in range(_NUM_CONCURRENT_EVAL_INSTANCES)
    ]
    # Wait for all concurrent tasks to complete.
    await asyncio.gather(*evaluation_tasks, return_exceptions=True) # `return_exceptions=True` allows gathering all results/exceptions.
    logging.info("All concurrent evaluation instances have completed their tasks.")

if __name__ == '__main__':
    # CRITICAL: Ensure API keys are set before running!
    if _TEST_API_KEY == "<API_KEY>" or _EVALUATION_API_KEY == "<API_KEY>":
        logging.error("CRITICAL: Please replace placeholder API keys in the script with your actual keys provided upon registration before running.")
    else:
        # Start the entire asynchronous process.
        asyncio.run(_run_all())
```

**Concurrency Model in `bc_eval.py`:**

*   `_run_all` first executes the test functions sequentially (recommended).
*   Then, it creates `_NUM_CONCURRENT_EVAL_INSTANCES` asyncio tasks.
*   Each task executes `_run_one_instance(idx)`, which in turn calls `await main()`.
*   **Crucially, this means each concurrent instance effectively creates its own `EvaluationSpace` when `main()` is called within it.** These separate `EvaluationSpace` objects will independently poll the server for games using the same `_EVALUATION_API_KEY`. The server is responsible for managing the allocation of distinct games from your submission's quota to these concurrent requests.
*   `asyncio.gather` runs these tasks concurrently and waits for all of them to finish.
*   The `try...except` block within `_run_one_instance` is vital for robustness, ensuring that if one evaluation instance encounters an error, it doesn't halt the other concurrent instances.

/// danger | Concurrent Game Limits
The ability to run multiple games concurrently and the exact number of allowed concurrent instances (`_NUM_CONCURRENT_EVAL_INSTANCES`) may depend on server capacity. Ideally, you should run < 6 concurrent connections. With more connections, there is a higher risk of unwanted disconnects.
///


This detailed walkthrough of `bc_eval.py` should equip you to effectively integrate your agent and participate in the AH2AC2 evaluation. 