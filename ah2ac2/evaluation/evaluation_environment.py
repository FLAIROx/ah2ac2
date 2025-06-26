"""
Defines the evaluation environment for interacting with a single game instance.

This module provides the `EvaluationEnvironment` class, which handles
the connection to a game server, and allows for resetting the
environment, stepping through game states, and receiving observations and rewards.
"""
import json
import logging
import jax.numpy as jnp

from typing import NamedTuple, Optional
from websockets.asyncio.client import connect, ClientConnection

logging.getLogger().setLevel(logging.INFO)


class EnvironmentInfo(NamedTuple):
    """
    Contains static information about the evaluation environment.

    Attributes:
        num_players: The total number of players in the game.
        candidate_controlling: A list of agents that the candidate/submission controls.
        is_debug: A boolean flag indicating if the environment is in debug mode.
        game_id: A unique identifier for the game instance.
    """
    num_players: int
    candidate_controlling: list[str]
    is_debug: bool
    game_id: int


class EvaluationEnvironment:
    """
    Manages the interaction with a single game environment.

    This class provides methods to connect to a game server, reset the environment,
    send actions, and receive observations, rewards, and game status.

    Attributes:
        connection: The active connection, if established.
        done: A boolean indicating if the current game episode has finished.
        score: The final score of the game episode, available when `done` is True.
    """
    def __init__(self, connection_url: str):
        """
        Initializes the EvaluationEnvironment with a connection URL.
        """
        self.socket_url = f"wss://{connection_url}"

        self.connection: Optional[ClientConnection] = None
        self.done: bool = False
        self.score: Optional[float] = None

        self._info: EnvironmentInfo | None = None
        self.__candidate_controlling: list[str] | None = None

    @property
    def info(self) -> EnvironmentInfo:
        """
        Provides static information about the environment.

        Raises:
            RuntimeError: If the environment has not been reset yet (i.e., no info received).

        Returns:
            An `EnvironmentInfo` object containing details like number of players,
            controlled agents, debug status, and game ID.
        """
        if self._info is None:
            raise RuntimeError("First reset environment.")
        return self._info

    async def reset(self):
        """
        Establishes a connection with the server and retrieves the initial state.

        This method connects to the game server, receives the initial observation,
        legal moves, and environment information.

        Raises:
            RuntimeError: If the environment is already active (connection exists). It is not possible to have two connections to the server at the same time using the same object.
            RuntimeError: If the server returns an error during connection or initial payload.

        Returns:
            A tuple `(obs, legal_moves)`:
                - `obs`: A dictionary mapping agent IDs to their initial observations (as JAX arrays).
                - `legal_moves`: A dictionary mapping agent IDs to their initial legal moves (as JAX arrays).
        """
        if self.connection is not None:
            raise RuntimeError("Environment is already active.")

        self.connection = await connect(
            self.socket_url,
            open_timeout=60,
            ping_interval=10,
            ping_timeout=60,
        )
        # Discard handshake.
        _ = await self.connection.recv()
        logging.info("Connection with the server established.")

        # Get first payload.
        response = json.loads(await self.connection.recv())
        if "error_details" in response:
            raise RuntimeError(response["error_details"])

        self._info = EnvironmentInfo(**response["info"])
        self.__candidate_controlling = response["info"]["candidate_controlling"]

        obs, legal_moves = {}, {}
        for agent_name, values in response.items():
            if agent_name == "info":
                continue
            obs[agent_name] = jnp.array(values["obs"])
            legal_moves[agent_name] = jnp.array(values["legal_moves"])

        # Prepare env.
        self.done = False
        self.score = None
        return obs, legal_moves

    async def step(self, actions: dict[str, int]):
        """
        Sends actions to the environment and receives the next state.

        Args:
            actions: A dictionary mapping agent IDs (controlled by the candidate) to their chosen actions (integers).

        Raises:
            RuntimeError: If the game is already done.
            RuntimeError: If the environment has not been reset yet (no connection).
            RuntimeError: If the server returns an error in response to the actions.

        Returns:
            A tuple `(obs, score, done, legal_moves)`:
                - `obs`: A dictionary mapping agent IDs to their new observations (as JAX arrays).
                - `score`: The current score of the game (float). This is the final score if `done` is True.
                - `done`: A boolean indicating if the game episode has finished.
                - `legal_moves`: A dictionary mapping agent IDs to their new legal moves (as JAX arrays).
        """
        if self.done:
            raise RuntimeError("Game is done. Reset environment first.")
        if self.connection is None:
            raise RuntimeError("First reset environment.")

        await self.connection.send(json.dumps(actions))

        response = json.loads(await self.connection.recv())
        if "error_details" in response:
            raise RuntimeError(response["error_details"])

        obs, legal_moves = {}, {}
        for key, values in response.items():
            if key not in self.__candidate_controlling:
                continue
            obs[key] = jnp.array(values["obs"])
            legal_moves[key] = jnp.array(values["legal_moves"])

        self.done = response["done"]
        self.score = response["score"]

        if self.done:
            await self.connection.close()
            self.connection = None

        return obs, self.score, self.done, legal_moves
