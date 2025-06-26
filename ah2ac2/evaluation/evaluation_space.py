"""
Manages the lifecycle of evaluation environments for the AH2AC2 challenge.

This module provides classes and methods to interact with the evaluation server,
retrieve information about evaluation environments, and step through them.
"""
import logging
import os
import requests

from dotenv import load_dotenv
from enum import Enum
from typing import Union
from pydantic import BaseModel
from ah2ac2.evaluation.evaluation_environment import EvaluationEnvironment

logging.basicConfig(level=logging.INFO)

load_dotenv()  # take environment variables


class EnvironmentStatus(str, Enum):
    """Status of an evaluation environment."""
    TODO = "TODO"  #: The environment is scheduled but not yet active.
    ACTIVE = "ACTIVE"  #: The environment is currently active and being played.
    DONE = "DONE"  #: The environment has finished, and results may be available.


class EvaluationEnvironmentInfo(BaseModel):
    """
    Information about a single evaluation environment.

    Attributes:
        status: The current status of this environment.
        score: The final score achieved in this environment, if DONE; otherwise None.
        num_players: The number of players in this environment.
        candidate_controlling: A list of agent IDs that the candidate (submission) controls in this environment.
    """
    status: EnvironmentStatus
    score: Union[int, None]
    num_players: Union[int, None]
    candidate_controlling: list[str]


class EvaluationInfo(BaseModel):
    """
    Overall information about the evaluation process for a submission.

    Attributes:
        current_env: Information about the currently active environment, if any.
        all_envs: A list containing information for all environments associated with the submission key.
        human_ai_eval_done: A flag indicating whether the human-AI evaluation phase is complete for this submission.
    """
    current_env: Union[EvaluationEnvironmentInfo, None] = None
    all_envs: list[EvaluationEnvironmentInfo] = []
    human_ai_eval_done: bool | None = False


class EvaluationSpace:
    """
    Manages the interaction with the AH2AC2 evaluation server.

    This class allows users to request new evaluation environments,
    retrieve information about the ongoing evaluation, and manage
    the lifecycle of these environments.

    Attributes:
        submission_key: The unique key identifying the submission.
        evaluation_environment: The currently active evaluation environment, if any.
    """
    __DEFAULT_BASE_URL = "prod-proxy.ah2ac2.com"
    __BASE_URL_ENV_VAR = "AH2AC2_EVALUATION_BASE_URL"

    def __init__(self, submission_key: str) -> None:
        """
        Initializes the EvaluationSpace with a submission key.

        Args:
            submission_key: The unique key for the submission.
        """
        self.submission_key: str = submission_key
        self.evaluation_environment: Union[EvaluationEnvironment, None] = None

        self.__base_url: str = os.getenv(self.__BASE_URL_ENV_VAR, self.__DEFAULT_BASE_URL)

    @property
    def info(self) -> EvaluationInfo:
        """
        Retrieves information about the overall evaluation process.

        This includes the status of all environments associated with the submission key.

        Returns:
            An `EvaluationInfo` object containing details about the evaluation.
        """
        url = f"https://{self.__base_url}/evaluation-info/{self.submission_key}"
        return EvaluationInfo(**requests.get(url).json())

    def next_environment(self) -> EvaluationEnvironment:
        """
        Requests the next official evaluation environment from the server.

        Raises:
            Exception: If a previous environment is still active and not done.

        Returns:
            An `EvaluationEnvironment` instance for the next environment.
        """
        if self.evaluation_environment is not None and not self.evaluation_environment.done:
            raise Exception("Previous environment is still active.")

        url = f"{self.__base_url}/play-next?submission_key={self.submission_key}"
        self.evaluation_environment: EvaluationEnvironment = EvaluationEnvironment(url)
        return self.evaluation_environment

    def new_test_environment(self, num_players: int, candidate_position: list[int]) -> EvaluationEnvironment:
        """
        Requests a new test environment with a random agent.

        This is useful for testing and debugging the agent\'s interaction
        with the environment.

        Args:
            num_players: The number of players in the test environment.
            candidate_position: A list of positions (0-indexed) that the candidate agent will control.

        Returns:
            An `EvaluationEnvironment` instance for the test environment.
        """
        url = f"{self.__base_url}/play-random-agent?num_players={num_players}"
        for candidate_pos in candidate_position:
            url += f"&candidate_positions={candidate_pos}"
        url += f"&test_submission_key={self.submission_key}"

        self.evaluation_environment = EvaluationEnvironment(url)
        return self.evaluation_environment
