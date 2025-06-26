# Datasets for AH2AC2

Welcome to the datasets section for the Ad-Hoc Human-AI Coordination Challenge (AH2AC2) where you can find details about the human gameplay data we provide for participation in AH2AC2.

## Overview

The AH2AC2 datasets come from [Hanab Live](https://hanab.live/), a popular online platform for playing Hanabi. As a part of the challenge, we provide open-source a dataset that consists of:

*   **1,858 two-player games**
*   **1,221 three-player games**

This deliberately limited dataset size is intended to encourage the development of data-efficient methods for human-AI coordination. It also preserves the integrity of the challenge.

## Navigating This Section

To help you get the most out of our datasets, we've organized the information into the following pages:

*   **[Data  Details](./details.md)**: Learn how to download the data, understand its format, and how to load the raw data.
*   **[Using the Dataset](./classes.md)**: We provide two utilities (`HanabiLiveGamesDataset` and `HanabiLiveGamesDataloader`) for easier data handling, batching, shuffling, and integration with [JaxMARL](https://github.com/FLAIROx/JaxMARL).
*   **[Tutorial: Unrolling Games](./tutorial.md)**: Find out how to use our datasets and dataloaders to unroll game trajectories within the JaxMARL framework.

We recommend starting with [Dataset Details](./details.md) to understand how to obtain and inspect the data.
