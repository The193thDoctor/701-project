# Hybrid Quantum-Classical Sentiment Classifier

A Quantum Machine Learning (QML) project that builds a hybrid quantum-classical model to perform sentiment analysis on the YELP Polarity dataset. The model leverages both classical neural network layers and quantum circuits to classify sentiments based on input data features.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Switching Encoding Methods](#switching-encoding-methods)
- [Project Structure](#project-structure)
- [Hyperparameters](#hyperparameters)
- [Layer-wise Training](#layer-wise-training)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Project Overview

This project implements a hybrid quantum-classical classifier designed to perform sentiment analysis on the YELP Polarity dataset. By integrating quantum circuits with classical neural network layers, the model aims to explore the potential advantages of quantum computing in natural language processing tasks.

## Features

- **Hybrid Architecture:** Combines quantum circuits with classical neural networks.
- **Flexible Data Encoding:** Supports both rotation-based encoding and amplitude encoding for input data.
- **Layer-wise Training:** Implements a layer-wise training strategy to enhance model stability and performance.
- **GPU Acceleration:** Utilizes PennyLane's Lightning GPU backend for efficient quantum circuit simulations.
- **Early Stopping:** Incorporates early stopping to prevent overfitting and reduce training time.
- **Modular Design:** Easily extendable to include more quantum layers or different encoding strategies.

## Installation

### Prerequisites

- Python 3.7 or higher
- [PyTorch](https://pytorch.org/) 1.7 or higher
- [PennyLane](https://pennylane.ai/) 0.20 or higher
- [PennyLane-Lightning](https://pennylane-lightning.readthedocs.io/en/latest/) for GPU acceleration
- Other Python packages as listed in `requirements.txt`

### Setup

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/quantum-sentiment-classifier.git
   cd quantum-sentiment-classifier

