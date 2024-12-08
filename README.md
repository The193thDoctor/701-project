# Hybrid Quantum-Classical Sentiment Classifier

A Quantum Machine Learning (QML) project that builds a hybrid quantum-classical model to perform sentiment analysis on the YELP Polarity dataset. The model compares the performance of quantum vs classical model 

## Table of Contents

- [Project Overview](#project-overview)
- [Usage](#usage)
- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Hyperparameters](#hyperparameters)
- [Layer-wise Training](#layer-wise-training)

## Project Overview

This project implements a hybrid quantum-classical classifier designed to perform sentiment analysis on the YELP Polarity dataset. By integrating quantum circuits with classical neural network layers, the model aims to explore the potential advantages of quantum computing in natural language processing tasks.

## Useful Files
data_loader.py         Pre-processing data and reduce dimension
classical_model.py     A standard neural network
quantum_model.py       Variational Quantum Circuits (VQC)
compare_plot.py        Plot the train/test accuracy and loss

### Prerequisites
- Other Python packages as listed in 'environment.yml'

### Setup

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/quantum-sentiment-classifier.git
   cd quantum-sentiment-classifier

