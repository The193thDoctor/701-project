# Quantum Machine Learning: A Preliminary Investigation

## Project Overview

This project implements a quantum machine learning classifier as a variational quantum circuit designed to perform sentiment analysis on the YELP Polarity dataset. We also build a classical neuron network as the baseline for comparison. By integrating quantum circuits with classical neural network layers, the model aims to explore the potential advantages of quantum computing in classification tasks.

### Useful Files

| File               | Description                              |
| ------------------ | ---------------------------------------- |
| data_loader.py     | Pre-processing data and reduce dimension |
| classical_model.py | A standard neural network                |
| quantum_model.py   | Variational Quantum Circuits (VQC)       |
| compare_plot.py    | Plot the train/test accuracy and loss    |

### Packages

- Python packages as listed in `environment.yml`

### Setup

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/The193thDoctor/701-project.git
   ```
2. **Run** **`data_loader.py` to preprocess data and store the dataloader**
3. **Run `classical_model.py` and `quantum_model.py` for model training**
4. **Run `compare_plot.py` to plot result**

### To turn on GPU (**Warning**: it's very slow)

1. Install `pennylane-lightning-gpu` as descripted in `environment.yml`
2. Modify the device settings in `data_loader.py` and `quantum_model.py`
