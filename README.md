# Multilayer Perceptron

Modular multilayer perceptron implementation for 42.

Supports:
- Arbitrary layers layout.
- Sigmoid and Softmax activations.
- MSE and Cross Entropy loss functions.
- Gradient Descent and Stochastic Gradient Descent.
- Early stopping with patience hyperparameter.

Example programs are given to be used on real fine needle aspiration biopsy data. The example model is trained to distinguish between cancerous and benign cells.

Usage:
- `python3 -m venv .venv`
- `pip install -r requirements.txt`
- `python3 prepare_data.py`
- `python3 train.py`
- `python3 predict.py`