# Torch MLP Estimator

You can use a PyTorch neural network as the flow-matching estimator by wrapping it with scikit-learn's `RegressorMixin` and `BaseEstimator`. This lets you mix deep learning with the same `fit`/`predict` API.

## Defining a Scikit-learn-Compatible MLP

```python
import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin


class TorchMLPRegressor(BaseEstimator, RegressorMixin):
    """A simple MLP regressor compatible with scikit-learn's API.

    Parameters
    ----------
    hidden_dim : int
        Number of units in each hidden layer.
    n_layers : int
        Number of hidden layers.
    lr : float
        Learning rate for Adam.
    n_epochs : int
        Number of training epochs.
    batch_size : int
        Mini-batch size for training.
    """

    def __init__(
        self,
        hidden_dim=32,
        n_layers=3,
        lr=1e-3,
        n_epochs=10,
        batch_size=12,
    ):
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size

    def fit(self, X, y):
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).reshape(len(y), -1)

        n_in, n_out = X.shape[1], y.shape[1]

        # build MLP
        layers = [nn.Linear(n_in, self.hidden_dim), nn.ReLU()]
        for _ in range(self.n_layers - 1):
            layers += [nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(self.hidden_dim, n_out))
        self.model_ = nn.Sequential(*layers)

        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True,
        )

        self.model_.train()
        for _ in range(self.n_epochs):
            for xb, yb in loader:
                loss = nn.functional.mse_loss(self.model_(xb), yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return self

    def predict(self, X):
        self.model_.eval()
        X = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            return self.model_(X).numpy()
```

## Using It with FlowMatchingBDT

```python
from sklearn.datasets import make_moons
from flowmatching_bdt import FlowMatchingBDT

data, _ = make_moons(n_samples=500, noise=0.05, random_state=0)

model = FlowMatchingBDT(
    estimator=TorchMLPRegressor(hidden_dim=32, n_epochs=10),
    multi_output=True,  # skip MultiOutputRegressor wrapper
    n_flow_steps=50,
    n_duplicates=10,
)
model.fit(data)
samples = model.predict(num_samples=500)
```

## Why This Works

The key ingredients that make any estimator compatible with `FlowMatchingBDT` are:

1. **Inherit from `BaseEstimator` and `RegressorMixin`** — this gives you `get_params`/`set_params` for free, which `sklearn.base.clone` needs to create fresh copies at each flow step.

2. **Store all constructor arguments as attributes** — `BaseEstimator` inspects `__init__` parameters to implement cloning. Every `__init__` argument must be stored as `self.<name>` with the exact same name.

3. **Implement `fit(X, y)` and `predict(X)`** — the standard scikit-learn interface.

!!! tip
    Setting `multi_output=True` tells `FlowMatchingBDT` to use your estimator directly instead of wrapping it in `MultiOutputRegressor`. This is the right choice for neural networks and any other estimator that natively handles multi-dimensional output.

