# Scikit-learn API

`FlowMatchingBDT` follows the familiar scikit-learn estimator pattern — `fit` to train and `predict` to generate. This makes it easy to integrate into existing scikit-learn workflows.

## Fit / Predict Pattern

```python
from sklearn.datasets import make_moons
from flowmatching_bdt import FlowMatchingBDT

data, _ = make_moons(n_samples=500, noise=0.05, random_state=0)

model = FlowMatchingBDT(n_flow_steps=5, n_duplicates=10)
model.fit(data)  # returns self, like sklearn estimators

samples = model.predict(num_samples=500)
```

After calling `fit`, the model exposes learned attributes (with trailing underscores, per sklearn convention):

- `model.models` — the list of trained per-step regressors
- `model.n_features` — dimensionality of the training data

## Constructor Parameters

All model configuration is passed through the constructor, just like scikit-learn:

```python
from sklearn.ensemble import HistGradientBoostingRegressor
from flowmatching_bdt import FlowMatchingBDT

model = FlowMatchingBDT(
    n_flow_steps=50,       # number of discrete time steps
    n_duplicates=100,      # noise pairings per data point
    estimator=HistGradientBoostingRegressor(max_depth=5),
)
```

## Conditional Generation

Pass conditioning labels or features via the `conditions` parameter in both `fit` and `predict`:

```python
import numpy as np
from sklearn.datasets import make_moons
from flowmatching_bdt import FlowMatchingBDT

data, labels = make_moons(n_samples=500, noise=0.05, random_state=42)

model = FlowMatchingBDT(n_flow_steps=5, n_duplicates=10)
model.fit(data, conditions=labels)

# generate samples conditioned on class 1
samples = model.predict(
    num_samples=500,
    conditions=np.ones(500),
)
```

## Using sklearn's `clone`

Since `FlowMatchingBDT` stores all init parameters as attributes, it works with `sklearn.base.clone`:

```python
from sklearn.base import clone
from flowmatching_bdt import FlowMatchingBDT

model = FlowMatchingBDT(n_flow_steps=30)
model_copy = clone(model)  # fresh copy with same parameters
```
