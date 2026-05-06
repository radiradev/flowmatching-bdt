# Custom Estimators

`FlowMatchingBDT` accepts any scikit-learn-compatible regressor via the `estimator` parameter. Each flow step clones this estimator and wraps it in `MultiOutputRegressor` to handle multi-dimensional velocity prediction.

```python
from sklearn.datasets import make_moons
data, _ = make_moons(n_samples=500, noise=0.1, random_state=0)
```

## Using Decision Tree

```python
from sklearn.tree import DecisionTreeRegressor
from flowmatching_bdt import FlowMatchingBDT

model = FlowMatchingBDT(
    estimator=DecisionTreeRegressor(),
    n_flow_steps=5,
    n_duplicates=10,
)
model.fit(data)
```

## Using Extra Trees

```python
from sklearn.ensemble import ExtraTreesRegressor
from flowmatching_bdt import FlowMatchingBDT

model = FlowMatchingBDT(
    estimator=ExtraTreesRegressor(n_estimators=10),
    n_flow_steps=5,
    n_duplicates=10,
)
model.fit(data)
```

## Requirements for Custom Estimators

Any estimator you pass must:

1. **Implement `fit(X, y)`** — called with noised samples `X` and velocity targets `y`
2. **Implement `predict(X)`** — called during Euler integration to predict the velocity field
3. **Be compatible with `sklearn.base.clone`** — the estimator is cloned once per flow step

By default, the estimator is wrapped in [`MultiOutputRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputRegressor.html), so it only needs to support single-output regression. If your estimator natively handles multi-output (e.g. neural networks), set `multi_output=True` to skip the wrapper:

```python
model = FlowMatchingBDT(
    estimator=DecisionTreeRegressor(),
    multi_output=True,
    n_flow_steps=5,
    n_duplicates=10,
)
```

!!! tip
    The default `HistGradientBoostingRegressor` is a good choice for most tabular datasets. Try XGBoost or LightGBM if you need more control over hyperparameters or GPU acceleration.
