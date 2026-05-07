# Probability Paths

A probability path defines how samples are interpolated between the source distribution $x_0$ (e.g gaussian at $t{=}0$) and the target distribution $x_1$ (real data at $t{=}1$). The path determines both the training targets and the dynamics of generation.


## Built-in Paths

### LinearPath (default)

The simplest linear interpolation between source and target. Samples travel in straight lines from source $x_0$ to target $x_1$ at constant speed:

$$\begin{aligned}
\mu_t &= (1 - t) \, x_0 + t \, x_1 \\
u_t &= x_1 - x_0
\end{aligned}$$

```python
from flowmatching_bdt import FlowMatchingBDT
from flowmatching_bdt.paths import LinearPath

model = FlowMatchingBDT(path=LinearPath())
```

### PolynomialPath

A generalisation of `LinearPath` where the speed along each trajectory is non-uniform:

$$\begin{aligned}
\mu_t &= (1 - t^k) \, x_0 + t^k \, x_1 \\
u_t &= k \, t^{k-1} \, (x_1 - x_0)
\end{aligned}$$

For `k > 1`, samples linger near the source distribution early on and accelerate towards the data manifold. For `k = 1`, this is equivalent to `LinearPath`.

```python
from flowmatching_bdt import FlowMatchingBDT
from flowmatching_bdt.paths import PolynomialPath

model = FlowMatchingBDT(path=PolynomialPath(k=2.0))
```

## Creating a Custom Path

Subclass `ProbabilityPath` and implement two methods:

- `compute_mu_t(x0, x1, t)` — the interpolated point at time `t`
- `compute_flow(x0, x1, t, xt)` — the target velocity field

### Example: Cosine Schedule Path

A cosine schedule moves slowly near both endpoints and fastest at the midpoint:

```python
import numpy as np
from flowmatching_bdt.paths import ProbabilityPath, pad_t_like_x


class CosinePath(ProbabilityPath):
    """Cosine-schedule interpolation between source and target.

    mu_t = (1 - s(t)) * x0 + s(t) * x1
    where s(t) = (1 - cos(pi * t)) / 2

    The velocity is ds/dt * (x1 - x0) = (pi/2) * sin(pi * t) * (x1 - x0).
    """

    def compute_mu_t(self, x0, x1, t):
        t = pad_t_like_x(t, x0)
        s = (1 - np.cos(np.pi * t)) / 2
        return (1 - s) * x0 + s * x1

    def compute_flow(self, x0, x1, t, xt):
        t = pad_t_like_x(t, x0)
        dsdt = np.pi / 2 * np.sin(np.pi * t)
        return dsdt * (x1 - x0)
```

### Using Your Custom Path

```python
from sklearn.datasets import make_moons
from flowmatching_bdt import FlowMatchingBDT

data, _ = make_moons(n_samples=500, noise=0.05, random_state=0)

model = FlowMatchingBDT(path=CosinePath(), n_flow_steps=5, n_duplicates=10)
model.fit(data)
samples = model.predict(num_samples=500)
```

## Path Design Guidelines

All paths above follow the general form $\mu_t = (1 - s(t))\,x_0 + s(t)\,x_1$, where $s(t)$ is a schedule function that controls the interpolation speed. When designing a custom path, keep these properties in mind:

- **Boundary conditions**: $\mu_t$ should satisfy $\mu_0 = x_0$ (source) and $\mu_1 = x_1$ (target).
- **Smoothness**: The velocity $u_t$ should be finite everywhere in $(0, 1)$. Paths where $u_t$ diverges near $t = 0$ (e.g. `PolynomialPath` with $k < 1$) can produce very large training targets at the first flow step.
- **Monotonicity**: The schedule function $s(t)$ mapping $[0, 1] \to [0, 1]$ should be monotonically increasing.
