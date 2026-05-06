from typing import Callable, Optional

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

from flowmatching_bdt.paths import LinearPath, ProbabilityPath


def duplicate(arr: np.ndarray, n_times: int) -> np.ndarray:
    if len(arr.shape) == 1:
        arr = arr[:, None]
    return np.tile(arr, (n_times, 1))


class FlowMatchingBDT(BaseEstimator):
    """Flow-matching generative model with gradient-boosted decision trees.

    Trains one regressor per flow step to predict the velocity field of a
    probability path between a source distribution and the data distribution.
    At inference time, integrates the learned vector field with Euler steps
    to draw new samples from random source noise.

    Parameters
    ----------
    n_flow_steps : int, default=50
        Number of discrete time steps along the probability path; one
        regressor is trained per step.
    n_duplicates : int, default=100
        Number of independent noise pairings per training point. Higher
        values give the regressor more Monte-Carlo coverage of the
        conditional distribution ``p(xt | x1)``.
    estimator : sklearn-compatible regressor, default=HistGradientBoostingRegressor()
        Base estimator cloned and fit at each flow step. Wrapped in
        ``MultiOutputRegressor`` unless ``multi_output=True``.
    multi_output : bool, default=False
        If ``True``, the estimator is assumed to handle multi-output
        regression natively and is **not** wrapped in
        ``MultiOutputRegressor``. Set this when using estimators like
        neural networks that already predict all output dimensions at once.
    source_distribution : callable, default=np.random.normal
        Function with signature ``f(size=...) -> ndarray`` returning samples
        from the source distribution at t=0.
    path : ProbabilityPath, default=LinearPath()
        Defines the interpolation between source and data via
        ``compute_mu_t`` and ``compute_flow``.

    Attributes
    ----------
    models : list of MultiOutputRegressor
        Trained per-step regressors, set by :meth:`fit`.
    n_features : int
        Dimensionality of the data, set by :meth:`fit`.

    Examples
    --------
    Unconditional generation on the two-moons dataset:

    >>> from sklearn.datasets import make_moons
    >>> from flowmatching_bdt import FlowMatchingBDT
    >>> data, _ = make_moons(n_samples=2000, noise=0.05, random_state=0)
    >>> model = FlowMatchingBDT()
    >>> model.fit(data)
    >>> samples = model.predict(num_samples=1000)

    Conditional generation by passing labels:

    >>> import numpy as np
    >>> data, labels = make_moons(n_samples=2000, noise=0.05, random_state=0)
    >>> model = FlowMatchingBDT()
    >>> model.fit(data, conditions=labels)
    >>> samples = model.predict(num_samples=1000, conditions=np.ones(1000))
    """

    def __init__(
        self,
        n_flow_steps: int = 50,
        n_duplicates: int = 100,
        estimator: RegressorMixin = HistGradientBoostingRegressor(),
        multi_output: bool = False,
        source_distribution: Callable[..., np.ndarray] = np.random.normal,
        path: ProbabilityPath = LinearPath(),
    ):
        self.n_flow_steps = n_flow_steps
        self.n_duplicates = n_duplicates
        self.estimator = estimator
        self.multi_output = multi_output
        self.source_distribution = source_distribution
        self.path = path

    def fit(
        self,
        x_train: np.ndarray,
        conditions: Optional[np.ndarray] = None,
    ) -> "FlowMatchingBDT":
        """Fit the generative model to data.

        Parameters
        ----------
        x_train : ndarray, shape (n_samples, n_features)
            Real data samples.
        conditions : ndarray, shape (n_samples,) or (n_samples, n_conditions), optional
            Conditioning features for conditional generation.

        Returns
        -------
        self : FlowMatchingBDT
            Fitted estimator.
        """
        self.n_features = x_train.shape[1]
        self.models = self._train(x_train, conditions=conditions)
        return self

    def predict(
        self,
        num_samples: int,
        conditions: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Generate new samples from the fitted model.

        Parameters
        ----------
        num_samples : int
            Number of samples to generate.
        conditions : ndarray, shape (num_samples,) or (num_samples, n_conditions), optional
            Conditioning features, one row per generated sample.

        Returns
        -------
        ndarray, shape (num_samples, n_features)
            Generated samples.
        """
        x0 = self.source_distribution(size=(num_samples, self.n_features))
        x1_preds = self._euler_solve(x0, conditions=conditions)
        return x1_preds

    def _xt_and_vt(self, x1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Generate noised samples and corresponding velocity targets.

        For each flow step, samples fresh source noise and computes the
        interpolated point ``xt`` along the probability path together with
        the target velocity ``vt = u_t(x0, x1, xt)``.

        Parameters
        ----------
        x1 : ndarray, shape (n_samples, n_features)
            Real data samples.

        Returns
        -------
        X_train : ndarray, shape (n_flow_steps, n_samples, n_features)
            Noised samples along the path.
        y_train : ndarray, shape (n_flow_steps, n_samples, n_features)
            Target velocity field values.
        """
        n_samples, n_features = x1.shape

        x0 = self.source_distribution(size=(self.n_flow_steps, n_samples, n_features))
        t_levels = np.linspace(1e-3, 1, self.n_flow_steps)

        X_train = self.path.compute_mu_t(x0, x1, t_levels)
        y_train = self.path.compute_flow(x0, x1, t_levels, X_train)

        return X_train, y_train

    def _train_single(
        self,
        xt: np.ndarray,
        vt: np.ndarray,
        conditions: Optional[np.ndarray] = None,
    ) -> MultiOutputRegressor:
        """Train one regressor for a single flow step.

        Parameters
        ----------
        xt : ndarray, shape (n_samples, n_features)
            Noised inputs at this flow step.
        vt : ndarray, shape (n_samples, n_features)
            Target velocity field values.
        conditions : ndarray, shape (n_samples,) or (n_samples, n_conditions), optional
            Conditioning features appended to ``xt``.

        Returns
        -------
        MultiOutputRegressor
            Fitted regressor mapping ``xt`` (and conditions) to ``vt``.
        """
        if self.multi_output:
            model = clone(self.estimator)
        else:
            model = MultiOutputRegressor(clone(self.estimator))
        if conditions is not None:
            xt = self._attach_conditions(xt, conditions)
        model.fit(xt, vt)
        return model

    def _train(
        self,
        x_train: np.ndarray,
        conditions: Optional[np.ndarray] = None,
    ) -> list[MultiOutputRegressor]:
        """Train one regressor per flow step in parallel.

        Parameters
        ----------
        x_train : ndarray, shape (n_samples, n_features)
            Real data samples.
        conditions : ndarray, shape (n_samples,) or (n_samples, n_conditions), optional
            Conditioning features for conditional generation.

        Returns
        -------
        list of MultiOutputRegressor
            One trained regressor per flow step, length ``n_flow_steps``.
        """
        x1 = duplicate(x_train, self.n_duplicates)

        if conditions is not None:
            conditions = duplicate(conditions, self.n_duplicates)

        xt, vt = self._xt_and_vt(x1)

        def train_noise_level(noise_level: int) -> MultiOutputRegressor:
            return self._train_single(xt[noise_level], vt[noise_level], conditions)

        with tqdm_joblib(tqdm(desc="Training progress", total=self.n_flow_steps)):
            models = Parallel(n_jobs=-1)(
                delayed(train_noise_level)(flow_step)
                for flow_step in range(self.n_flow_steps)
            )
        return models


    def _model_t(
        self,
        t: float,
        xt: np.ndarray,
        conditions: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Evaluate the learned velocity field at time ``t``.

        ``t`` is snapped to the nearest of the ``n_flow_steps`` trained
        models.

        Parameters
        ----------
        t : float
            Time in [0, 1].
        xt : ndarray, shape (n_samples, n_features)
            Current state.
        conditions : ndarray, shape (n_samples,) or (n_samples, n_conditions), optional
            Conditioning features to append to ``xt``.

        Returns
        -------
        ndarray, shape (n_samples, n_features)
            Predicted velocity at ``(t, xt)``.
        """
        flow_step = int(round(t * (self.n_flow_steps - 1)))
        if conditions is not None:
            xt = self._attach_conditions(xt, conditions)
        return self.models[flow_step].predict(xt)

    def _euler_solve(
        self,
        x0: np.ndarray,
        conditions: Optional[np.ndarray] = None,
        n_steps: int = 100,
    ) -> np.ndarray:
        """Integrate the learned ODE from t=0 to t=1 using Euler steps.

        Parameters
        ----------
        x0 : ndarray, shape (n_samples, n_features)
            Initial state at t=0, drawn from the source distribution.
        conditions : ndarray, shape (n_samples,) or (n_samples, n_conditions), optional
            Conditioning features.
        n_steps : int, default=100
            Number of Euler steps. Step size is ``1 / (n_steps - 1)``.

        Returns
        -------
        ndarray, shape (n_samples, n_features)
            State at t=1.
        """
        h = 1 / (n_steps - 1)
        x = x0
        t = 0

        for _ in range(n_steps - 1):
            x = x + h * self._model_t(t=t, xt=x, conditions=conditions)
            t += h

        return x

    @staticmethod
    def _attach_conditions(xt: np.ndarray, conditions: np.ndarray) -> np.ndarray:
        """Concatenate ``conditions`` to ``xt`` along the feature axis.

        Promotes 1D ``conditions`` to a column vector first.
        """
        if conditions.ndim == 1:
            conditions = conditions[:, None]
        return np.concatenate([xt, conditions], axis=1)
