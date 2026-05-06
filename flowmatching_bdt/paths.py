import numpy as np


def pad_t_like_x(t, x):
    """Reshape time vector t to match dimensions of x."""
    if isinstance(t, (float, int)):
        return t
    return np.reshape(t, (-1, *([1] * (x.ndim - 1))))


class ProbabilityPath:
    """Base class for probability paths in flow matching.

    A probability path defines an interpolation between a source sample
    ``x0`` and a data sample ``x1``. Subclasses must override:

    - :meth:`compute_mu_t` — the interpolated point ``xt`` at time ``t``.
    - :meth:`compute_flow` — the target velocity field ``u_t`` at
      ``(x0, x1, t, xt)`` that the flow-matching regressor learns to predict.

    Examples
    --------
    Subclassing with a custom path:

    >>> class MyPath(ProbabilityPath):
    ...     def compute_mu_t(self, x0, x1, t):
    ...         return ...
    ...     def compute_flow(self, x0, x1, t, xt):
    ...         return ...
    """

    def compute_mu_t(
        self, x0: np.ndarray, x1: np.ndarray, t: np.ndarray
    ) -> np.ndarray:
        """Compute the interpolated point along the path at time ``t``.

        Parameters
        ----------
        x0 : ndarray
            Source samples.
        x1 : ndarray
            Data samples (target).
        t : ndarray, shape (n_steps,) or scalar
            Time value(s) in ``[0, 1]``.

        Returns
        -------
        ndarray
            Interpolated point(s) ``xt``.
        """
        raise NotImplementedError

    def compute_flow(
        self,
        x0: np.ndarray,
        x1: np.ndarray,
        t: np.ndarray,
        xt: np.ndarray,
    ) -> np.ndarray:
        """Compute the target velocity field ``u_t`` at ``(x0, x1, t, xt)``.

        This is the regression target for flow matching: a
        :class:`FlowMatchingBDT` learns to predict ``u_t`` as a function of
        ``xt`` (and time, via per-step models).

        Parameters
        ----------
        x0 : ndarray
            Source samples.
        x1 : ndarray
            Data samples.
        t : ndarray, shape (n_steps,) or scalar
            Time value(s) in ``[0, 1]``.
        xt : ndarray
            The interpolated point(s) at time ``t`` (typically the output
            of :meth:`compute_mu_t`).

        Returns
        -------
        ndarray
            Target velocity ``u_t``.
        """
        raise NotImplementedError


class LinearPath(ProbabilityPath):
    """Linear interpolation between source and target.

    Defines ``mu_t = (1 - t) * x0 + t * x1`` with constant velocity
    ``u_t = x1 - x0``. This is the standard "conditional OT" path used in
    most flow-matching examples — straight-line trajectories from each
    ``x0`` to its paired ``x1``.

    Examples
    --------
    >>> import numpy as np
    >>> from flowmatching_bdt.paths import LinearPath
    >>> path = LinearPath()
    >>> x0 = np.array([[0.0, 0.0]])
    >>> x1 = np.array([[1.0, 1.0]])
    >>> path.compute_mu_t(x0, x1, np.array([0.5]))
    array([[[0.5, 0.5]]])
    >>> path.compute_flow(x0, x1, np.array([0.5]), None)
    array([[1., 1.]])
    """

    def compute_mu_t(
        self, x0: np.ndarray, x1: np.ndarray, t: np.ndarray
    ) -> np.ndarray:
        t = pad_t_like_x(t, x0)
        return t * x1 + (1 - t) * x0

    def compute_flow(
        self,
        x0: np.ndarray,
        x1: np.ndarray,
        t: np.ndarray,
        xt: np.ndarray,
    ) -> np.ndarray:
        del t, xt
        return x1 - x0


class PolynomialPath(ProbabilityPath):
    """Polynomial-schedule interpolation between source and target.

    Defines ``mu_t = (1 - t**k) * x0 + t**k * x1`` with velocity
    ``u_t = k * t**(k - 1) * (x1 - x0)``.

    Trajectories are still straight lines from ``x0`` to ``x1``, but the
    speed along each line is non-uniform. For ``k = 1`` this reduces to
    :class:`LinearPath`. For ``k > 1`` motion is slow near ``t = 0`` and
    fast near ``t = 1`` (samples linger in source noise, then accelerate
    onto the data manifold). For ``0 < k < 1`` the reverse.

    Parameters
    ----------
    k : float, default=2.0
        Polynomial exponent. Must be positive. Note that ``k < 1`` makes
        the velocity diverge as ``t -> 0``, which can amplify training
        targets at the first flow step.

    Examples
    --------
    >>> import numpy as np
    >>> from flowmatching_bdt.paths import PolynomialPath
    >>> path = PolynomialPath(k=2.0)
    >>> x0 = np.array([[0.0, 0.0]])
    >>> x1 = np.array([[1.0, 1.0]])
    >>> path.compute_mu_t(x0, x1, np.array([0.5]))  # 0.5**2 = 0.25 of the way
    array([[[0.25, 0.25]]])

    Plug into a flow-matching model:

    >>> from flowmatching_bdt import FlowMatchingBDT
    >>> model = FlowMatchingBDT(path=PolynomialPath(k=2.0))
    """

    def __init__(self, k: float = 2.0):
        self.k = k

    def compute_mu_t(
        self, x0: np.ndarray, x1: np.ndarray, t: np.ndarray
    ) -> np.ndarray:
        t = pad_t_like_x(t, x0)
        return (1 - t**self.k) * x0 + t**self.k * x1

    def compute_flow(
        self,
        x0: np.ndarray,
        x1: np.ndarray,
        t: np.ndarray,
        xt: np.ndarray,
    ) -> np.ndarray:
        del xt
        t = pad_t_like_x(t, x0)
        return self.k * t ** (self.k - 1) * (x1 - x0)
