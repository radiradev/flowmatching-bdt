import numpy as np


def pad_t_like_x(t, x):
    """Reshape time vector t to match dimensions of x."""
    if isinstance(t, (float, int)):
        return t
    return np.reshape(t, (-1, *([1] * (x.ndim - 1))))


class ProbabilityPath:
    """Base class for probability paths in flow matching."""

    def compute_mu_t(self, x0, x1, t):
        raise NotImplementedError

    def compute_flow(self, x0, x1, t, xt):
        raise NotImplementedError


class LinearPath(ProbabilityPath):
    """Linear interpolation: mu_t = t * x1 + (1 - t) * x0, u_t = x1 - x0."""

    def compute_mu_t(self, x0, x1, t):
        t = pad_t_like_x(t, x0)
        return t * x1 + (1 - t) * x0

    def compute_flow(self, x0, x1, t, xt):
        del t, xt
        return x1 - x0
