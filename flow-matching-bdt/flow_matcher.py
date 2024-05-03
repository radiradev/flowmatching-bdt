import numpy as np

def pad_t_like_x(t, x):
    """Function to reshape the time vector t by the number of dimensions of x.
    
    Parameters
    ----------
    x : ndarray, shape (bs, *dim)
        represents the source minibatch
    t : ndarray, shape (bs)
    
    Returns
    -------
    t : ndarray, shape (bs, number of x dimensions)
    """
    if isinstance(t, (float, int)):
        return t
    return np.reshape(t, (-1, *([1] * (x.ndim - 1))))

class ConditionalFlowMatcher:
    """Base class for conditional flow matching methods using numpy."""

    def __init__(self, sigma=0.0):
        """Initialize the ConditionalFlowMatcher class with hyper-parameter sigma."""
        self.sigma = sigma

    def compute_mu_t(self, x0, x1, t):
        """Compute the mean of the probability path N(t * x1 + (1 - t) * x0, sigma)."""
        t = pad_t_like_x(t, x0)
        return t * x1 + (1 - t) * x0

    def compute_sigma_t(self, t):
        """Compute the standard deviation of the probability path N(t * x1 + (1 - t) * x0, sigma)."""
        del t  # Unused variable t in the context of this function
        return self.sigma

    def sample_xt(self, x0, x1, t, epsilon):
        """Draw a sample from the probability path N(t * x1 + (1 - t) * x0, sigma)."""
        mu_t = self.compute_mu_t(x0, x1, t)
        sigma_t = self.compute_sigma_t(t)
        sigma_t = pad_t_like_x(sigma_t, x0)
        return mu_t + epsilon * sigma_t

    def compute_conditional_flow(self, x0, x1, t, xt):
        """Compute the conditional vector field ut(x1|x0) = x1 - x0."""
        del t, xt  # Unused variables
        return x1 - x0

    def sample_noise_like(self, x):
        """Sample noise from a normal distribution N(0, 1) with the same shape as x."""
        return np.random.randn(*x.shape)

    def sample_location_and_conditional_flow(self, x0, x1, t=None, return_noise=False):
        """Compute the sample xt and the conditional vector field ut."""
        if t is None:
            t = np.random.rand(x0.shape[0])
        assert len(t) == x0.shape[0], "t must have the same batch size dimension as x0"
        
        eps = self.sample_noise_like(x0)
        xt = self.sample_xt(x0, x1, t, eps)
        ut = self.compute_conditional_flow(x0, x1, t, xt)
        if return_noise:
            return t, xt, ut, eps
        else:
            return t, xt, ut

    def compute_lambda(self, t):
        """Compute the lambda function."""
        sigma_t = self.compute_sigma_t(t)
        return 2 * sigma_t / (self.sigma**2 + 1e-8)