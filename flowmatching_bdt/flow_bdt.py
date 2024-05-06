from xgboost import XGBRegressor
from flowmatching_bdt.flow_matcher import ConditionalFlowMatcher
import numpy as np
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib
from tqdm import tqdm


# helper function
def duplicate(arr, n_times):
    if len(arr.shape) == 1:
        arr = arr[:, None]
    
    return np.tile(arr, (n_times, 1))


class FlowMatchingBDT:
    def __init__(
        self,
        n_flow_steps=50,
        n_duplicates=100,
        max_depth=7,
        n_estimators=100,
        eta=0.3,
        tree_method="approx",
        reg_lambda=0.0,
        reg_alpha=0.0,
        subsample=1.0,
    ):  
        # BDT params
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.eta = eta
        self.tree_method = tree_method
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.subsample = subsample

        # Flow params
        self.n_flow_steps = n_flow_steps
        self.n_duplicates = n_duplicates

    def xt_and_vt(self, x1):
        """
        Generate noised samples and corresponding velocity fields.
        These are used to train the BDTs at each flow step.

        Parameters
        ----------
        x1 : ndarray, shape (n_samples, n_features)
            The input samples.
        
        Returns
        -------
        X_train : ndarray, shape (n_flow_steps, n_samples, n_features)
            The noised samples.
        
        y_train : ndarray, shape (n_flow_steps, n_samples, n_features)
            The corresponding velocity fields.
        """

        n_samples, n_features = x1.shape
        
        x0 = np.random.normal(size=(n_samples, n_features))
        t_levels = np.linspace(1e-3, 1, self.n_flow_steps)
        
        X_train = np.zeros((self.n_flow_steps, x0.shape[0], x0.shape[1]))
        
        FM = ConditionalFlowMatcher()
        y_train = np.zeros((self.n_flow_steps, x0.shape[0], x0.shape[1]))
        for i in range(self.n_flow_steps):
            t = np.ones(x0.shape[0]) * t_levels[i]
            _, xt, ut = FM.sample_location_and_conditional_flow(x0, x1, t)

            X_train[i], y_train[i] = xt, ut

        return X_train, y_train
    
    def train_single(self, xt, vt, conditions=None):
        """
        Train a single BDT model.
        """

        model = XGBRegressor(
            objective="reg:squarederror",
            max_depth=self.max_depth,
            n_estimators=self.n_estimators,
            eta=self.eta,
            num_target = vt.shape[1],
            tree_method=self.tree_method,
            reg_lambda=self.reg_lambda,
            reg_alpha=self.reg_alpha,
            subsample=self.subsample,
        )

        if conditions is not None:
            xt = np.concatenate([xt, conditions], axis=1)

        # learn to predict the velocity field given a noised input
        model.fit(xt, vt)
        return model

    def train(self, x_train, conditions=None):
        """
        Train the BDT models for each flow step.

        Parameters:
        -----------
        x_train : ndarray, shape (n_samples, n_features)
            The input samples.
        
        conditions: ndarray, shape (n_samples, n_conditions) or None
            Optional conditions to be concatenated to the input samples.
        
        Returns:
        --------
        models : list of XGBRegressor
            The trained BDT models.
        """

        x1 = duplicate(x_train, self.n_duplicates)

        if conditions is not None:
            conditions = duplicate(conditions, self.n_duplicates)
        
        xt, vt = self.xt_and_vt(x1)

        def train_noise_level(noise_level):
            return self.train_single(xt[noise_level], vt[noise_level], conditions)

        # trains n_flow_steps models in parallel
        with tqdm_joblib(tqdm(desc="Training progress", total=self.n_flow_steps)) as progress_bar:
            models = Parallel(n_jobs=-1)(
                delayed(train_noise_level)(flow_step)
                for flow_step in range(self.n_flow_steps)
            )
        return models


    def fit(self, x_train, conditions=None):
        """
        Fit the generative mdoel to the input samples.

        Parameters:
        -----------
        x_train : ndarray, shape (n_samples, n_features)
            The input samples.
        
        conditions: ndarray, shape (n_samples, n_conditions) or None
            Optional conditions to be concatenated to the input samples.
        """

        self.n_features = x_train.shape[1] # set for predict method
        self.models = self.train(x_train, conditions=conditions) # n_flow_steps models


    def model_t(self, t, xt, conditions=None):
        flow_step = int(round(t * (self.n_flow_steps - 1)))

        if conditions is not None:
            xt = np.concatenate([xt, conditions], axis=1)
        
        return self.models[flow_step].predict(xt)

    def euler_solve(self, x0, conditions=None, n_steps=100):
        """
        Euler solve the ODE defined by the model_t method.

        Parameters:
        -----------
        x0 : ndarray, shape (n_samples, n_features)
            The initial conditions.
        
        conditions: ndarray, shape (n_samples, n_conditions) or None
            Optional conditions to be concatenated to the input samples.
        
        n_steps : int
            The number of steps to take in the Euler solve.
        """
        h = 1 / (n_steps - 1)
        x = x0
        t = 0

        for _ in range(n_steps - 1):
            x = x + h * self.model_t(t=t, xt=x, conditions=conditions)
            t += h

        return x

    def predict(self, num_samples, conditions=None):
        """
        Predict new samples using the fitted model.

        Parameters:
        -----------
        num_samples : int
            The number of samples to generate.
        
        conditions: ndarray, shape (num_samples, n_conditions) or None
            Optional conditions to be concatenated to the input samples.
        """
        x0 = np.random.normal(size=(num_samples, self.n_features))
        x1_preds = self.euler_solve(x0, conditions=conditions)

        return x1_preds
