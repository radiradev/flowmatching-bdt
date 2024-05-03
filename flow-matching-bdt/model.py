from xgboost import XGBRegressor
from flow_matcher import ConditionalFlowMatcher
import numpy as np
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib



class Model:
    def __init__(
        self,
        n_flow_steps=50,
        n_noise_levels=100,
        max_depth=7,
        n_estimators=100,
        eta=0.3,
        tree_method="hist",
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
        self.n_noise_levels = n_noise_levels

    def xt_and_ut(self, x1):
        x0 = np.random.normal(size=x1.shape)
        t_levels = np.linspace(1e-3, 1, self.n_flow_steps)
        FM = ConditionalFlowMatcher()

        # input 
        X_train = np.zeros((self.n_flow_steps, x0.shape[0], x0.shape[1]))

        # to predict
        y_train = np.zeros((self.n_flow_steps, x0.shape[0], x0.shape[1]))
        for i in range(self.n_flow_steps):
            t = np.ones(x0.shape[0]) * t_levels[i]
            _, xt, ut = FM.sample_location_and_conditional_flow(x0, x1, t)

            X_train[i], y_train[i] = x0, ut

        return X_train, y_train
    
    def train_single(self, X_train, y_train, conditions=None):
        model = XGBRegressor(
            objective="reg:squarederror",
            max_depth=self.max_depth,
            n_estimators=self.n_estimators,
            eta=self.eta,
            tree_method=self.tree_method,
            reg_lambda=self.reg_lambda,
            reg_alpha=self.reg_alpha,
            subsample=self.subsample,
        )

        if conditions is not None:
            X_train = np.concatenate([X_train, conditions], axis=1)

        model.fit(X_train, y_train)
        return model
    
    def duplicate(self, X, k_times):
        return np.tile(X, (k_times, 1))

    def train(self, X_train, conditions=None):
        raise NotImplementedError("This method is not implemented yet.")

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
