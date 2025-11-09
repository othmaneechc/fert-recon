import numpy as np
from sklearn.linear_model import Ridge, ElasticNet


class LinearRegressor:
    def __init__(self, alpha=1.0, fit_intercept=True):
        self.model = Ridge(alpha=alpha, fit_intercept=fit_intercept)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


class ElasticNetRegressor:
    def __init__(self, alpha=1.0, l1_ratio=0.5, fit_intercept=True, max_iter=1000, tol=1e-4):
        self.model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
