import numpy as np
import pandas as pd

class GBDTRegressor:
    def __init__(self, lib="lightgbm", params=None):
        self.lib = lib
        self.params = params or {}
        self.model = None

    def fit(self, X, y):
        if self.lib == "lightgbm":
            import lightgbm as lgb
            self.model = lgb.LGBMRegressor(**self.params)
            self.model.fit(X, y)
        else:
            from xgboost import XGBRegressor
            self.model = XGBRegressor(**self.params)
            self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
