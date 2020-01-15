import pandas as pd
import numpy as np
import xgboost as xgb


class SaleForecast():
    def __init__(self):
        self.data_resampling = None
        self.weights = None
        self.data_normalization = None

    def configuration(self, data_resampling, weights, data_normalization):
        self.data_resampling = data_resampling
        self.weights = weights
        self.data_normalization = data_normalization

    def train(self, feature_df, y_df):
        # todo: grid search for hyper-parameters.
        xgb_model = xgb.XGBRegressor(max_depth=8, n_estimators=1000)
        xgb_model.fit(feature_df, y_df)

        return xgb_model




