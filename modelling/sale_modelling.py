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

    def train(self, data_df):
        # todo: grid search for hyper-parameters.
        train_df = data_df[data_df['date_block_num'] != 33]
        val_df = data_df[data_df['date_block_num'] == 33]
        x_train = train_df.drop(['item_cnt_mon'], axis=1)
        y_train = train_df[['item_cnt_mon']]
        x_val = val_df.drop(['item_cnt_mon'], axis=1)
        y_val = val_df[['item_cnt_mon']]

        xgb_model = xgb.XGBRegressor(
            max_depth=8,
            n_estimators=1000,
            min_child_weight=300,
            colsample_bytree=0.8,
            subsample=0.8,
            eta=0.3,
            seed=42)

        xgb_model.fit(
            x_train,
            y_train,
            eval_metric="rmse",
            eval_set=[(x_train, y_train), (x_val, y_val)],
            verbose=True,
            early_stopping_rounds=10)

        prediction = xgb_model.predict(x_val)
        val_rmse = self.calculate_rmse(prediction, y_val)
        print("The validation rmse is {}".format(val_rmse))
        return xgb_model

    @staticmethod
    def calculate_rmse(prediction, gt):
        rmse = np.sqrt(np.mean(np.power(prediction - np.array(gt), 2)))
        return rmse
