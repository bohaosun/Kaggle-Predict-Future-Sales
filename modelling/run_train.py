import logging
import numpy as np
import pandas as pd
import os
import sys
from modelling.config import args
from modelling.data_preprocessing import DataPrep
from modelling.feature_engineering import SaleFeature
from modelling.sale_modelling import SaleForecast
from modelling.sale_evaluation import SaleEvaluation
from datetime import datetime, timedelta


def run_train(input_dir, output_dir):
    # data pre-processing
    data_pre = DataPrep()
    data_pre.configuration(input_dir=input_dir)
    input_df = data_pre.transform()

    # feature engineering
    sale_features = SaleFeature()
    sale_features.configuration(shop_group_by_month=True, item_group_by_month=True, category_group_by_month=True)
    feature_df, y_df, feature_df_val, y_df_val = sale_features.transform(input_df)

    # modelling, XGBoost currently
    sale_model = SaleForecast()
    sale_model.configuration(data_resampling=False, weights=False, data_normalization=False)
    model = sale_model.train(feature_df, y_df)
    forecast_df_val = model.predict(feature_df_val)

    # model's forecast results' evaluation
    sale_eval = SaleEvaluation()
    sale_eval.configuration(output_dir)
    sale_eval.evaluate(forecast_df_val, y_df_val)

    # todo: there should be a function to generate submission data in csv format.


if __name__ == '__main__':
    input_dir = args.input_dir
    output_dir = args.output_dir

    run_time_str = (datetime.utcnow() + timedelta(hours=8)).strftime("%Y%m%d%H%M")
    output_dir = os.path.join(os.getcwd(), 'model_{}'.format(run_time_str))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    run_train(input_dir, output_dir)