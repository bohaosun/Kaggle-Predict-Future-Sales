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
    data_pre.configuration(input_dir=input_dir, output_dir=output_dir)
    train_df, test_df = data_pre.transform()

    # feature engineering
    sale_features = SaleFeature()
    sale_features.configuration(shop_group_by_month=True, item_group_by_month=True, category_group_by_month=True)
    train_df, test_df = sale_features.transform(train_df, test_df)

    # modelling, XGBoost currently
    sale_model = SaleForecast()
    sale_model.configuration(data_resampling=False, weights=False, data_normalization=False)
    model = sale_model.train(train_df)

    # generate submission data file
    y_test = model.predict(test_df)
    pd.DataFrame(y_test).to_csv(os.path.join(output_dir, "y_test.csv"))
    submission_df = pd.concat([test_df['ID'], pd.DataFrame(y_test)], axis=1)
    submission_df.to_csv(os.path.join(output_dir, "submission.csv"))


if __name__ == '__main__':
    input_dir = args.input_dir
    output_dir = args.output_dir

    run_time_str = (datetime.utcnow() + timedelta(hours=8)).strftime("%Y%m%d%H%M")
    output_dir = os.path.join(output_dir, 'model_{}'.format(run_time_str))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    run_train(input_dir, output_dir)