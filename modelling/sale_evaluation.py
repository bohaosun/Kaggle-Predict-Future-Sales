import numpy as np
import logging


class SaleEvaluation():
    def __init__(self):
        self.output_dir = None

    def configuration(self, output_dir):
        self.output_dir = output_dir

    def calculate_rmse(self, forecast_df, gt_df):
        forecast_arr = np.array(forecast_df['itm_cnt_month'])
        gt_arr = np.array(gt_df['itm_cnt_month'])
        rmse = np.sqrt(sum(np.square(forecast_arr - gt_arr)))
        print("RMSE for forecast result is {}".format(rmse))
        logging.info("RMSE for forecast result is {}".format(rmse))
        return rmse






