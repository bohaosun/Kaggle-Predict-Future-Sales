import pandas as pd
import os


class DataPrep:
    def __init__(self):
        self.input_dir = None
        self.remove_unique_feature = None

    def configuration(self, input_dir, remove_unique_feature):
        self.input_dir = input_dir
        self.remove_unique_feature = remove_unique_feature

    def read_data_files(self):
        test_df = pd.read_csv(os.path.join(self.input_dir, "test.csv"))
        sales_train_df = pd.read_csv(os.path.join(self.input_dir, "sales_train.csv"))
        items_df = pd.read_csv(os.path.join(self.input_dir, "items.csv"))
        submission_df = pd.read_csv(os.path.join(self.input_dir, "items.csv"))
        return test_df, items_df, sales_train_df

    def transform(self):
        test_df, items_df, sales_train_df = self.transform()

        # merge these three files into one file: sales_train_df
        sales_train_df = sales_train_df.merge(test_df, on=['shop_id', 'item_id'])
        # todo: there should be a seperate function to remove unique features.
        sales_train_df = sales_train_df.merge(items_df, on=['item_id'])[
            ['date_block_num', 'shop_id', 'item_id', 'item_price', 'item_cnt_day', 'ID', 'item_category_id']]

        # remove invalid data, <0
        # todo: there should be a function to check each feature and remove rows where negative values exist.
        sales_train_df = sales_train_df[sales_train_df['item_cnt_day'] > 0]
        return sales_train_df