import pandas as pd
import os
from modelling.config import args


class DataPrep:
    def __init__(self):
        self.input_dir = None
        self.remove_unique_feature = None

    def configuration(self, input_dir, remove_unique_feature=False):
        self.input_dir = input_dir
        self.remove_unique_feature = remove_unique_feature

    def read_data_files(self):
        test_df = pd.read_csv(os.path.join(self.input_dir, "test.csv"))
        sales_train_df = pd.read_csv(os.path.join(self.input_dir, "sales_train.csv"))
        items_df = pd.read_csv(os.path.join(self.input_dir, "items.csv"))
        return test_df, items_df, sales_train_df

    def transform(self):
        test_df, items_df, sales_train_df = self.read_data_files()

        # There are three sets of shops are the same, based on their names, like shop_id :[0,57], [1,58], [10,11]
        shop_id = sales_train_df['shop_id']
        shop_id[shop_id == 0] = 57
        shop_id[shop_id == 1] = 58
        shop_id[shop_id == 11] = 10
        sales_train_df['shop_id'] = shop_id

        # remove invalid value, item_cnt_day<0
        sales_train_df = sales_train_df[sales_train_df['item_cnt_day'] > 0]
        monthly_cnt = sales_train_df.groupby(['shop_id', 'item_id', 'date_block_num']).aggregate(
            {'item_price': 'mean', 'item_cnt_day': 'sum'})
        sales_train_df = monthly_cnt.reset_index(level=[0, 1, 2])
        sales_train_df.rename(columns={'item_cnt_day': 'item_cnt_mon'}, inplace=True)

        # merge these three files into one file: sales_train_df
        sales_train_df = pd.merge(sales_train_df, test_df, left_on=['shop_id', 'item_id'],
                                  right_on=['shop_id', 'item_id'], how='left')
        sales_train_df = sales_train_df.merge(items_df, on=['item_id'], how='left')
        sales_train_df = sales_train_df.drop(['item_name'], axis=1)

        test_df = test_df.merge(items_df, on=['item_id'], how='left')[
            ['shop_id', 'item_id', 'ID', 'item_category_id']]
        test_df['date_block_num'] = 34
        item_price = sales_train_df.groupby(['ID']).mean()['item_price']
        item_price = item_price.reset_index()
        test_df = test_df.merge(item_price, on=['ID'], how='left')

        output_train_path = os.path.join(args.output_dir, "output_basic_train.csv")
        output_test_path = os.path.join(args.output_dir, "output_basic_test.csv")
        sales_train_df.to_csv(output_train_path)
        test_df.to_csv(output_test_path)
        return sales_train_df, test_df