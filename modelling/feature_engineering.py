import numpy as np
import pandas as pd


class SaleFeature:
    def __init__(self):
        self.category_group_by_month = None
        self.shop_group_by_month = None
        self.item_group_by_month = None

    def configuration(self, category_group_by_month, shop_group_by_month, item_group_by_month):
        self.category_group_by_month = category_group_by_month
        self.shop_group_by_month = shop_group_by_month
        self.item_group_by_month = item_group_by_month

    def transform(self, train_df, test_df):
        test_df['item_cnt_mon'] = 0
        data_df = pd.concat([train_df, test_df], axis=0)

        # add month, and year features
        data_df['month'] = np.mod(data_df['date_block_num'], 12)
        data_df['year'] = data_df['date_block_num'] // 12

        # add sin_month, cos_month to data_df
        data_df = self.add_sin_cos_month(data_df, 'month')

        data_df['item_cnt_mon'] = data_df['item_cnt_mon'].clip(0, 20)

        # add category monthly sale
        if self.category_group_by_month:
            data_df = self.add_monthly_sale_by_feature(data_df, 'item_category_id', ['category_monthly_sale',
                                                                                     'category_monthly_sale_shift1',
                                                                                     'category_monthly_sale_shift2'])

        # add shop monthly sale
        if self.shop_group_by_month:
            data_df = self.add_monthly_sale_by_feature(data_df, 'shop_id', ['shop_monthly_sale',
                                                                            'shop_monthly_shift1',
                                                                            'shop_monthly_shift2'])

        # add item monthly sale
        if self.item_group_by_month:
            data_df = self.add_monthly_sale_by_feature(data_df, 'item_id', ['item_monthly_sale', 'item_monthly_shift1',
                                                                            'item_monthly_shift2'])

        # todo: add price tendency, increasing or decreasing

        # use the last month as test set
        train_df = data_df[data_df['date_block_num'] < 34]
        test_df = data_df[data_df['date_block_num'] == 34]
        return train_df, test_df.drop(['item_cnt_mon'], axis=1)

    @staticmethod
    def add_sin_cos_month(data_df, month_feature_name):
        data_df['sin_mon'] = np.sin(data_df[month_feature_name])
        data_df['cos_mon'] = np.cos(data_df[month_feature_name])
        return data_df

    @staticmethod
    def add_monthly_sale_by_feature(data_df, groupby_feature, new_feature_name_ls):
        group_monthly_df = data_df.groupby([groupby_feature, 'date_block_num']).aggregate({'item_cnt_mon': 'mean'})
        for i in range(data_df.shape[1]):
            category = data_df.loc[i, 'item_category_id']
            month = data_df.loc[i, 'month']
            try:
                data_df.loc[i, new_feature_name_ls[0]] = group_monthly_df.loc[category, month][0]
            except:
                data_df.loc[i, new_feature_name_ls[0]] = np.nan
            try:
                data_df.loc[i, new_feature_name_ls[1]] = group_monthly_df.loc[category, max(month-1, 0)][0]
            except:
                data_df.loc[i, new_feature_name_ls[1]] = np.nan
            try:
                data_df.loc[i, new_feature_name_ls[2]] = group_monthly_df.loc[category, max(month-2, 0)][0]
            except:
                data_df.loc[i, new_feature_name_ls[2]] = np.nan
        return data_df

