import pandas as pd
import numpy as np
import pickle as pkl

class DataPreprocessing:
    def __init__(self, df=None, df_path=None, verbose=False, train_indices_path=None, test_indices_path=None,
                cbrt_standard_scaler_path = "../data/standard_scaler_cbrt_price_per_m.pkl"):
        if df_path is not None:
            self.df = pd.read_csv(df_path)
        else:
            self.df = df

        self.expected_columns = ['id', 'market', 'created_at_first', 'updated_at', 'district_lon', 
                                'district_lat', 'title', 'description', 'params', 'price', 'no_rooms',
                                'm', 'price_per_m', 'map_lon', 'map_lat']

        self.given_columns = self.df.columns.tolist()
        assert tuple(self.given_columns) == tuple(self.expected_columns)

        self.verbose = verbose

        self.train_indices = np.load(train_indices_path)
        self.test_indices = np.load(test_indices_path)

        self.X = self.df.drop(columns=['price', 'price_per_m', 'id'])
        self.Y = self.df[['price_per_m']]

        with open(cbrt_standard_scaler_path, 'rb') as f:
            self.cbrt_standard_scaler = pkl.load(f)

        # Transform the price_per_m column to cbrt_standardized_price_per_m
        # The method overrides the Y attribute
        self.Y = self.forward_transform_price_per_m(self.Y)

    def forward_transform_price_per_m(self, Y):

        Y = Y.copy() # Avoid changing the original dataframe
        # and avoid getting slicing errors

        Y.loc[:,'cbrt_price_per_m'] = Y['price_per_m'].apply(lambda x: np.cbrt(x) if x > 0 else 0)
        Y = Y.drop(columns=['price_per_m'])

        Y["standard_cbrt_price_per_m"] = self.cbrt_standard_scaler.transform(Y)
        Y = Y.drop(columns=['cbrt_price_per_m'])

        return Y

    def inverse_transform_price_per_m(self, Y):
        Y = Y.copy()
        Y["cbrt_price_per_m"] = self.cbrt_standard_scaler.inverse_transform(Y)
        Y["price_per_m"] = Y["cbrt_price_per_m"].apply(lambda x: x**3 if x > 0 else 0)
        Y = Y.drop(columns=['cbrt_price_per_m'])

        return Y
    
    def get_train_test_split(self, X):
        # X is a dataframe with the same indices as the original dataframe
        X_train = X.iloc[self.train_indices]
        X_test = X.iloc[self.test_indices]

        return X_train, X_test

if __name__ == "__main__":
    dp = DataPreprocessing(df_path="data/real_estate_ads_2022_10.csv", train_indices_path='data/train_indices.npy', 
                            test_indices_path='data/test_indices.npy',
                            cbrt_standard_scaler_path = "data/standard_scaler_cbrt_price_per_m.pkl")

    print(dp.Y)
