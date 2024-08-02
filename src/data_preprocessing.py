import pandas as pd
import numpy as np
import pickle as pkl

from sklearn.preprocessing import StandardScaler

from src.params_parser import ParamsParser
from src.outlier_imputer import OutlierImputer

class DataPreprocessing:
    def __init__(self, df=None, df_path=None, verbose=False, train_indices_path=None, test_indices_path=None,
                transform_original_features_flag=True, remove_label_outliers_flag=False,
                params_parser_args={}, get_params_from_params=True, get_tfidf_embeddings_flag=True,
                get_bert_embeddings_flag=True, get_textual_features_flag=True,
                transform_time_features_flag=True, transform_cyclic_features_flag=True,
                remove_nonfrequent_params_flag=True, remove_missing_params_flag=True,
                drop_unnecessary_columns_at_end_flag=True,
                nonfrequent_threshold=20, missing_values_threshold=0.2,
                create_combination_features_flag=True,
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

        self.nonfrequent_threshold = nonfrequent_threshold
        self.missing_values_threshold = missing_values_threshold

        self.X = self.df.drop(columns=['price', 'price_per_m', 'id'])
        self.Y = self.df[['price_per_m']]

        self.transform_original_features_flag = transform_original_features_flag
        if self.transform_original_features_flag:
            self.X["market_secondary"] = self.X["market"].apply(lambda x: 1 if x == "secondary" else 0)
            self.X = self.X.drop(columns=['market'])

            self.X["no_rooms"] = self.X["no_rooms"].apply(lambda x: 11 if x=="more" else int(x))

        if self.verbose:
            print("Data loaded")

        with open(cbrt_standard_scaler_path, 'rb') as f:
            self.cbrt_standard_scaler = pkl.load(f)

        self.remove_label_outliers_flag = remove_label_outliers_flag
        if not self.remove_label_outliers_flag:
            # Transform the price_per_m column to cbrt_standardized_price_per_m
            # The method overrides the Y attribute
            self.Y = self.forward_transform_price_per_m(self.Y)

        else:
            oi = OutlierImputer()
            y_train, y_test = self.get_train_test_split(self.Y)
            oi.fit(y_train)
            self.Y = pd.DataFrame(oi.transform(self.Y), columns=['price_per_m'])

            self.standard_scaler = StandardScaler()
            self.standard_scaler.fit(y_train)
            self.Y = pd.DataFrame(self.standard_scaler.transform(self.Y), columns=['price_per_m'])



        if self.verbose:
            print(f"The total number of columns is {len(self.X.columns)}")
        if get_params_from_params:
            self.params_parser = ParamsParser(self.X[['params']].copy(), **params_parser_args)
            self.X = self.X.join(self.params_parser.params)
            self.X.drop(columns=['params'], inplace=True)

            if self.verbose:
                print("Params parsed. New columns added")
                print(f"The total number of columns is {len(self.X.columns)}")

        self.get_tfidf_embeddings_flag = get_tfidf_embeddings_flag
        if self.get_tfidf_embeddings_flag:
            self.get_tfidf_embeddings()
            if self.verbose:
                print("TF-IDF embeddings added")
                print(f"The total number of columns is {len(self.X.columns)}")

        self.get_bert_embeddings_flag = get_bert_embeddings_flag
        if self.get_bert_embeddings_flag:
            self.get_bert_embeddings()
            if self.verbose:
                print("BERT embeddings added")
                print(f"The total number of columns is {len(self.X.columns)}")

        self.get_textual_features_flag = get_textual_features_flag
        if self.get_textual_features_flag:
            self.get_textual_features()
            if self.verbose:
                print("Textual features added")
                print(f"The total number of columns is {len(self.X.columns)}")

        self.X = self.X.drop(columns=['title', 'description'])
        if self.verbose:
            print("Text columns dropped")
            print(f"The total number of columns is {len(self.X.columns)}")

        self.transform_time_features_flag = transform_time_features_flag
        if self.transform_time_features_flag:
            self.transform_time_features()
            if self.verbose:
                print("Time features transformed")
                print(f"The total number of columns is {len(self.X.columns)}")

        self.transform_cyclic_features_flag = transform_cyclic_features_flag
        if self.transform_cyclic_features_flag:
            self.transform_cyclic_features()
            if self.verbose:
                print("Cyclic features transformed")
                print(f"The total number of columns is {len(self.X.columns)}")

        self.remove_nonfrequent_params_flag = remove_nonfrequent_params_flag
        if self.remove_nonfrequent_params_flag:
            self.remove_nonfrequent_params()
            if self.verbose:
                print("Nonfrequent params removed")
                print(f"The total number of columns is {len(self.X.columns)}")

        self.remove_missing_params_flag = remove_missing_params_flag
        if self.remove_missing_params_flag:
            self.remove_missing_params()
            if self.verbose:
                print("Missing params removed")
                print(f"The total number of columns is {len(self.X.columns)}")

        self.drop_unnecessary_columns_at_end_flag = drop_unnecessary_columns_at_end_flag
        if self.drop_unnecessary_columns_at_end_flag:
            self.X = self.X.drop(columns=['params'], errors='ignore')

    def get_tfidf_embeddings(self):
        title_embeddings_tfidf_pca = pd.read_csv("../data/new_features/pca_tfidf_titles.csv")
        title_embeddings_tfidf_pls2 = pd.read_csv("../data/new_features/pls2_tfidf_titles.csv")
        description_embeddings_tfidf_pca = pd.read_csv("../data/new_features/pca_tfidf_descriptions.csv")
        description_embeddings_tfidf_pls2 = pd.read_csv("../data/new_features/pls2_tfidf_descriptions.csv")

        self.X = self.X.join(title_embeddings_tfidf_pca)
        self.X = self.X.join(title_embeddings_tfidf_pls2)
        self.X = self.X.join(description_embeddings_tfidf_pca)
        self.X = self.X.join(description_embeddings_tfidf_pls2)

    def get_bert_embeddings(self):
        title_embeddings_bert_pca = pd.read_csv("../data/new_features/pca_bert_titles.csv")
        title_embeddings_bert_pls2 = pd.read_csv("../data/new_features/pls2_bert_titles.csv")

        self.X = self.X.join(title_embeddings_bert_pca)
        self.X = self.X.join(title_embeddings_bert_pls2)


    def get_textual_features(self):
        self.X['title_len'] = self.X['title'].apply(lambda x: len(x) if isinstance(x, str) else 0)
        self.X['description_len'] = self.X['description'].apply(lambda x: len(x) if isinstance(x, str) else 0)

        self.X['title_words'] = self.X['title'].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)
        self.X['description_words'] = self.X['description'].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)

        self.X['title_uppercase'] = self.X['title'].apply(lambda x: sum(1 for c in x if c.isupper()) if isinstance(x, str) else 0)
        self.X['description_uppercase'] = self.X['description'].apply(lambda x: sum(1 for c in x if c.isupper()) if isinstance(x, str) else 0)

        self.X['unique_words_title'] = self.X['title'].apply(lambda x: len(set(x.split())) if isinstance(x, str) else 0)
        self.X['unique_words_description'] = self.X['description'].apply(lambda x: len(set(x.split())) if isinstance(x, str) else 0)

        self.X['average_word_length_title'] = self.X['title'].apply(lambda x: np.mean([len(word) for word in x.split()]) if isinstance(x, str) else 0)
        self.X['average_word_length_description'] = self.X['description'].apply(lambda x: np.mean([len(word) for word in x.split()]) if isinstance(x, str) else 0)

        self.X['total_number_presence_title'] = self.X['title'].apply(lambda x: sum(1 for c in x if c.isdigit()) if isinstance(x, str) else 0)
        self.X['total_number_presence_description'] = self.X['description'].apply(lambda x: sum(1 for c in x if c.isdigit()) if isinstance(x, str) else 0)
        

    def transform_time_features(self):
        self.X['created_at_first'] = pd.to_datetime(self.X['created_at_first'])
        self.X['updated_at'] = pd.to_datetime(self.X['updated_at'])

        self.X['created_at_first_year'] = self.X['created_at_first'].dt.year
        self.X['created_at_first_month'] = self.X['created_at_first'].dt.month
        self.X['created_at_first_day'] = self.X['created_at_first'].dt.day

        self.X['updated_at_year'] = self.X['updated_at'].dt.year
        self.X['updated_at_month'] = self.X['updated_at'].dt.month
        self.X['updated_at_day'] = self.X['updated_at'].dt.day

        # Now get the hour and dayoftheweek features
        self.X['created_at_first_hour'] = self.X['created_at_first'].dt.hour
        self.X['created_at_first_dayofweek'] = self.X['created_at_first'].dt.dayofweek

        self.X['updated_at_hour'] = self.X['updated_at'].dt.hour
        self.X['updated_at_dayofweek'] = self.X['updated_at'].dt.dayofweek

        # Get a "duration" in between the created_at_first and updated_at columns
        self.X['duration_of_update'] = (self.X['updated_at'] - self.X['created_at_first']).dt.total_seconds()

        if "param_free_from" in self.X.columns.tolist():
            self.X['duration_of_free_from'] = (self.X['param_free_from'] - self.X['updated_at']).dt.total_seconds()

        self.X.drop(columns=['created_at_first', 'updated_at'], inplace=True)

    def transform_cyclic_features(self):
        cyclic_features = ["created_at_first_hour", "created_at_first_dayofweek", "updated_at_hour", "updated_at_dayofweek"]

        for feature in cyclic_features:
            self.X[f"{feature}_sin"] = np.sin(2 * np.pi * self.X[feature] / self.X[feature].max())
            self.X[f"{feature}_cos"] = np.cos(2 * np.pi * self.X[feature] / self.X[feature].max())

        self.X.drop(columns=cyclic_features, inplace=True)

    def remove_nonfrequent_params(self):
        one_hot_columns = [col for col in self.X.columns if self.X[col].dtype == "int" or self.X[col].dtype == "bool"]
        nonfrequent_columns = self.X[one_hot_columns].sum().loc[lambda x: x < self.nonfrequent_threshold].index.tolist()

        self.X.drop(columns=nonfrequent_columns, inplace=True)

    def remove_missing_params(self):
        percentage_missing = self.X.isnull().mean()
        missing_columns = percentage_missing.loc[lambda x: x > self.missing_values_threshold].index.tolist()

        self.X.drop(columns=missing_columns, inplace=True)

    def forward_transform_price_per_m(self, Y):

        Y = Y.copy() # Avoid changing the original dataframe
        # and avoid getting slicing errors

        Y.loc[:,'cbrt_price_per_m'] = Y['price_per_m'].apply(lambda x: np.cbrt(x) if x > 0 else 0)
        Y = Y.drop(columns=['price_per_m'])

        Y["standard_cbrt_price_per_m"] = self.cbrt_standard_scaler.transform(Y)
        Y = Y.drop(columns=['cbrt_price_per_m'])

        return Y

    def backward_transform_price_per_m(self, Y):
        Y = Y.copy()
        Y["cbrt_price_per_m"] = self.cbrt_standard_scaler.inverse_transform(Y)
        Y["price_per_m"] = Y["cbrt_price_per_m"].apply(lambda x: x**3 if x > 0 else 0)
        Y = Y.drop(columns=['cbrt_price_per_m'])

        return Y

    def backward_standardize_price(self, Y):
        Y = self.standard_scaler.inverse_transform(Y)
        return Y
    
    def get_train_test_split(self, X):
        # X is a dataframe with the same indices as the original dataframe
        X_train = X.iloc[self.train_indices]
        X_test = X.iloc[self.test_indices]

        return X_train, X_test


    def get_train_test_split_for_only_secondary_market_ads(self):

        X = self.X.copy
        y = self.Y.copy

        X_train = self.X.iloc[self.train_indices]
        X_test = self.X.iloc[self.test_indices]

        y_train = self.Y.iloc[self.train_indices]
        y_test = self.Y.iloc[self.test_indices]

        X_train = X_train.loc[X_train['market'] == 'secondary']
        X_test = X_test.loc[X_test['market'] == 'secondary']
        y_train = y_train.loc[X_train['market'] == 'secondary']
        y_test = y_test.loc[X_test['market'] == 'secondary']

        X_train = X_train.drop(columns=['market'])
        X_test = X_test.drop(columns=['market'])

        return X_train, X_test


if __name__ == "__main__":
    dp = DataPreprocessing(df_path="data/real_estate_ads_2022_10.csv", train_indices_path='data/train_indices.npy', 
                            test_indices_path='data/test_indices.npy',
                            cbrt_standard_scaler_path = "data/standard_scaler_cbrt_price_per_m.pkl")

    print(dp.Y)
