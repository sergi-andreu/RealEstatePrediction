import pandas as pd
import re
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer


class ParamsParser:
    def __init__(self, params, get_features_flag=True, drop_ugly_params_flag=True, 
                remove_repeated_columns_flag=True, 
                drop_params_column_flag=True, correct_media_types_television_flag=True,
                one_hot_encode_flag=True, transform_types_flag=True,
                expand_params_flag=True, verbose=False):
        assert isinstance(params, pd.DataFrame)
        self.params = params # params is a dataframe with one column, "params"

        self.verbose = verbose # verbose flag

        self.param_keys = self.get_param_keys()

        self.ugly_params = ["m", "location", "roofing", "remote_services", "recreational"]

        self.one_hot_encoding_features = ["param_price[currency]", "param_rent[currency]",
                                        "param_building_type", "param_windows_type", "param_heating_types",
                                        "param_building_ownership", "param_heating", "param_roof_type",
                                        "param_garret_type", "param_building_material", "param_construction_status"]

        self.get_features_flag = get_features_flag
        if self.get_features_flag: 
            self.get_new_features_raw()
            if self.verbose:
                print("New features added from params :)")

        # Cleaning prices and rent (could be a flag, but should be mandatory for consistency of the data)
        for params in ["param_price", "param_rent"]:
            self.params.loc[:,params] = self.clean_prices_param(self.params[params])
            if self.verbose:
                print(f"{params} cleaned")

        self.drop_ugly_params_flag = drop_ugly_params_flag
        if self.drop_ugly_params_flag:
            self.drop_ugly_params()
            if self.verbose:
                print("Ugly params dropped")

        self.convert_lists_to_floats()
        if self.verbose:
            print("Lists converted to floats")

        self.expand_params_flag = expand_params_flag
        if self.expand_params_flag:
            self.params = self.expand_params()

        self.repeated_columns = ["param_price", "param_rent", "param_market", "param_price_per_m",
                                "param_rooms_num"]

        if remove_repeated_columns_flag:
            self.remove_params(self.repeated_columns)
            if self.verbose:
                print("Repeated columns removed")

        self.correct_media_types_television_flag = correct_media_types_television_flag
        if self.correct_media_types_television_flag:
            self.params["param_media_types_cable_television"] = self.params["param_media_types_cable_television"] + self.params["param_media_types_cable-television"]
            self.params = self.params.drop(columns=["param_media_types_cable-television"])
            if self.verbose:
                print("Media types cable television corrected")

        if one_hot_encode_flag:
            self.one_hot_encode()
            if self.verbose:
                print("One hot encoding done")

        self.drop_params_column_flag = drop_params_column_flag
        if self.drop_params_column_flag:
            self.drop_params_column()

        self.transform_types_flag = transform_types_flag
        if self.transform_types_flag:
            self.transform_types()
            if self.verbose:
                print("Types transformed")

    def get(self, key):
        return self.params.get(key)

    def get_param_keys(self):
        all_params_keys = [[ll.split("<=>")[0] for ll in param.split("<br>")] for param in self.params["params"].tolist()]
        # now we flatten the list:
        flatten_all_params_keys = [item for sublist in all_params_keys for item in sublist]
        param_keys = set(flatten_all_params_keys)
        return param_keys
    
    def get_param_value_for_param_key(self, param_key):
        param = re.escape(param_key)
        pattern = re.compile(f'{param}<=>(.*?)<br>')
        currencies = [re.findall(pattern, t) for t in self.params["params"].tolist()]

        return currencies
    
    def get_new_features_raw(self):
        for param in self.param_keys:
            currencies = self.get_param_value_for_param_key(param)
            self.params[f"param_{param}"] = currencies

    def drop_ugly_params(self):
        for param in self.ugly_params:
            self.params = self.params.drop(columns=[f"param_{param}"])
            self.param_keys.remove(param)
            if self.verbose:
                print(f"Ugly param {param} dropped")

    def clean_prices_param(self, col):
        cleaned_col = col.apply(lambda x: x[1] if len(x)>1 else np.nan)
        cleaned_col = cleaned_col.apply(lambda x: float(x) if x else np.nan)
        return cleaned_col

    def convert_lists_to_floats(self):
        for param in self.param_keys:
            try:
                self.params[f"param_{param}"] = self.params[f"param_{param}"].apply(lambda x: x[0] if x else np.nan)
            except: # if the column is already a float
                pass

    def expand_params(self):

        expanded_columns_names = []

        df = self.params.copy()
        for param in self.param_keys:
            df[f"param_{param}"] = df[f"param_{param}"].apply(lambda x : x.split("<->") if isinstance(x, str) else [x])

            if df[f"param_{param}"].apply(lambda x: len(x)).max() <= 1:
                df[f"param_{param}"] = df[f"param_{param}"].apply(lambda x: x[0] if x else np.nan)

            else:
                clean_list = lambda x: [i for i in x if i not in [0, "0", "", np.nan, None]]
                df[f"param_{param}"] = df[f"param_{param}"].apply(clean_list)
                df[f"param_{param}"] = df[f"param_{param}"].apply(lambda x: tuple(x) if x else ())
                mlb = MultiLabelBinarizer()
                transformed_ = mlb.fit_transform(df[f"param_{param}"])
                column_names = [f"param_{param}_{name}" for name in mlb.classes_]
                tmp_df = pd.DataFrame(transformed_,columns=column_names, index=df.index)
                df = pd.concat([df, tmp_df], axis=1)
                df = df.drop(columns=[f"param_{param}"])

                expanded_columns_names += column_names

        return df

    def one_hot_encode(self):
        clean_values = lambda x: x if x not in [0, "0", "", np.nan, None] else np.nan
        for par in self.one_hot_encoding_features:
            self.params[par] = self.params[par].apply(clean_values)

        self.params = pd.get_dummies(self.params, columns=self.one_hot_encoding_features, prefix=self.one_hot_encoding_features,
                                    drop_first=True)
        # Here we drop_first to avoid multicollinearity and repeated information variables


    def remove_params(self, params):
        self.params = self.params.drop(columns=params)
        self.param_keys = [param for param in self.param_keys if param not in params]
        if self.verbose:
            print("Params removed: ", params)

    def identify_nonfrequent_params(self, min_frequency=20):
        binary_columns = [col for col in self.params.columns if self.params[col].dtype == "int" or self.params[col].dtype == "bool"]
        print(binary_columns)

        return binary_columns

    def drop_params_column(self):
        self.params = self.params.drop(columns=["params"])
        if self.verbose:
            print("Params column dropped")

    def transform_types(self):
        object_features = self.params.select_dtypes(include="object").columns

        def try_numeric_transformation(x):
            try:
                return int(x)
            except:
                return np.nan

        for feature in object_features:
            if feature in ["param_build_year", "param_terrain_area", "param_building_floors_num", "param_floor_no", "param_floors_num"]:
                self.params[feature] = self.params[feature].apply(try_numeric_transformation)

                if feature in ["param_building_floors_num", "param_floor_no", "param_floors_num"]:
                    self.params[feature] = self.params[feature].apply(lambda x: x if x < 100 else np.nan)
                    self.params[feature] = self.params[feature].apply(lambda x: x if x > 0 else np.nan)

                if feature in ["param_build_year"]:
                    self.params[feature] = self.params[feature].apply(lambda x: x if x < 2022 else np.nan)
                    self.params[feature] = self.params[feature].apply(lambda x: x if x > 1000 else np.nan)

                if feature in ["param_terrain_area"]:
                    self.params[feature] = self.params[feature].apply(lambda x: x if x < 100000 else np.nan)
                    self.params[feature] = self.params[feature].apply(lambda x: x if x > 0 else np.nan)

            if feature in ["param_free_from"]:
                self.params[feature] = pd.to_datetime(self.params[feature], errors='coerce')


    