class ParamsParser:
    def __init__(self, params, get_features_flag=True, drop_ugly_params_flag=True, verbose=False):
        assert isinstance(params, pd.DataFrame)
        self.params = params # params is a dataframe with one column, "params"

        self.param_keys = self.get_param_keys_from_str(self.params["params"].tolist())

        self.ugly_params = ["m"]

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



    def get(self, key):
        return self.params.get(key)

    def get_param_keys(self):
        all_params_keys = [[ll.split("<=>")[0] for ll in param.split("<br>")] for param in self.params["param"].tolist()]
        param_keys = {param for param_list in params for param in all_params_keys}
        return param_keys
    
    def get_param_value_for_param_key(self, param_key):
        param = re.escape(param_key)
        pattern = re.compile(f'{param}<=>(.*?)<br>')
        currencies = [re.findall(pattern, t) for t in self.params["param"].tolist()]

        return currencies
    
    def get_new_features_raw(self):
        for param in self.param_keys:
            currencies = self.get_param_value_for_param_key(param)
            self.params[f"param_{param}"] = currencies

    def drop_ugly_params(self):
        for param in self.ugly_params:
            self.params = self.params.drop(columns=[f"param_{param}"])
            if self.verbose:
                print(f"Ugly param {param} dropped")

    def clean_prices_param(self, col):
        return col.apply(lambda x: x[1] if x else None).astype("float")


    

