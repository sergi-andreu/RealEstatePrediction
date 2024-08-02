import pandas as pd
import numpy as np

from sklearn.metrics import (
    explained_variance_score,
    r2_score,
    mean_absolute_percentage_error,
    median_absolute_error,
    mean_absolute_error,
    mean_squared_log_error,
)

def percentage_of_errors_less_than_threshold_score(y_true, y_pred, threshold=20):

    absolute_percentage_error = np.abs((y_true - y_pred) / y_true) * 100
    # now get the percentage of times that this lies below the threshold
    percentage_of_errors_less_than_threshold = (absolute_percentage_error < threshold).mean() * 100
    return percentage_of_errors_less_than_threshold

class Metrics:
    # Initialize the metrics

    def __init__(self, dp=None, backward_transform_flag=True,
                    backward_standardize_flag=False, clip_to_positive=True):

        self.dp = dp # We need the datapreprocessing object to transform the data back to original scale
        # Quite inefficient to pass the whole object, but it is the easiest way to do it
        # (just in case I need the whole object in the future)
        self.backward_transform_flag = backward_transform_flag # If True, we will transform the data back to original scale
        # which is important for the metrics

        self.backward_standardize_flag = backward_standardize_flag 
        # If True, we will standardize the data back to original scale
        # this is done whenever we don't do a cbrt transform but an outlier-removing one

        self.clip_to_positive = clip_to_positive # If True, we will clip the predictions to positive values

        self.train_explained_variance = []
        self.train_r2 = []
        self.train_mape = []
        self.train_median_absolute_error = []
        self.train_mean_absolute_error = []
        self.train_mean_squared_log_error = []
        self.train_custom_1 = []
        self.train_custom_5 = []
        self.train_custom_10 = []
        self.train_custom_20 = []

        self.test_explained_variance = []
        self.test_r2 = []
        self.test_mape = []
        self.test_median_absolute_error = []
        self.test_mean_absolute_error = []
        self.test_mean_squared_log_error = []
        self.test_custom_1 = []
        self.test_custom_5 = []
        self.test_custom_10 = []
        self.test_custom_20 = []

    def append(self, metrics):
        self.train_explained_variance.append(metrics["train_explained_variance"])
        self.train_r2.append(metrics["train_r2"])
        self.train_mape.append(metrics["train_mape"])
        self.train_median_absolute_error.append(metrics["train_median_absolute_error"])
        self.train_mean_absolute_error.append(metrics["train_mean_absolute_error"])
        self.train_mean_squared_log_error.append(metrics["train_mean_squared_log_error"])
        self.train_custom_1.append(metrics["train_custom_1"])
        self.train_custom_5.append(metrics["train_custom_5"])
        self.train_custom_10.append(metrics["train_custom_10"])
        self.train_custom_20.append(metrics["train_custom_20"])

        self.test_explained_variance.append(metrics["test_explained_variance"])
        self.test_r2.append(metrics["test_r2"])
        self.test_mape.append(metrics["test_mape"])
        self.test_median_absolute_error.append(metrics["test_median_absolute_error"])
        self.test_mean_absolute_error.append(metrics["test_mean_absolute_error"])
        self.test_mean_squared_log_error.append(metrics["test_mean_squared_log_error"])
        self.test_custom_1.append(metrics["test_custom_1"])
        self.test_custom_5.append(metrics["test_custom_5"])
        self.test_custom_10.append(metrics["test_custom_10"])
        self.test_custom_20.append(metrics["test_custom_20"])

    def get_average(self):
        average_metrics = {
            "train_explained_variance": np.mean(self.train_explained_variance),
            "train_r2": np.mean(self.train_r2),
            "train_mape": np.mean(self.train_mape),
            "train_median_absolute_error": np.mean(self.train_median_absolute_error),
            "train_mean_absolute_error": np.mean(self.train_mean_absolute_error),
            "train_mean_squared_log_error": np.mean(self.train_mean_squared_log_error),
            "train_custom_1": np.mean(self.train_custom_1),
            "train_custom_5": np.mean(self.train_custom_5),
            "train_custom_10": np.mean(self.train_custom_10),
            "train_custom_20": np.mean(self.train_custom_20),
            "test_explained_variance": np.mean(self.test_explained_variance),
            "test_r2": np.mean(self.test_r2),
            "test_mape": np.mean(self.test_mape),
            "test_median_absolute_error": np.mean(self.test_median_absolute_error),
            "test_mean_absolute_error": np.mean(self.test_mean_absolute_error),
            "test_mean_squared_log_error": np.mean(self.test_mean_squared_log_error),
            "test_custom_1": np.mean(self.test_custom_1),
            "test_custom_5": np.mean(self.test_custom_5),
            "test_custom_10": np.mean(self.test_custom_10),
            "test_custom_20": np.mean(self.test_custom_20),
        }

        return average_metrics

    def get_std(self):
        std_metrics = {
            "train_explained_variance": np.std(self.train_explained_variance),
            "train_r2": np.std(self.train_r2),
            "train_mape": np.std(self.train_mape),
            "train_median_absolute_error": np.std(self.train_median_absolute_error),
            "train_mean_absolute_error": np.std(self.train_mean_absolute_error),
            "train_mean_squared_log_error": np.std(self.train_mean_squared_log_error),
            "train_custom_1": np.std(self.train_custom_1),
            "train_custom_5": np.std(self.train_custom_5),
            "train_custom_10": np.std(self.train_custom_10),
            "train_custom_20": np.std(self.train_custom_20),
            "test_explained_variance": np.std(self.test_explained_variance),
            "test_r2": np.std(self.test_r2),
            "test_mape": np.std(self.test_mape),
            "test_median_absolute_error": np.std(self.test_median_absolute_error),
            "test_mean_absolute_error": np.std(self.test_mean_absolute_error),
            "test_mean_squared_log_error": np.std(self.test_mean_squared_log_error),
            "test_custom_1": np.std(self.test_custom_1),
            "test_custom_5": np.std(self.test_custom_5),
            "test_custom_10": np.std(self.test_custom_10),
            "test_custom_20": np.std(self.test_custom_20),
        }

        return std_metrics


    def get_single_train_val_metrics(self, model, X_train, y_train, X_val, y_val):

        y_train_pred = pd.DataFrame(model.predict(X_train), index=y_train.index)
        y_val_pred = pd.DataFrame(model.predict(X_val), index=y_val.index)

        if self.backward_transform_flag:
            # This is important. Otherwise, we will be getting the metrics for the transformed labels
            # since in this case the transformation is quite aggressive (square root),
            # the metrics will be quite different
            y_train = self.dp.backward_transform_price_per_m(y_train)["price_per_m"]
            y_train_pred = self.dp.backward_transform_price_per_m(y_train_pred)["price_per_m"]
            y_val = self.dp.backward_transform_price_per_m(y_val)["price_per_m"]
            y_val_pred = self.dp.backward_transform_price_per_m(y_val_pred)["price_per_m"]

        if self.backward_standardize_flag:
            y_train = self.dp.backward_standardize_price(y_train)
            y_train_pred = self.dp.backward_standardize_price(y_train_pred)
            y_val = self.dp.backward_standardize_price(y_val)
            y_val_pred = self.dp.backward_standardize_price(y_val_pred)
        
        if self.clip_to_positive:
            y_train_pred = y_train_pred.clip(lower=1e-9)
            y_val_pred = y_val_pred.clip(lower=1e-9)

        train_explained_variance = explained_variance_score(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        train_mape = mean_absolute_percentage_error(y_train, y_train_pred)
        train_median_absolute_error = median_absolute_error(y_train, y_train_pred)
        train_mean_absolute_error = mean_absolute_error(y_train, y_train_pred)
        train_mean_squared_log_error = mean_squared_log_error(y_train, y_train_pred)
        train_custom_1 = percentage_of_errors_less_than_threshold_score(y_train, y_train_pred, 1)
        train_custom_5 = percentage_of_errors_less_than_threshold_score(y_train, y_train_pred, 5)
        train_custom_10 = percentage_of_errors_less_than_threshold_score(y_train, y_train_pred, 10)
        train_custom_20 = percentage_of_errors_less_than_threshold_score(y_train, y_train_pred, 20)

        test_explained_variance = explained_variance_score(y_val, y_val_pred)
        test_r2 = r2_score(y_val, y_val_pred)
        test_mape = mean_absolute_percentage_error(y_val, y_val_pred)
        test_median_absolute_error = median_absolute_error(y_val, y_val_pred)
        test_mean_absolute_error = mean_absolute_error(y_val, y_val_pred)
        test_mean_squared_log_error = mean_squared_log_error(y_val, y_val_pred)
        test_custom_1 = percentage_of_errors_less_than_threshold_score(y_val, y_val_pred, 1)
        test_custom_5 = percentage_of_errors_less_than_threshold_score(y_val, y_val_pred, 5)
        test_custom_10 = percentage_of_errors_less_than_threshold_score(y_val, y_val_pred, 10)
        test_custom_20 = percentage_of_errors_less_than_threshold_score(y_val, y_val_pred, 20)

        return {
            "train_explained_variance": train_explained_variance,
            "train_r2": train_r2,
            "train_mape": train_mape,
            "train_median_absolute_error": train_median_absolute_error,
            "train_mean_absolute_error": train_mean_absolute_error,
            "train_mean_squared_log_error": train_mean_squared_log_error,
            "train_custom_1" : train_custom_1,
            "train_custom_5": train_custom_5,
            "train_custom_10": train_custom_10,
            "train_custom_20": train_custom_20,
            "test_explained_variance": test_explained_variance,
            "test_r2": test_r2,
            "test_mape": test_mape,
            "test_median_absolute_error": test_median_absolute_error,
            "test_mean_absolute_error": test_mean_absolute_error,
            "test_mean_squared_log_error": test_mean_squared_log_error,
            "test_custom_1" : test_custom_1,
            "test_custom_5": test_custom_5,
            "test_custom_10": test_custom_10,
            "test_custom_20": test_custom_20}