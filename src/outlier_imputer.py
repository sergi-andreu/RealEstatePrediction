import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class OutlierImputer(BaseEstimator, TransformerMixin):
    def __init__(self, lower_percentile=5, upper_percentile=95, multiplier=1.5):
        self.lower_percentile = lower_percentile # Lower percentile for the IQR
        self.upper_percentile = upper_percentile # Upper percentile for the IQR
        self.multiplier = multiplier # Multiplier for the IQR

    def _calculate_bounds_and_median(self, column):
        Q1 = np.percentile(column, self.lower_percentile)
        Q3 = np.percentile(column, self.upper_percentile)
        IQR = Q3 - Q1
        lower_bound = Q1 - (self.multiplier * IQR)
        upper_bound = Q3 + (self.multiplier * IQR)
        non_outliers = column[(column >= lower_bound) & (column <= upper_bound)]
        median = np.median(non_outliers)
        return lower_bound, upper_bound, median

    def fit(self, X):
        bounds_and_medians = X.apply(self._calculate_bounds_and_median)
        self.lower_bounds_ = bounds_and_medians.apply(lambda x: x[0])
        self.upper_bounds_ = bounds_and_medians.apply(lambda x: x[1])
        self.medians_ = bounds_and_medians.apply(lambda x: x[2])
        return self

    def _impute_outliers(self, column, lower_bound, upper_bound, median):
        return column.apply(lambda x: median if x < lower_bound or x > upper_bound else x)

    def transform(self, X, y=None):
        X_imputed = X.copy()
        for column in X.columns:
            X_imputed[column] = self._impute_outliers(X[column], self.lower_bounds_[column], self.upper_bounds_[column], self.medians_[column])
        return X_imputed

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)