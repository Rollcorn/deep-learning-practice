import numpy as np
from sklearn.model_selection._split import _BaseKFold

"""
Implement a cross validation data splitter that satisfies following guarantees:

Train and test do not intersect sample-wise and time-wise
Train contains minimum requested number of days
Test is exactly requested number of days
Last train+test pair covers the entire dataset
Each following train fold is a superset of previous one
Notice that there is no need to linearly increase the train set, as in real world application that may not be how dataset grows
"""
class DaysTimeSeriesSplit(_BaseKFold):
    def __init__(self, n_splits, test_size_in_days, min_train_size_in_days, days_column_name='ts'):
        self.n_splits = n_splits
        self.test_size_in_days = test_size_in_days
        self.min_train_size_in_days = min_train_size_in_days
        self.days_column_name = days_column_name

    def split(self, X, y=None, groups=None):
        dates = X[self.days_column_name].values
        unique_days = np.sort(np.unique(dates))

        max_train_idx = len(unique_days) - self.test_size_in_days - 1
        min_train_idx = self.min_train_size_in_days - 1
        split_step = (max_train_idx - min_train_idx) // self.n_splits

        if len(unique_days) - self.test_size_in_days - self.min_train_size_in_days < (self.n_splits - 1) * split_step or split_step == 0:
            raise ValueError("Not enough data points to create the number of splits")

        for split_idx in range(self.n_splits):
            if split_idx == self.n_splits - 1:
                max_train_date_idx = max_train_idx
            else:
                max_train_date_idx = min_train_idx + split_step * split_idx

            test_start_idx = max_train_date_idx + 1
            test_end_idx = test_start_idx + self.test_size_in_days - 1
            if max_train_date_idx + 1 < self.min_train_size_in_days:
                raise ValueError("Not enough data points in train or test set for split")

            test_start_date = unique_days[test_start_idx]
            test_end_date = unique_days[test_end_idx]

            min_train_date = unique_days[0]
            max_train_date = unique_days[max_train_date_idx]

            train_mask = (dates <= max_train_date) & (dates >= min_train_date)
            test_mask = (dates >= test_start_date) & (dates <= test_end_date)
            train_indices = X.index[train_mask]
            test_indices = X.index[test_mask]
            yield train_indices, test_indices

