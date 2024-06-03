import unittest

import numpy as np
import pandas as pd
from src.cross_val import DaysTimeSeriesSplit

class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0xBAD_5EED)

    def create_random_dataframe_containing_days_column(self, date_range, distinct_dates_count=None, rows_count=100):
        if not isinstance(date_range, tuple):
            date_range = (0, date_range)
        if distinct_dates_count is None:
            distinct_dates_count = date_range[1] - date_range[0]
        possible_dates = np.sort(
            self.rng.choice(
                a=date_range[1] - date_range[0],
                size=distinct_dates_count,
                replace=False,
                shuffle=False
            ) + date_range[0]
        )
        if distinct_dates_count == rows_count:
            partition_cumsum = np.arange(rows_count + 1)
        else:
            partition_cumsum = np.insert(np.sort(self.rng.choice(
                a=rows_count - 2,
                size=distinct_dates_count - 1,
                replace=False,
                shuffle=False
            ) + 1), (0, distinct_dates_count - 1), values=(0, rows_count))
        date_column = np.empty(rows_count, dtype=int)
        for date, start_pos, end_pos in zip(possible_dates, partition_cumsum[:-1], partition_cumsum[1:]):
            date_column[start_pos: end_pos] = date
        self.rng.shuffle(date_column)

        return pd.DataFrame(
            data={'user_id': self.rng.integers(rows_count, size=rows_count),
                  'org_id': self.rng.integers(rows_count, size=rows_count), 'ts': date_column})

    def test_just_days_ts_split_corner_cases(self):
        with self.subTest(msg='Test too large test size'):
            input_dataframe = self.create_random_dataframe_containing_days_column(
                date_range=(0, 10),
                rows_count=20,
            )
            splitter = DaysTimeSeriesSplit(n_splits=1, test_size_in_days=11, min_train_size_in_days=1)
            with self.assertRaises(ValueError):
                _ = list(splitter.split(input_dataframe))

        with self.subTest(msg='Test too many splits'):
            input_dataframe = self.create_random_dataframe_containing_days_column(
                date_range=(0, 10),
                rows_count=20,
            )
            splitter = DaysTimeSeriesSplit(n_splits=11, test_size_in_days=1, min_train_size_in_days=1)
            with self.assertRaises(ValueError):
                _ = list(splitter.split(input_dataframe))

        with self.subTest(msg='Test large min_train_size'):
            input_dataframe = self.create_random_dataframe_containing_days_column(
                date_range=(0, 10),
                rows_count=20,
            )
            splitter = DaysTimeSeriesSplit(n_splits=1, test_size_in_days=2, min_train_size_in_days=10)
            with self.assertRaises(ValueError):
                _ = list(splitter.split(input_dataframe))

    def test_just_days_ts_split_guarantees(self):
        for _ in range(100):
            input_dataframe = self.create_random_dataframe_containing_days_column(
                date_range=(0, 100),
                distinct_dates_count=51,
                rows_count=500,
            )
            n_splits = 5
            test_size = 10
            min_train_size = 10
            splitter = DaysTimeSeriesSplit(n_splits=n_splits, test_size_in_days=test_size,
                                           min_train_size_in_days=min_train_size)
            splits = list(splitter.split(input_dataframe))
            assert len(splits) == n_splits
            for (train_set, _), (next_train_set, _) in zip(splits[:-1], splits[1:]):
                self.assertTrue(set(train_set) < set(next_train_set))
            for train_set, test_set in splits:
                self.assertFalse(set(train_set) & set(test_set))
                self.assertTrue(input_dataframe.iloc[train_set]['ts'].nunique() >= min_train_size)
                self.assertTrue(input_dataframe.iloc[test_set]['ts'].nunique() == test_size)
                self.assertTrue(
                    (
                            input_dataframe.iloc[train_set]['ts'].unique()[:, None] <
                            input_dataframe.iloc[test_set]['ts'].unique()[None, :]
                    ).all()
                )
            self.assertTrue(len(train_set) + len(test_set) == 500)
