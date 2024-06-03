import unittest

import numpy as np
import pandas as pd
import scipy.stats
from sklearn.pipeline import Pipeline

from src.data_loading import load_data
from src.models.based import RandomRecommender
from src.transforms import UserInfoJoiner


class TestRandomRecommender(unittest.TestCase):
    def test_random_recommender(self):
        data = load_data('test_data/original_data_sample')
        with self.subTest("enough org data"):
            n_items = 20
            recommender = Pipeline([
                ('join_user_info', UserInfoJoiner(user_info=data.users)),
                ('recommender', RandomRecommender(organisations_and_cities=data.organisations,
                                                  n=n_items))
                # random state is not fixed here, so make your own default value
            ])
            recommender.fit(data.reviews)
            predict = recommender.predict(data.test_users)
            assert isinstance(predict, pd.Series)
            assert len(predict) == len(data.test_users)
            # assert np.all(predict.index == data.test_users.index)
            assert all(len(p_i) == n_items for p_i in predict.values)
            for user, recs in predict.items():
                assert all(data.users.loc[user]['city'] != data.organisations.loc[org]['city'] for org in recs)
            # chi square uniformity test
            predict = recommender.predict(data.users.drop('city', axis=1))
            sample = np.concatenate(predict.values)
            _, counts = np.unique(sample, return_counts=True)
            result = scipy.stats.chisquare(counts / len(sample))
            assert result.pvalue >= 0.05

        with self.subTest("not enough org data"):
            n_items = 20
            cutoff = 5
            org_data = data.organisations.groupby('city').head(cutoff)
            recommender = Pipeline([
                ('join_user_info', UserInfoJoiner(user_info=data.users)),
                ('recommender', RandomRecommender(organisations_and_cities=org_data,
                                                  n=n_items))
                # random state is not fixed here, so make your own default value
            ])
            recommender.fit(data.reviews)
            predict = recommender.predict(data.test_users)
            assert isinstance(predict, pd.Series)
            assert len(predict) == len(data.test_users)
            # assert np.all(predict.index == data.test_users.index)
            assert all(len(p_i) == cutoff for p_i in predict.values)
            for user, recs in predict.items():
                assert all(data.users.loc[user]['city'] != data.organisations.loc[org]['city'] for org in recs)
            # chi square uniformity test
            predict = recommender.predict(data.users.drop('city', axis=1))
            sample = np.concatenate(predict.values)
            _, counts = np.unique(sample, return_counts=True)
            result = scipy.stats.chisquare(counts / len(sample))
            assert result.pvalue >= 0.05
