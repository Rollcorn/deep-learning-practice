import unittest

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

import secret
from src.data_loading import load_data
from src.models.statistics_based import PopularityRecommender
from src.transforms import UserInfoJoiner, OrgInfoJoiner


class TestPopularRecommender(unittest.TestCase):
    def test_popular_recommender(self):
        with self.subTest('just one popular'):
            # there are four orgs, two in msk, two in spb
            org_info = pd.DataFrame([
                {'org_id': 1, 'city': 'msk'},
                {'org_id': 2, 'city': 'msk'},
                {'org_id': 3, 'city': 'spb'},
                {'org_id': 4, 'city': 'spb'},
            ])
            # there are four users as well
            user_info = pd.DataFrame([
                {'user_id': 1, 'city': 'msk'},
                {'user_id': 2, 'city': 'msk'},
                {'user_id': 3, 'city': 'spb'},
                {'user_id': 4, 'city': 'spb'},
            ])
            # now for the reviews
            reviews = pd.DataFrame([
                # org 1 in msk has three visitors
                {'user_id': 1, 'org_id': 1, 'rating': 3},
                {'user_id': 2, 'org_id': 1, 'rating': 2},
                {'user_id': 3, 'org_id': 1, 'rating': 1},
                # org 2 in msk has two visitors
                {'user_id': 1, 'org_id': 2, 'rating': 5},
                {'user_id': 2, 'org_id': 2, 'rating': 5},
                # org 3 in spb has just one unique visitor
                {'user_id': 4, 'org_id': 3, 'rating': 4},
                {'user_id': 4, 'org_id': 3, 'rating': 5},
                {'user_id': 4, 'org_id': 3, 'rating': 5},
                # org 4 has two unique visitors so it should be more popular in spb
                {'user_id': 4, 'org_id': 4, 'rating': 2},
                {'user_id': 3, 'org_id': 4, 'rating': 2},
            ])
            # we predict for known users from both spb and msk, but it should be indifferent
            predict_for_users = pd.DataFrame([
                {'user_id': 1},
                {'user_id': 2},
                {'user_id': 3},
                {'user_id': 4},
            ])
            pipeline = Pipeline([
                ('user_info_joiner', UserInfoJoiner(user_info=user_info)),
                ('org_info_joiner', OrgInfoJoiner(org_info=org_info)),
                ('popular_recommender', PopularityRecommender(n_items=1))
            ])
            predictions = pipeline.fit(reviews).predict(predict_for_users)
            # users from msk are recommended top org from spb
            assert predictions.loc[1] == np.array([4])
            assert predictions.loc[2] == np.array([4])
            # users from spb are recommended top org from msk
            assert predictions.loc[3] == np.array([1])
            assert predictions.loc[4] == np.array([1])

        with self.subTest('20 popular'):
            data = load_data('test_data/original_data_sample')
            n_items = 5  # (there is no ambiguity for this number of most popular items)
            pipeline = Pipeline([
                ('user_info_joiner', UserInfoJoiner(user_info=data.users)),
                ('org_info_joiner', OrgInfoJoiner(org_info=data.organisations)),
                ('popular_recommender', PopularityRecommender(n_items=n_items))
            ])

            predictions = pipeline.fit(data.reviews).predict(data.test_users)
            self.assertTrue(
                np.all(predictions.loc[437] == np.array([1489, 603, 1226, 733, 952])) or
                np.all(predictions.loc[437] == np.array([1489, 603, 1226, 952, 733]))
            )
            self.assertTrue(
                np.all(predictions.loc[826] == np.array([1462, 1173, 1306, 278, 439])) or
                np.all(predictions.loc[826] == np.array([1462, 1173, 1306, 439, 278]))
            )



def test_guarantees(self):
        data = load_data('test_data/original_data_sample')
        n_guarantee_tests = 10
        n_items = 5
        rng = np.random.default_rng(0xBAD_5EED)
        for test_idx in range(n_guarantee_tests):
            random_review_sample = data.reviews.sample(frac=0.3, random_state=rng)
            pipeline = Pipeline([
                ('user_info_joiner', UserInfoJoiner(user_info=data.users)),
                ('org_info_joiner', OrgInfoJoiner(org_info=data.organisations)),
                ('popular_recommender', PopularityRecommender(n_items=n_items))
            ])
            predict_for = \
                data.test_users.join(
                    data.users.set_index('user_id'), on='user_id', how='left'
                ).groupby('city').sample(1, random_state=rng)['user_id'].reset_index()

            predictions = pipeline.fit(random_review_sample).predict(predict_for)
            validator = secret.PopularityValidator(
                reviews=random_review_sample,
                test_users=predict_for,
                user_data=data.users,
                org_data=data.organisations,
                k=n_items
            )
            for user, prediction in predictions.items():
                for item in prediction:
                    self.assertTrue(validator(user, item))
