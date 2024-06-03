import unittest

import numpy as np
import pandas as pd

from src.data_loading import load_data
from src.transforms import UserInfoJoiner, OrgInfoJoiner, MissingRatingImputer


class TestTransformers(unittest.TestCase):
    def setUp(self):
        self.data = load_data('test_data/original_data_sample')

    def test_user_info_joiner(self):
        joiner = UserInfoJoiner(user_info=self.data.users)
        users = self.data.test_users
        processed_users = joiner.transform(users)
        assert 'user_city' in processed_users.columns
        assert processed_users.iloc[0]['user_city'] == 'msk'
        assert processed_users.iloc[195]['user_city'] == 'spb'

    def test_org_info_joiner(self):
        joiner = OrgInfoJoiner(org_info=self.data.organisations)
        reviews = self.data.reviews
        processed_reviews = joiner.transform(reviews)
        assert all(('org_' + column_name in processed_reviews.columns) for column_name in
                   ['city', 'average_bill', 'rating', 'rubrics_id', 'features_id'])
        assert processed_reviews.iloc[45].to_dict() == {'user_id': 458, 'org_id': 82, 'rating': 5.0, 'ts': 685,
                                                        'aspects': '6 38', 'org_city': 'msk',
                                                        'org_average_bill': 1500.0, 'org_rating': 4.400871459694989,
                                                        'org_rubrics_id': '30774',
                                                        'org_features_id': '1018 1415 1509 1524 1082283206 10462 20424 1416 273469383 20422 11177 11704 11867 11629'}

    def test_rating_imputer(self):
        reviews = pd.DataFrame(
            [
                {'rating': 1.0},
                {'rating': 3.0},
                {'rating': 5.0},
                {'rating': np.nan}
            ]
        )

        with self.subTest('mean imputer'):
            imputer = MissingRatingImputer(strategy='mean')
            mean_imputed = imputer.fit_transform(reviews)
            assert mean_imputed.iloc[3]['rating'] == 3.0
        with self.subTest('constant imputer'):
            imputer = MissingRatingImputer(strategy='constant', value=4.5)
            constant_imputed = imputer.fit_transform(reviews)
            assert constant_imputed.iloc[3]['rating'] == 4.5
