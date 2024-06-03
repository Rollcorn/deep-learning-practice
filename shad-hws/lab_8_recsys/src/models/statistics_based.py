import numpy as np
import pandas as pd
import scipy

from src.models.based import BaseEstimatorPerUserCity, CityOrgMapping


class PopularityRecommender(BaseEstimatorPerUserCity):
    def __init__(self, n_items: int = 20):
        self.n_items = n_items

    def fit(self, X, y=None):
        """compute popularity nof orgs globally and store it in city_to_org_ids"""
        org_popularity = X.groupby('org_id')['user_id'].nunique().reset_index()
        org_popularity.columns = ['org_id', 'unique_visitors']
        city_orgs = X[['org_id', 'org_city']].drop_duplicates().set_index('org_id')
        org_popularity = org_popularity.join(city_orgs, on='org_id')

        city_dict = {}
        for city, orgs in org_popularity.groupby('org_city'):
            top_orgs = orgs.sort_values('unique_visitors', ascending=False).head(self.n_items)['org_id'].tolist()
            city_dict[city] = top_orgs
        self.city_to_org_ids = CityOrgMapping(msk=city_dict['msk'], spb=city_dict['spb'])
        return self

    def predict_user_org(self, users, orgs):
            """
            predict popular orgs to given users
            :param users:
            :param orgs:
            :return:
            """
            return [orgs]*len(users)

class BayesRatingRecommender(PopularityRecommender):
    def __init__(self, n_items: int = 20, alpha=0.1):
        self.alpha = alpha
        self._z = scipy.stats.norm.ppf(1 - alpha / 2)
        self.n_items = n_items

    def get_bayesian_lower_bound_per_group(self, group):
        raise NotImplementedError('BONUS')

    def get_bayesian_lower_bound_top(self, orgs: pd.DataFrame):
        raise NotImplementedError('BONUS')

    def fit(self, X, y=None):
        raise NotImplementedError('BONUS')

#
# class BayesRatingRecommender(PopularityRecommender):
#     def __init__(self, n_items: int = 20, alpha=0.1):
#         self.alpha = alpha
#         self._z = scipy.stats.norm.ppf(1 - alpha / 2)
#         self.n_items = n_items
#
#     def get_bayesian_lower_bound_per_group(self, group):
#         raise NotImplementedError('BONUS')
#
#     def get_bayesian_lower_bound_top(self, orgs: pd.DataFrame):
#         raise NotImplementedError('BONUS')
#
#     def fit(self, X, y=None):
#         raise NotImplementedError('BONUS')
