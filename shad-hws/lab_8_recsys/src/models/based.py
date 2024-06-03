from typing import Any, NamedTuple
import sys

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator as sklearn_BaseEstimator


class BaseEstimator(sklearn_BaseEstimator):
    def __getattr__(self, item):
        if '_' + item in self.__dict__:
            return '...'
        else:
            raise AttributeError(f'{item} is not a member of {self.__class__} instance, check for typos')


class CityOrgMapping(NamedTuple):
    spb: Any
    msk: Any


class BaseEstimatorPerUserCity(BaseEstimator):
    city_to_org_ids: CityOrgMapping = None

    def fit(self, X, y=None):
        if self.city_to_org_ids is None:
            city_orgs_mapping = X[['org_id', 'org_city']].drop_duplicates().groupby('org_city')['org_id'].apply(
                np.array)
            self.city_to_org_ids = CityOrgMapping(msk=city_orgs_mapping['msk'], spb=city_orgs_mapping['spb'])
        return self

    def predict_user_org(self, users, orgs):
        return [np.random.choice(orgs, size=1, replace=False) for _ in users]

    def split_users_by_city(self, X):
        msk_mask = X['user_city'] == 'msk'
        return X['user_id'][msk_mask], X['user_id'][~msk_mask]

    def predict(self, X):
        msk_users, spb_users = self.split_users_by_city(X)
        prediction = pd.Series(index=X['user_id'], name='prediction', dtype=object)
        prediction.loc[msk_users] = self.predict_user_org(msk_users, self.city_to_org_ids.spb)
        prediction.loc[spb_users] = self.predict_user_org(spb_users, self.city_to_org_ids.msk)
        return prediction


class RandomRecommender(BaseEstimatorPerUserCity):
    def __init__(self, organisations_and_cities: pd.DataFrame, n_items: int = 20, random_state=None):
        self.n_items = n_items
        self._organisations_and_cities = None  # noqa (used for __repr__)
        self._random_state = random_state
        self.city_to_org_ids = CityOrgMapping(
            msk=organisations_and_cities[organisations_and_cities['city'] == 'msk']['org_id'].values,
            spb=organisations_and_cities[organisations_and_cities['city'] == 'spb']['org_id'].values
        )
        self.rng = np.random.default_rng(random_state)

    def fit(self, X: pd.DataFrame | None = None, y: pd.Series | None = None):
        return self

    def predict_user_org(self, users, orgs):
        """
        suggest organisations from `orgs` to users from `users`
        :param users: array-like
        :param orgs: array-like
        :return: list[np.ndarray] or compatible structure
        """
        return [self.rng.choice(orgs, size=np.min([self.n_items, len(orgs)]), replace=False) for _ in range(len(users))]


class EstimatorWithFallback(BaseEstimatorPerUserCity):
    def __init__(self, fallback_estimator=RandomRecommender, **kwargs):
        self.fallback_estimator = fallback_estimator(**kwargs)

    def fit(self, X, y=None):
        super().fit(X, y)
        self.fallback_estimator.fit(X)
        self.known_users = X['user_id'].unique()
        return self

    def predict_for_known_users(self, X):
        return super().predict(X)

    def predict(self, X):
        fallback_prediction = self.fallback_estimator.predict(X)
        intersection = X['user_id'].isin(self.known_users)
        known_users = X[intersection]
        prediction = self.predict_for_known_users(known_users).dropna()
        if len(prediction) > 0:
            expanded_prediction = prediction.rename('prediction').to_frame().join(
                fallback_prediction.rename('fallback_prediction'), how='left')
            complete_known_prediction = expanded_prediction.apply(lambda row: pd.unique(
                np.concatenate(
                    (row['prediction'], row['fallback_prediction'])
                )
            )[:self.fallback_estimator.n_items], axis=1)
            fallback_prediction.loc[complete_known_prediction.index] = complete_known_prediction
        return fallback_prediction
