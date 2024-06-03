from sklearn.base import TransformerMixin

from src.models.based import BaseEstimator

"""
Implement following data transforms:

one that joins user data to train dataset record by user_id
one that joins organisation data to train dataset record by org_id
one that fixes missing rating values in train dataset by substituting population average or fixed constant
"""
class UserInfoJoiner(BaseEstimator, TransformerMixin):
    def __init__(self, user_info):
        self._user_info = user_info.set_index('user_id').rename(columns=lambda s: 'user_' + s)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        join user info (city is important) to reviews dataframe
        :param X:
        :return:
        """
        return X.join(self._user_info, on='user_id', how='left')

class OrgInfoJoiner(BaseEstimator, TransformerMixin):
    def __init__(self, org_info):
        self._org_info = org_info.set_index('org_id').rename(columns=lambda s: 'org_' + s)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        join org info (city is important) to reviews dataframe, mind that orgs are not always present in input dataframe
        say, in case of prediction
        :param X:
        :return:
        """
        X_c = X.copy()
        if 'org_id' in X_c:
            return X_c.join(self._org_info, on='org_id', how='left')
        return X_c

class MissingRatingImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy, value=None):
        self.strategy = strategy
        self.value = value

    def fit(self, X, y=None):
        if self.strategy == 'mean':
            self.value = X['rating'].mean()
        return self

    def transform(self, X):
        """
        impute missing rating values (there are some in original data)
        :param X:
        :return:
        """
        X = X.copy()
        X['rating'].fillna(self.value, inplace=True)
        return X