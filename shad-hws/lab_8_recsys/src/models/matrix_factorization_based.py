import numpy as np
import pandas as pd

from src.models.based import EstimatorWithFallback
from src.models.statistics_based import PopularityRecommender


class SVDRecommender(EstimatorWithFallback):
    def __init__(self, n_components, n_items, fallback_estimator=PopularityRecommender, random_state=None):
        super().__init__(fallback_estimator, n_items=n_items)
        self.n_components = n_components
        self._random_state = random_state
        self.n_items = n_items
        self.rng = np.random.default_rng(random_state)

    def make_sparse(self, X):
        """
        Make sparse matrix from reviews (hint: use original org/user_ids as indices)
        :param X:
        :return:
        """
        raise NotImplementedError('TASK')

    def fit(self, X, y=None):
        """
        compute user and item representations using (presumably) sparse svd
        :param X:
        :param y:
        :return:
        """
        super().fit(X)
        raise NotImplementedError('TASK')
        return self

    def predict_user_org(self, users, orgs):
        """
        use embeddings to compute user predictions (ordered by reconstructed score)
        :param users: predict for these users
        :param orgs: use orgs from this set
        :return:
        """
        raise NotImplementedError('TASK')
        return pd.Series(index=users, data=..., name='prediction')


class ALSRecommender(EstimatorWithFallback):
    def __init__(self, n_items, feature_dim, regularizer, num_iter, fallback_estimator=PopularityRecommender,
                 random_state=None):
        super().__init__(fallback_estimator, n_items=n_items)
        self.n_items = n_items
        self.feature_dim = feature_dim
        self.regularizer = regularizer
        self.num_iter = num_iter
        self._random_state = random_state
        self.rng = np.random.default_rng(random_state)

    def update_other_embeddings(self, embeddings, ranking_matrix):
        """
        compute alternate set of embeddings
        :param embeddings: matrix [n_entries, n_features]
        :param ranking_matrix: sparse representation of interactions
        :return:
        """
        raise NotImplementedError('TASK')

    def compute_loss(self, user_embeddings, item_embeddings, ranking_matrix):
        """
        compute reconstruction and regularizing loss combination
        :param user_embeddings: matrix [n_users, n_features]
        :param item_embeddings: matrix [n_items, n_features]
        :param ranking_matrix: sparse representation of interactions
        :return:
        """
        raise NotImplementedError('TASK')

    def make_sparse(self, X):
        """
        Make sparse matrix from reviews (hint: use original org/user_ids as indices)
        :param X:
        :return:
        """
        raise NotImplementedError('TASK')

    def fit(self, X, y=None):
        """
        tune embeddings for self.n_iter iterations, record loss history
        :param X:
        :param y:
        :return:
        """
        super().fit(X)
        self._history = []
        raise NotImplementedError('TASK')
        return self

    def predict_user_org(self, users, orgs):
        """
        use embeddings to compute user predictions (ordered by reconstructed score)
        :param users: predict for these users
        :param orgs: use orgs from this set
        :return:
        """
        return pd.Series(index=users, data=..., name='prediction')
