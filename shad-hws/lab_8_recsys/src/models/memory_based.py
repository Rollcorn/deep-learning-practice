import pandas as pd
import scipy.sparse as sps
from scipy.sparse import csr_matrix
import scipy.sparse.linalg as linal
import numpy as np
from src.models.based import EstimatorWithFallback
from src.models.statistics_based import PopularityRecommender


def cosine_similarity(matrix_1, matrix_2):
    """
    cosine similarity between two sparse matrices <u, v> / ||u|| / ||v||
    :param matrix_1:
    :param matrix_2:
    :return:
    """
    # mat_identity = csr_matrix(np.identity(matrix_1.shape[0]))
    # mat_ones = csr_matrix(np.ones((matrix_1.shape[0], matrix_1.shape[0])))
    if not isinstance(matrix_1, csr_matrix):
        matrix_1 = csr_matrix(matrix_1)
    if not isinstance(matrix_2, csr_matrix):
        matrix_2 = csr_matrix(matrix_2)

    norm_1 = linal.norm(matrix_1, axis=1)
    norm_2 = linal.norm(matrix_2, axis=1)

    norm_1[norm_1 == 0] = 1e-10
    norm_2[norm_2 == 0] = 1e-10

    matrix_1_normalized = matrix_1.multiply(1 / norm_1[:, None])
    matrix_2_normalized = matrix_2.multiply(1 / norm_2[:, None])
    cosine_sim = matrix_1_normalized.dot(matrix_2_normalized.T)

    cosine_sim.setdiag(0)
    sum_similarities = np.abs(cosine_sim).sum(axis=0)
    sum_similarities[sum_similarities == 0] = 1e-10
    cosine_sim = cosine_sim.multiply(1 / sum_similarities)

    return cosine_sim


def pearson_similarity(matrix_1, matrix_2):
    """
    pearson similarity between two sparse matrices <u - Eu, v - Ev> / Vu / Vv
    :param matrix_1:
    :param matrix_2:
    :return:
    """
    if not sps.issparse(matrix_1):
        matrix_1 = sps.csr_matrix(matrix_1)
    if not sps.issparse(matrix_2):
        matrix_2 = sps.csr_matrix(matrix_2)

    mask_1 = matrix_1.sign().toarray()
    mask_2 = matrix_2.sign().toarray()
    coef_corel = np.zeros((matrix_1.shape[0], matrix_2.shape[0]))
    for i in range(matrix_1.shape[0]):
        for j in range(matrix_2.shape[0]):
            mask_intersect = mask_1[i] * mask_2[j]
            # avg_1 = (np.sum(mask_intersect * matrix_1.toarray()[i, :])
            #          / np.where(np.sum(mask_intersect) == 0, 1, np.sum(mask_intersect)))
            avg_1 = (np.sum(mask_1[i] * matrix_1.toarray()[i, :])
                     / np.where(np.sum(mask_1[i]) == 0, 1, np.sum(mask_1[i])))
            avg_2 = (np.sum(mask_2[j] * matrix_2.toarray()[j, :])
                     / np.where(np.sum(mask_2[j]) == 0, 1, np.sum(mask_2[j])))

            v = mask_intersect * (matrix_2.toarray()[j, :] - avg_2)
            u = mask_intersect * (matrix_1.toarray()[i, :] - avg_1)
            coef_corel[i, j] = (np.sum(u * v)
                                / np.where(np.linalg.norm(u) < 1e-10, 1, np.linalg.norm(u))
                                / np.where(np.linalg.norm(v) < 1e-10, 1, np.linalg.norm(v)))

    similarity_matrix =sps.csr_matrix(coef_corel) - sps.eye(matrix_1.shape[0])
    norm_abs = sps.linalg.norm(similarity_matrix, ord=1, axis=0)[None, :]
    return similarity_matrix.multiply(1 / np.where(norm_abs < 1e-10, 1, norm_abs))

    # matrix_1_mean = matrix_1.sum(axis=1) / np.sum(matrix_1.todense() != 0, axis=1)
    # matrix_2_mean = matrix_2.sum(axis=1) / np.sum(matrix_2.todense() != 0, axis=1)
    # matrix_1 = matrix_1 - sps.csr_array(
    #     np.where(matrix_1.todense() == 0, 0, np.ones(matrix_1.shape) * matrix_1_mean[:, None])
    # )
    # matrix_2 = matrix_2 - sps.csr_array(
    #     np.where(matrix_2.todense() == 0, 0, np.ones(matrix_2.shape) * matrix_2_mean[:, None])
    # )
    #
    # norm1 = sps.linalg.norm(matrix_1, axis=1)[:, None]
    # norm2 = sps.linalg.norm(matrix_2, axis=1)[:, None]
    #
    # matrix_1 = matrix_1.multiply(1 / np.where(norm1 < 1e-10, 1, norm1))
    # matrix_2 = matrix_2.multiply(1 / np.where(norm2 < 1e-10, 1, norm2))
    #
    # # Compute the cosine similarity matrix
    # similarity_matrix = matrix_1.dot(matrix_2.transpose()) - sps.eye(matrix_1.shape[0])
    # norm_abs = sps.linalg.norm(similarity_matrix, ord=1, axis=0)[None, :]
    # return similarity_matrix.multiply(1 / np.where(norm_abs < 1e-10, 1, norm_abs))


class UserBasedRecommender(EstimatorWithFallback):
    def __init__(self, similarity_measure, n_items, fallback_estimator=PopularityRecommender, **kwargs):
        super().__init__(fallback_estimator, n_items=n_items, **kwargs)
        self.similarity_measure = similarity_measure
        if self.similarity_measure == 'cosine':
            self.similarity_fn = cosine_similarity
        elif self.similarity_measure == 'pearson':
            self.similarity_fn = pearson_similarity
        else:
            raise NotImplementedError(f'similarity measure {self.similarity_measure} is not implemented')
        self.n_items = n_items

    def make_sparse(self, X):
        """
        Make sparse matrix from reviews (hint: use original org/user_ids as indices)
        :param X:
        :return:
        """
        users = X['user_id'].unique()
        orgs = X['org_id'].unique()
        user_map = {user: i for i, user in enumerate(users)}
        org_map = {org: i for i, org in enumerate(orgs)}
        rows = X['user_id'].map(user_map)
        cols = X['org_id'].map(org_map)
        data = X['rating']
        matrix = csr_matrix((data, (rows, cols)), shape=(len(users), len(orgs)))
        return matrix

    def fit(self, X, y=None):
        super().fit(X)
        self._x = self.make_sparse(X)
        return self

    def select_users(self, matrix, users):
        """helper function which selects users, e.g. zeroes out the unused part of matrix"""
        mask = np.zeros(matrix.shape[0], dtype=bool)
        mask[users] = True
        user_selection = matrix.multiply(mask[:, None])
        return user_selection

    def select_orgs(self, matrix, orgs):
        """helper function which selects orgs, e.g. zeroes out the unused part of matrix"""
        mask = np.zeros(matrix.shape[1], dtype=bool)
        mask[orgs] = True
        org_selection = matrix.multiply(mask)
        return org_selection

    def predict_user_org(self, users, orgs):
        rating = self.compute_rating(self._x, orgs, users)
        rating.eliminate_zeros()
        if rating.nnz > 0:
            ranking_df = rating.tocoo()
            ranking_df = pd.DataFrame({'rating': ranking_df.data, 'user_id': ranking_df.row, 'org_id': ranking_df.col})
            prediction = ranking_df.groupby('user_id').apply(
                lambda group: group.nlargest(self.n_items, columns='rating')['org_id'].values, include_groups=False)
        else:
            prediction = pd.Series()
        return prediction

    def compute_rating(self, matrix, orgs, users):
        """
        compute the actual rating given by similar users weighted by their similarity (only for interactions)
        :param matrix:
        :param orgs:
        :param users:
        :return:
        """
        user_similarities = self.similarity_fn(matrix, matrix)
        user_similarities = self.select_users(user_similarities, users)
        selected_users_ratings = self.select_orgs(matrix, orgs)
        weighted_ratings = user_similarities.dot(selected_users_ratings)
        sum_similarities = user_similarities.sum(axis=1).A1
        sum_similarities[sum_similarities == 0] = 1e-10
        predicted_ratings = weighted_ratings.multiply(1 / sum_similarities[:, None])
        return predicted_ratings.tocsr()


class ItemBasedRecommender(UserBasedRecommender):
    def make_sparse(self, X):
        """
        Make sparse matrix from reviews (hint: use original org/user_ids as indices)
        :param X:
        :return:
        """
        raise NotImplementedError('TASK')

    def select_users(self, matrix, users):
        """the same as in user based"""
        raise NotImplementedError('TASK')

    def select_orgs(self, matrix, orgs):
        """the same as in user based"""
        raise NotImplementedError('TASK')

    def compute_rating(self, matrix, orgs, users):
        """the same as in user based"""
        raise NotImplementedError('TASK')
