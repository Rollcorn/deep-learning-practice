from unittest import TestCase

import numpy as np
import pandas as pd
import scipy as scp
import scipy.sparse.linalg
from sklearn.pipeline import Pipeline

from src.models.matrix_factorization_based import SVDRecommender, ALSRecommender
from src.transforms import UserInfoJoiner, OrgInfoJoiner
from tests import secret


class TestFactorizationBased(TestCase):
    def test_svd(self):
        with self.subTest('regular'):
            rng = np.random.default_rng(0xBAD_5EED)
            semi_u, semi_v = rng.standard_normal(size=(100, 100)), rng.standard_normal(size=(100, 100))
            u, _ = scp.linalg.qr(semi_u)
            vh, _ = scp.linalg.qr(semi_v)
            u = u[:, :10]
            vh = vh[:10, :]
            sigma = np.arange(10, 0, -1)
            matrix = u * sigma[None, :] @ vh
            X = pd.DataFrame([
                {
                    'user_id': user_id,
                    'org_id': org_id,
                    'rating': matrix[user_id, org_id],
                    'org_city': rng.choice(['msk', 'spb']),
                    'user_city': rng.choice(['msk', 'spb']),
                } for user_id in range(100) for org_id in range(100)
            ])
            svd_model = SVDRecommender(n_components=10, n_items=20, random_state=rng)
            svd_model.fit(X)
            self.assertTrue(np.allclose(np.sort(svd_model.s), np.sort(sigma)))

        with self.subTest('against reference impl'):
            rng = np.random.default_rng(0xBAD_5EED)
            user_info = pd.DataFrame({'user_id': np.arange(100), 'city': ['msk'] * 50 + ['spb'] * 50})
            org_info = pd.DataFrame({'org_id': np.arange(100), 'city': ['msk'] * 50 + ['spb'] * 50})
            pipeline = Pipeline([
                ('user_info_joiner', UserInfoJoiner(user_info=user_info)),
                ('org_info_joiner', OrgInfoJoiner(org_info=org_info)),
                ('popular_recommender', SVDRecommender(n_components=5, n_items=20, random_state=rng))
            ])
            reference_pipeline = Pipeline([
                ('user_info_joiner', UserInfoJoiner(user_info=user_info)),
                ('org_info_joiner', OrgInfoJoiner(org_info=org_info)),
                ('popular_recommender', secret.SVDRecommender(n_components=5, n_items=20, random_state=rng))
            ])
            for test_idx in range(10):
                random_review_sample = scp.sparse.random_array((100, 100), density=0.75, format='coo', random_state=rng)
                random_review_sample = pd.DataFrame(
                    {'user_id': random_review_sample.row, 'org_id': random_review_sample.col,
                     'rating': random_review_sample.data})
                predict_for = pd.DataFrame(
                    {'user_id': rng.choice(100, size=20, replace=False)})

                predictions = pipeline.fit(random_review_sample).predict(predict_for)
                reference_predictions = reference_pipeline.fit(random_review_sample).predict(predict_for)
                for user in predictions.index:
                    self.assertTrue(len(np.intersect1d(predictions.loc[user], reference_predictions.loc[user])) > 10)

    def test_als_update(self):
        with self.subTest('correct shapes'):
            rng = np.random.default_rng(0xBAD_5EED)
            als = ALSRecommender(n_items=20, feature_dim=20, regularizer=1.0, num_iter=100, random_state=rng)
            user_embedding = rng.standard_normal(size=(100, 20))
            item_embedding = rng.standard_normal(size=(200, 20))
            ranking_matrix = scp.sparse.random_array((100, 200), density=0.75, format='csr', random_state=rng)
            als.compute_loss(user_embedding, item_embedding, ranking_matrix)
            self.assertEqual(als.update_other_embeddings(user_embedding, ranking_matrix.T).shape, item_embedding.shape)
            self.assertEqual(als.update_other_embeddings(item_embedding, ranking_matrix).shape, user_embedding.shape)
        with self.subTest('loss is lowering'):
            rng = np.random.default_rng(0xBAD_5EED)
            als = ALSRecommender(n_items=1, feature_dim=20, regularizer=0.0, num_iter=100, random_state=rng)
            ranking_matrix = scp.sparse.random_array((100, 200), density=0.75, format='coo', random_state=rng)
            X = pd.DataFrame(
                {'user_id': ranking_matrix.row, 'org_id': ranking_matrix.col, 'org_city': rng.choice(['msk', 'spb'], size=ranking_matrix.nnz),
                 'rating': ranking_matrix.data})
            als.fit(X)
            self.assertTrue(als._history[0] > als._history[1] > als._history[-1])
            reconstruction_loss = secret.reconstruction_loss(ranking_matrix, als.user_embeddings, als.item_embeddings)
            u, s, vh = scipy.sparse.linalg.svds(ranking_matrix, k=20)
            svd_reconstruction_loss = np.sum((u @ np.diag(s) @ vh - ranking_matrix) ** 2)
            self.assertTrue(
                reconstruction_loss < svd_reconstruction_loss * 1.1)  # 10 percent margin for als reconstruction

    def test_als(self):
        rng = np.random.default_rng(0xBAD_5EED)
        user_info = pd.DataFrame({'user_id': np.arange(100), 'city': ['msk'] * 50 + ['spb'] * 50})
        org_info = pd.DataFrame({'org_id': np.arange(100), 'city': ['msk'] * 50 + ['spb'] * 50})
        pipeline = Pipeline([
            ('user_info_joiner', UserInfoJoiner(user_info=user_info)),
            ('org_info_joiner', OrgInfoJoiner(org_info=org_info)),
            ('popular_recommender',
             ALSRecommender(feature_dim=5, n_items=20, random_state=rng, regularizer=0.0, num_iter=100))
        ])
        reference_pipeline = Pipeline([
            ('user_info_joiner', UserInfoJoiner(user_info=user_info)),
            ('org_info_joiner', OrgInfoJoiner(org_info=org_info)),
            ('popular_recommender',
             secret.ALSRecommender(feature_dim=5, n_items=20, random_state=rng, regularizer=0.0, num_iter=100))
        ])
        for test_idx in range(10):
            random_review_sample = scp.sparse.random_array((100, 100), density=0.75, format='coo', random_state=rng)
            random_review_sample = pd.DataFrame(
                {'user_id': random_review_sample.row, 'org_id': random_review_sample.col,
                 'rating': random_review_sample.data})
            predict_for = pd.DataFrame(
                {'user_id': rng.choice(100, size=20, replace=False)})

            predictions = pipeline.fit(random_review_sample).predict(predict_for)
            reference_predictions = reference_pipeline.fit(random_review_sample).predict(predict_for)
            for user in predictions.index:
                self.assertTrue(len(np.intersect1d(predictions.loc[user], reference_predictions.loc[user])) > 10)
