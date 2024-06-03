from unittest import TestCase

import numpy as np
import scipy.sparse as sps
import scipy.spatial.distance as spd
from sklearn.pipeline import Pipeline

import secret
from src.data_loading import load_data
from src.models.memory_based import cosine_similarity, pearson_similarity, UserBasedRecommender, ItemBasedRecommender
from src.transforms import UserInfoJoiner, OrgInfoJoiner


class TestMemoryBased(TestCase):
    def test_cosine_similarity(self):
        with self.subTest('regular'):
            a = np.array([[1, 2, 3, 4], [2, 2, 5, 4], [3, 4, 3, 5]], dtype=float)
            a_csr = sps.csr_array(a)
            sim = cosine_similarity(a_csr, a_csr)
            distance = spd.cdist(a, a, 'cosine')
            ref_sim = 1 - distance - np.eye(distance.shape[0])
            ref_sim = ref_sim / (np.abs(ref_sim).sum(0, keepdims=True) + 1e-8)
            self.assertTrue(np.allclose(sim.todense(), ref_sim))

        with self.subTest('sparse'):
            a = np.array([[1, 0, 3, 0], [0, 2, 5, 0], [0, 0, 0, 5]], dtype=float)
            a_csr = sps.csr_array(a)
            a_csr.eliminate_zeros()
            sim = cosine_similarity(a_csr, a_csr)
            distance = spd.cdist(a, a, 'cosine')
            ref_sim = 1 - distance - np.eye(distance.shape[0])
            ref_sim = ref_sim / (np.abs(ref_sim).sum(0, keepdims=True) + 1e-8)
            self.assertTrue(np.allclose(sim.todense(), ref_sim))

        # with self.subTest('against reference impl'):
        #     rng = np.random.default_rng(0xBAD_5EED)
        #     for _ in range(10):
        #         a = sps.random_array((10, 10), density=0.25, format='csr', random_state=rng)
        #         self.assertTrue(
        #             np.allclose(
        #                 cosine_similarity(a, a).todense(),
        #                 secret.cosine_similarity(a, a).todense()
        #             )
        #         )

    def test_pearson_similarity(self):
        with self.subTest('regular'):
            a = np.array([[1, 2, 3, 4], [2, 2, 5, 4], [3, 4, 3, 5]], dtype=float)
            a_csr = sps.csr_array(a)
            sim = pearson_similarity(a_csr, a_csr)
            distance = spd.cdist(a, a, 'correlation')
            ref_sim = 1 - distance - np.eye(distance.shape[0])
            ref_sim = ref_sim / (np.abs(ref_sim).sum(0, keepdims=True) + 1e-8)
            self.assertTrue(np.allclose(sim.todense(), ref_sim))

        with self.subTest('with_zeros'):
            a = np.array([[1, 0, 3, 0], [0, 2, 5, 0], [0, 0, 0, 5]], dtype=float)
            b = np.array([[0, 1, 2, 3], [2, 3, 0, 5], [0, 0, 0, 1]], dtype=float)
            a_csr = sps.csr_array(a)
            b_csr = sps.csr_array(b)

            sim = pearson_similarity(a_csr, b_csr)
            distance = spd.cdist(a, b, 'correlation')
            ref_sim = 1 - distance - np.eye(distance.shape[0])
            ref_sim = ref_sim / (np.abs(ref_sim).sum(0, keepdims=True) + 1e-8)
            self.assertTrue(np.allclose(sim.todense(), ref_sim))

        # with self.subTest('against reference impl'):
        #     rng = np.random.default_rng(0xBAD_5EED)
        #     for _ in range(10):
        #         a = sps.random_array((10, 10), density=0.25, format='csr', random_state=rng)
        #         self.assertTrue(
        #             np.allclose(
        #                 pearson_similarity(a, a).todense(),
        #                 secret.pearson_similarity(a, a).todense()
        #             )
        #         )

    def test_user_based(self):
        user_based_recommender = UserBasedRecommender(similarity_measure='cosine', n_items=2)
        with self.subTest('selecting'):
            matrix = sps.csr_matrix([
                [1, 0, 2, 3, 4],
                [2, 3, 0, 4, 5],
                [1, 2, 4, 0, 5],
                [1, 2, 3, 5, 0],
            ])
            matrix.eliminate_zeros()

            users = [1, 2]
            user_selection = user_based_recommender.select_users(matrix, users)
            answer = np.array([
                [0, 0, 0, 0, 0],
                [2, 3, 0, 4, 5],
                [1, 2, 4, 0, 5],
                [0, 0, 0, 0, 0],
            ])
            self.assertTrue(np.allclose(user_selection.todense(), answer))

            orgs = [2, 3]
            org_selection = user_based_recommender.select_orgs(matrix, orgs)
            answer = np.array([
                [0, 0, 2, 3, 0],
                [0, 0, 0, 4, 0],
                [0, 0, 4, 0, 0],
                [0, 0, 3, 5, 0],
            ])
            self.assertTrue(np.allclose(org_selection.todense(), answer))

        with self.subTest('compute rating'):
            matrix = sps.csr_matrix([
                [0, 1, 1, 1],  # twice as greater weight
                [0, 0, 1, 1],  # user of interest
                [1, 1, 0, 1],  # equally ->
                [1, 1, 1, 0],  # -> similar
            ], dtype=float)
            matrix.eliminate_zeros()
            rating = user_based_recommender.compute_rating(matrix, orgs=[0, 1], users=[1])
            self.assertAlmostEqual(rating[1, 0], 0.5)
            self.assertAlmostEqual(rating[1, 1], 1.0)

        with self.subTest('regular'):
            matrix = sps.csr_matrix([
                [0, 1, 1, 1],  # twice as greater weight
                [0, 0, 1, 1],  # user of interest
                [1, 1, 0, 1],  # equally ->
                [1, 1, 1, 0],  # -> similar
            ], dtype=float)
            matrix.eliminate_zeros()
            user_based_recommender._x = matrix
            prediction = user_based_recommender.predict_user_org(users=[1], orgs=[0, 1])
            self.assertTrue(np.all(prediction.loc[1] == np.array([1, 0])))
            # first has rating of 1 and 0th has rating of 0.5 as per previous test case

        # with self.subTest('against reference impl'):
        #     data = load_data('test_data/original_data_sample')
        #     rng = np.random.default_rng(0xBAD_5EED)
        #     for _ in range(10):
        #         random_review_sample = data.reviews.sample(frac=0.3, random_state=rng)
        #         pipeline = Pipeline([
        #             ('user_info_joiner', UserInfoJoiner(user_info=data.users)),
        #             ('org_info_joiner', OrgInfoJoiner(org_info=data.organisations)),
        #             ('popular_recommender', UserBasedRecommender(n_items=20, similarity_measure='cosine'))
        #         ])
        #         reference_pipeline = Pipeline([
        #             ('user_info_joiner', UserInfoJoiner(user_info=data.users)),
        #             ('org_info_joiner', OrgInfoJoiner(org_info=data.organisations)),
        #             ('popular_recommender', secret.UserBasedRecommender(n_items=20, similarity_measure='cosine'))
        #         ])
        #         predict_for = \
        #             random_review_sample.join(
        #                 data.users.set_index('user_id'), on='user_id', how='left'
        #             ).groupby('city').sample(1, random_state=rng)['user_id'].reset_index()
        #
        #         predictions = pipeline.fit(random_review_sample).predict(predict_for)
        #         reference_predictions = reference_pipeline.fit(random_review_sample).predict(predict_for)
        #         for user in predictions.index:
        #             self.assertTrue(np.allclose(reference_predictions.loc[user], predictions.loc[user]))

    def test_item_based(self):
        item_based_recommender = ItemBasedRecommender(similarity_measure='cosine', n_items=2)
        with self.subTest('selecting'):
            matrix = sps.csr_matrix([
                [1, 0, 2, 3, 4],
                [2, 3, 0, 4, 5],
                [1, 2, 4, 0, 5],
                [1, 2, 3, 5, 0],
            ]).T
            matrix.eliminate_zeros()

            users = [1, 2]
            user_selection = item_based_recommender.select_users(matrix, users)
            answer = np.array([
                [0, 0, 0, 0, 0],
                [2, 3, 0, 4, 5],
                [1, 2, 4, 0, 5],
                [0, 0, 0, 0, 0],
            ]).T
            self.assertTrue(np.allclose(user_selection.todense(), answer))

            orgs = [2, 3]
            org_selection = item_based_recommender.select_orgs(matrix, orgs)
            answer = np.array([
                [0, 0, 2, 3, 0],
                [0, 0, 0, 4, 0],
                [0, 0, 4, 0, 0],
                [0, 0, 3, 5, 0],
            ]).T
            self.assertTrue(np.allclose(org_selection.todense(), answer))

        with self.subTest('compute rating'):
            matrix = sps.csr_matrix([
                [0, 1, 1, 1],  # twice as greater weight
                [0, 0, 1, 1],  # item of interest
                [1, 1, 0, 1],  # equally ->
                [1, 1, 1, 0],  # -> similar
            ], dtype=float)
            matrix.eliminate_zeros()
            rating = item_based_recommender.compute_rating(matrix, orgs=[1], users=[0, 1])
            self.assertAlmostEqual(rating[0, 1], 0.5)
            self.assertAlmostEqual(rating[1, 1], 1.0)

        with self.subTest('regular'):
            matrix = sps.csr_matrix([
                [0, 1, 1, 1],  # twice as greater weight
                [0, 0, 1, 1],  # item of interest
                [1, 1, 0, 1],  # equally ->
                [1, 1, 1, 0],  # -> similar
            ], dtype=float)
            matrix.eliminate_zeros()
            item_based_recommender._x = matrix
            prediction = item_based_recommender.predict_user_org(users=[1], orgs=[0, 1])
            self.assertTrue(np.all(prediction.loc[1] == np.array([1, 0])))
            # first has rating of 1 and 0th has rating of 0.5 as per previous test case
