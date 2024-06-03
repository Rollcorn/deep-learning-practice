import unittest

import numpy as np
import pandas as pd

import secret
from src.scoring import rel_per_user, mrr_per_user, hit_rate_per_user, mnap_at_k_per_user, calc_coverage, \
    calc_surprisal


class TestScoring(unittest.TestCase):
    def test_rel_per_user(self):
        with self.subTest('regular case'):
            row = pd.Series({
                'predictions': np.array([1, 2, 3, 4, 5]),
                'targets': np.array([1, 3, 4, 6, 7])
            })
            answer = np.array([True, False, True, True, False])
            self.assertTrue(np.all(rel_per_user(row, k=5) == answer))
        with self.subTest('zero rel'):
            row = pd.Series({
                'predictions': np.array([1, 2, 3, 4, 5]),
                'targets': np.array([6, 7, 8, 9])
            })
            answer = np.array([False, False, False, False, False])
            self.assertTrue(np.all(rel_per_user(row, k=5) == answer))
        with self.subTest('many predictions'):
            row = pd.Series({
                'predictions': np.arange(100),
                'targets': np.array([3])
            })
            answer = np.array([False, False, False, True, False])
            self.assertTrue(np.all(rel_per_user(row, k=5) == answer))

        # with self.subTest('against reference impl'):
        #     rng = np.random.default_rng(0xBAD_5EED)
        #     for _ in range(10):
        #         row = pd.Series({
        #             'predictions': rng.choice(50, size=20, replace=False),
        #             'targets': rng.choice(50, size=15, replace=False)
        #         })
        #         self.assertTrue(np.all(rel_per_user(row, k=20) == secret.rel_per_user(row, k=20)))

    def test_mnap_per_user(self):
        with self.subTest('no targets'):
            row = pd.Series({
                'predictions': np.array([1, 2, 3, 4, 5]),
                'targets': np.array([], dtype=int)
            })
            row['rel'] = rel_per_user(row, k=5)
            self.assertTrue(np.isnan(mnap_at_k_per_user(row, k=5)))

        with self.subTest('regular case'):
            row = pd.Series({
                'predictions': np.array([1, 2, 3, 4, 5]),
                'targets': np.array([2, 4, 6, 7, 8, 9])
            })
            row['rel'] = rel_per_user(row, k=5)
            answer = 1 / 5 * (1 / 2 + 2 / 4)
            self.assertAlmostEqual(mnap_at_k_per_user(row, k=5), answer)

        with self.subTest('not enough targets'):
            row = pd.Series({
                'predictions': np.array([1, 2, 3, 4, 5]),
                'targets': np.array([2, 4])
            })
            row['rel'] = rel_per_user(row, k=5)
            answer = 1 / 2 * (1 / 2 + 2 / 4)
            self.assertAlmostEqual(mnap_at_k_per_user(row, k=5), answer)

        # with self.subTest('against reference impl'):
        #     rng = np.random.default_rng(0xBAD_5EED)
        #     for _ in range(10):
        #         row = pd.Series({
        #             'predictions': rng.choice(50, size=20, replace=False),
        #             'targets': rng.choice(50, size=15, replace=False)
        #         })
        #         row['rel'] = rel_per_user(row, k=20)
        #         self.assertAlmostEqual(mnap_at_k_per_user(row, k=20), secret.mnap_at_k_per_user(row, k=20))

    def test_hit_rate_per_user(self):
        with self.subTest('no targets'):
            row = pd.Series({
                'predictions': np.array([1, 2, 3, 4, 5]),
                'targets': np.array([], dtype=int)
            })
            row['rel'] = rel_per_user(row, k=5)
            self.assertTrue(np.isnan(hit_rate_per_user(row)))

        with self.subTest('regular case'):
            row = pd.Series({
                'predictions': np.array([1, 2, 3, 4, 5]),
                'targets': np.array([2, 4, 6, 7, 8, 9])
            })
            row['rel'] = rel_per_user(row, k=5)
            answer = 1
            self.assertAlmostEqual(hit_rate_per_user(row), answer)

            row = pd.Series({
                'predictions': np.array([1, 2, 3, 4, 5]),
                'targets': np.array([6, 7, 8, 9])
            })
            row['rel'] = rel_per_user(row, k=5)
            answer = 0
            self.assertAlmostEqual(hit_rate_per_user(row), answer)

        # with self.subTest('against reference impl'):
        #     rng = np.random.default_rng(0xBAD_5EED)
        #     for _ in range(10):
        #         row = pd.Series({
        #             'predictions': rng.choice(50, size=20, replace=False),
        #             'targets': rng.choice(50, size=15, replace=False)
        #         })
        #         row['rel'] = rel_per_user(row, k=20)
        #         self.assertAlmostEqual(hit_rate_per_user(row), secret.hit_rate_per_user(row))

    def test_mrr_per_user(self):
        with self.subTest('no targets'):
            row = pd.Series({
                'predictions': np.array([1, 2, 3, 4, 5]),
                'targets': np.array([], dtype=int)
            })
            row['rel'] = rel_per_user(row, k=5)
            self.assertTrue(np.isnan(mrr_per_user(row)))

        with self.subTest('regular case'):
            row = pd.Series({
                'predictions': np.array([1, 2, 3, 4, 5]),
                'targets': np.array([2, 4, 6, 7, 8, 9])
            })
            row['rel'] = rel_per_user(row, k=5)
            answer = 1 / 2
            self.assertAlmostEqual(mrr_per_user(row), answer)

            row = pd.Series({
                'predictions': np.array([1, 2, 3, 4, 5]),
                'targets': np.array([6, 7, 8, 9])
            })
            row['rel'] = rel_per_user(row, k=5)
            answer = 0
            self.assertAlmostEqual(mrr_per_user(row), answer)

        # with self.subTest('against reference impl'):
        #     rng = np.random.default_rng(0xBAD_5EED)
        #     for _ in range(10):
        #         row = pd.Series({
        #             'predictions': rng.choice(50, size=20, replace=False),
        #             'targets': rng.choice(50, size=15, replace=False)
        #         })
        #         row['rel'] = rel_per_user(row, k=20)
        #         self.assertAlmostEqual(mrr_per_user(row), secret.mrr_per_user(row))

    def test_coverage(self):
        with self.subTest('degenerate prediction'):
            rows = pd.Series([np.array([1, 2, 3, 4, 5])] * 10, name='predictions')
            orgs = pd.DataFrame({'org_id': np.arange(50) + 1, 'city': ['msk'] * 50})
            answer = 5 / 50
            self.assertAlmostEqual(calc_coverage(rows.explode(), orgs), answer)

        with self.subTest('good and bad coverage'):
            rows = pd.Series([np.array([1, 2, 3, 4, 5])] * 5 + [np.array([6])] * 5, name='predictions')
            orgs = pd.DataFrame(
                {'org_id': np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), 'city': ['msk'] * 5 + ['spb'] * 5})
            answer = 6 / 10
            self.assertAlmostEqual(calc_coverage(rows.explode(), orgs), answer)

        # with self.subTest('against reference impl'):
        #     rng = np.random.default_rng(0xBAD_5EED)
        #     for _ in range(10):
        #         rows = pd.Series([
        #             rng.choice(50, size=5, replace=False) for __ in range(10)
        #         ])
        #         orgs = pd.DataFrame({'org_id': np.arange(50), 'city': rng.choice(['msk', 'spb'], size=50)})
        #         exploded = rows.explode()
        #         self.assertAlmostEqual(calc_coverage(exploded, orgs), secret.calc_coverage(exploded, orgs))

    def test_surprisal(self):
        with self.subTest('regular case'):
            rows = pd.Series(
                data=[np.array([1, 2, 3, 4, 5]), np.array([1, 2, 3]), np.array([4, 5])],
                index=pd.Index(data=[1, 2, 3], name='user_id'),
                name='predictions'
            )
            train_data = pd.DataFrame({'org_id': [1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5]})
            N = (5 + 4 + 3 + 2 + 1)
            answer = 1 / 3 * (
                    1 / (5 * np.log2(N)) * (
                    - np.log2(5 / N)
                    - np.log2(4 / N)
                    - np.log2(3 / N)
                    - np.log2(2 / N)
                    - np.log2(1 / N)
            ) +
                    1 / (3 * np.log2(N)) * (
                            - np.log2(5 / N)
                            - np.log2(4 / N)
                            - np.log2(3 / N)
                    ) +
                    1 / (2 * np.log2(N)) * (
                            - np.log2(2 / N)
                            - np.log2(1 / N)
                    )
            )
            self.assertAlmostEqual(calc_surprisal(rows.explode(), train_data), answer)

        with self.subTest('very new predictions'):
            rows = pd.Series(
                data=[np.array([4, 5, 6])],
                index=pd.Index(data=[1], name='user_id'),
                name='predictions'
            )
            train_data = pd.DataFrame({'org_id': [1, 1, 2, 2, 3, 3]})
            answer = 1
            self.assertAlmostEqual(calc_surprisal(rows.explode(), train_data), answer)

        # with self.subTest('against reference impl'):
        #     rng = np.random.default_rng(0xBAD_5EED)
        #     for _ in range(10):
        #         rows = pd.Series(
        #             data=[rng.choice(50, size=5, replace=False) for __ in range(10)],
        #             index=pd.Index(data=np.arange(10), name='user_id'),
        #             name='predictions'
        #         )
        #         train_data = pd.DataFrame({'org_id': rng.choice(50, size=100, replace=True)})
        #         self.assertAlmostEqual(calc_surprisal(rows.explode(), train_data),
        #                                secret.calc_surprisal(rows.explode(), train_data))
