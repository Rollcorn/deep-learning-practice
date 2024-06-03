import unittest
from collections import Counter

import numpy as np
import pandas as pd

from data_loading import create_remap, load_data


def compare_ndarray_sequences_up_to_permutation(left, right):
    return Counter(elem.tobytes() for elem in left) == Counter(elem.tobytes() for elem in right)


class TestDataLoading(unittest.TestCase):
    def setUp(self):
        self.data = load_data('test_data/original_data_sample')

    def test_remap(self):
        random_series = pd.Series(index=np.arange(100), data=np.random.default_rng(42).permutation(100))
        forward_mapping, backward_mapping = create_remap(random_series)
        mapped_series = random_series.map(forward_mapping)
        self.assertTrue(np.all(np.sort(mapped_series) == np.arange(100)))
        restored_values = backward_mapping[mapped_series.values]
        self.assertTrue(np.all(random_series.values == restored_values))

    def test_inverse_mappings(self):
        original_reviews = pd.read_csv('test_data/original_data_sample/reviews.csv')

        self.assertTrue(Counter(original_reviews['user_id'].values) ==
                        Counter(self.data.map_users_back(self.data.reviews['user_id']).values))

        def generate_fake_homogeneous_predictions(user_org_pairs_df):
            return user_org_pairs_df.groupby('user_id').apply(lambda group: group['org_id'].values[:1])

        orgs_lists_homogeneous = generate_fake_homogeneous_predictions(self.data.reviews)
        orgs_lists_homogeneous_remapped = self.data.map_organizations_back(orgs_lists_homogeneous)
        orgs_lists_homogeneous_original = generate_fake_homogeneous_predictions(original_reviews)
        self.assertTrue(compare_ndarray_sequences_up_to_permutation(
            orgs_lists_homogeneous_remapped.values,
            orgs_lists_homogeneous_original.values
        ))

        def generate_fake_heterogeneous_predictions(user_org_pairs_df):
            return user_org_pairs_df.groupby('user_id').apply(lambda group: group['org_id'].values)

        orgs_lists_heterogeneous = generate_fake_heterogeneous_predictions(self.data.reviews)
        orgs_lists_heterogeneous_remapped = self.data.map_organizations_back(orgs_lists_heterogeneous)
        orgs_lists_heterogeneous_original = generate_fake_heterogeneous_predictions(original_reviews)
        self.assertTrue(compare_ndarray_sequences_up_to_permutation(
            orgs_lists_heterogeneous_remapped.values,
            orgs_lists_heterogeneous_original.values
        ))

if __name__ == "__main__":
    unittest.main()
