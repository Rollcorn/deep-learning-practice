import os
from dataclasses import dataclass, field
from functools import cache

import numpy as np
import pandas as pd


def create_remap(series: pd.Series) -> tuple[dict, np.ndarray]:
    """
    Return two mappers from ids to the smallest integer set and back (one dict and one ndarray for speed).
    :param series: series with ids
    :return: forward mapping dict and reverse mapping ndarray
    """
    unique_ids = series.unique()
    forward_mapping = {id_: idx for idx, id_ in enumerate(unique_ids)}
    reverse_mapping = np.array(unique_ids)
    return forward_mapping, reverse_mapping


@dataclass
class Data:
    reviews: pd.DataFrame
    users: pd.DataFrame
    organisations: pd.DataFrame
    features: pd.DataFrame
    aspects: pd.DataFrame
    rubrics: pd.DataFrame
    test_users: pd.DataFrame

    __user_reverse_mapping: np.ndarray = field(init=False)
    __organisation_reverse_mapping: np.ndarray = field(init=False)

    def __post_init__(self):
        user_forward_mapping, self.__user_reverse_mapping = create_remap(self.users['user_id'])
        org_forward_mapping, self.__organisation_reverse_mapping = create_remap(self.organisations['org_id'])

        self.users['user_id'] = self.users['user_id'].map(user_forward_mapping.get).values
        self.organisations['org_id'] = self.organisations['org_id'].map(org_forward_mapping.get).values
        self.reviews['user_id'] = self.reviews['user_id'].map(user_forward_mapping.get).values
        self.reviews['org_id'] = self.reviews['org_id'].map(org_forward_mapping.get).values
        self.test_users['user_id'] = self.test_users['user_id'].map(user_forward_mapping.get).values

    def map_users_back(self, user_ids: pd.Series) -> pd.Series:
        """
        Map users back to their original IDs.
        :param user_ids: pd.Series of user IDs in integer representation.
        :return: pd.Series of user IDs in original representation.
        """
        new_values = self.__user_reverse_mapping[user_ids]
        return pd.Series(new_values, name=user_ids.name, index=user_ids.index)

    def map_organizations_back(self, org_lists: pd.Series) -> pd.Series:
        """
        Map organizations back to their original IDs.
        :param org_lists: pd.Series containing uneven np.ndarrays of org IDs (in integer representation).
        :return: pd.Series containing uneven np.ndarrays of org IDs (in original meaning).
        """
        try:
            new_values = org_lists.apply(lambda org_list: self.__organisation_reverse_mapping[org_list])
            return pd.Series(new_values, name=org_lists.name, index=org_lists.index)
        except ValueError as _:
            return org_lists.apply(lambda org_list: self.__organisation_reverse_mapping[org_list])

@cache
def load_data(data_directory_path: str) -> Data:
    return Data(**{
        file.removesuffix('.csv'): pd.read_csv(os.path.join(data_directory_path, file), low_memory=False) for file in
        os.listdir(data_directory_path)
    })
