import secrets
from collections import defaultdict
from copy import copy

import numpy as np
import pandas as pd

from src.cross_val import DaysTimeSeriesSplit
from src.data_loading import Data


def rel_per_user(series, k=20):
    """
    computes relevance per user
    :param series: series (use as dict) representing a row from predictions & scores dataframe
    :param k: how many predictions to consider
    :return: np.ndarray(dtype=bool) with encoded relevance
    """
    predictions = series['predictions'][:k]
    targets = series['targets']
    return np.isin(predictions, targets)


def mnap_at_k_per_user(series, k=20):
    """
    computes MNAP per user (use relevance), should output `nan` if no interactions for given user are in target
    :param series: series (use as dict) representing a row from predictions & scores dataframe
    :param k: how many predictions to consider
    :return: MNAP score (float [0..1])
    """
    relevance = series['rel'][:k]
    if not np.any(relevance):
        return np.nan

    precision_at_k = np.cumsum(relevance) / (np.arange(len(relevance)) + 1)
    avg_precision = np.sum(precision_at_k[relevance]) / np.min([len(series['targets']), k])
    return avg_precision


def hit_rate_per_user(series):
    """
    computes hit rate per user (use relevance), should output `nan` if no interactions for given user are in target
    :param series: series (use as dict) representing a row from predictions & scores dataframe
    :return: hit rate score (float [0..1])
    """
    relevance = series['rel']
    if len(series['targets']) == 0:
        return np.nan
    return np.any(relevance)


def mrr_per_user(series):
    """
    computes MRR per user (use relevance), should output `nan` if no interactions for given user are in target
    :param series: series (use as dict) representing a row from predictions & scores dataframe
    :return: mrr score (float [0..1])
    """
    relevance = series['rel']

    if series['targets'].size == 0:
        return np.nan
    if not np.any(relevance):
        return 0
    return 1 / (np.argmax(relevance) + 1)


def calc_coverage(exploded_predict, orgs):
    """
    computes coverage over entire prediction series
    :param exploded_predict: exploded (one prediction per row) prediction series
    :param orgs: dataframe with org information
    :return: coverage score (float [0..1])
    """
    unique_predictions = exploded_predict.unique()
    total_orgs = orgs['org_id'].nunique()
    return len(unique_predictions) / total_orgs


def calc_surprisal(exploded_predict, x_train):
    """
    computes surprisal over entire dataset
    :param exploded_predict: exploded (one prediction per row) prediction series
    :param x_train: interactions (reviews) dataframe
    :return: surprisal score (float [0..1])
    """
    all_interactions = len(x_train)

    # количество взаимодействий с каждым элементом
    item_counts = x_train['org_id'].value_counts()

    def self_information(item):
        item_count = item_counts.get(item, 0)
        return -np.log2(max(item_count, 1) / all_interactions)

    users = exploded_predict.index.unique()

    user_surprisals = []
    for user in users:
        user_predictions = exploded_predict.loc[user]
        if not isinstance(user_predictions, pd.Series):
            user_predictions = pd.Series([user_predictions])
        recommendations = len(user_predictions)
        user_self_info = user_predictions.apply(self_information).sum()
        user_surprisals.append(user_self_info / recommendations)

    return (1 / (len(users) * np.log2(all_interactions))) * sum(user_surprisals)


class Scorer:
    def __init__(self, k: int, cv_splitter: DaysTimeSeriesSplit, data: Data):
        self.k = k
        self._cv_splitter = cv_splitter
        self._data = data
        self._score_table = pd.DataFrame()

    def leaderboard(self):
        return self._score_table

    def scoring_fn(self, predict, x_train, x_test):
        predict = predict.rename('predictions')
        x_test_with_cities = (x_test
                              .join(self._data.users.set_index('user_id')['city'], on='user_id')
                              .join(self._data.organisations.set_index('org_id')['city'], on='org_id',
                                    rsuffix='_user',
                                    lsuffix='_org'))
        list_compressed_target = x_test_with_cities.groupby('user_id').apply(
            lambda s: np.array(s['org_id'][(s['rating'] >= 4.0) & (s['city_user'] != s['city_org'])]),
            include_groups=False).rename('targets')
        joined = pd.merge(predict, list_compressed_target, left_index=True, right_index=True)
        joined['rel'] = joined.apply(rel_per_user, k=self.k, axis=1)
        joined['mnap'] = joined.apply(mnap_at_k_per_user, k=self.k, axis=1)
        joined['hit_rate'] = joined.apply(hit_rate_per_user, axis=1)
        joined['mrr'] = joined.apply(mrr_per_user, axis=1)
        metric_vals = joined[['mnap', 'hit_rate', 'mrr']].mean()
        exploded_predict: pd.Series = predict.explode()
        metric_vals['coverage'] = calc_coverage(exploded_predict, self._data.organisations)
        metric_vals['surprisal'] = calc_surprisal(exploded_predict, x_train)
        return metric_vals

    def score(self, model, experiment_name=None):
        if experiment_name is None:
            experiment_name = secrets.token_urlsafe(8)
        X = self._data.reviews
        metrics = defaultdict(list)
        for train_index, test_index in self._cv_splitter.split(X):
            X_train = X.iloc[train_index]
            X_test = X.iloc[test_index]
            fitted_model = copy(model).fit(X_train)
            predict = fitted_model.predict(X_test[['user_id']].drop_duplicates())
            current_fold_score = self.scoring_fn(predict, X_train, X_test)
            for key, value in current_fold_score.items():
                metrics[key].append(value)
        log_dict = {
            'name': experiment_name,
            'model': str(model),
        }
        for key, value in metrics.items():
            mean_score = np.mean(value)
            std_score = np.std(value)
            log_dict.update(
                {
                    key + '_mean': mean_score,
                    key + '_std': std_score
                }
            )
        self._score_table = self._score_table._append(log_dict, ignore_index=True)
        return log_dict
