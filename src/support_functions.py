# Common imports
import random

import numpy as np
import pandas as pd

from src.policy import Policy


def generate_troubled_life_policy_data(no_of_policies, runtime, file_path):
    with open(file_path, 'w') as csvfile:
        policies = [Policy(id, 100000) for id in range(1001, 1001 + no_of_policies)]

        Policy.write_header_to_csv(csvfile)

        for policy in policies:
            policy.write_to_csv(csvfile)

        for year in range(runtime):
            for policy in policies:
                policy.collect_premium(year)
                policy.write_to_csv(csvfile)

            for policy in random.sample(policies, no_of_policies // 10):
                policy.adapt_premium(random.uniform(0.2, 0.3))
                policy.write_to_csv(csvfile)

            for policy in policies:
                policy.add_yearly_interest()
                policy.write_to_csv(csvfile)


def load_troubled_life_policy_data(file_path):
    policy_histories = pd.read_csv(filepath_or_buffer=file_path, header=0, index_col=[0, 1, 2])

    return policy_histories.sort_index()


def get_policy_history_lengths(policy_histories):
    # Collect the length of each policy's history
    policy_histories_lengths = np.array([(id, policy_histories.loc[id].shape[0]) for id in policy_histories.index.levels[0]])
    max_policy_history_length = policy_histories_lengths.max(axis=0)[1]

    return policy_histories_lengths, max_policy_history_length


def pad_troubled_life_policy_histories(policy_histories, policy_histories_lengths, max_policy_history_length):
    # Let's zero-padd all policy histories to the maximum number of versions
    for id, policy_history_length in policy_histories_lengths:
        # Append at the end, set id to current group and current_year to 10000+
        # such that sorting will definitely move all padded rows to the end of the group
        for i in range(max_policy_history_length - policy_history_length):
            policy_histories.loc[id, 10000 + i, 0] = ["pad", 0, 0, False]

    return policy_histories.sort_index()


def prepare_labels_features_lengths(policy_histories, policy_histories_lengths, max_policy_history_length):
    # Extract features and labels from dataset as numpy.ndarray(s)
    features = policy_histories[['premium', 'current_capital']].as_matrix()
    labels = policy_histories['troubled'].replace(to_replace=[False, True], value=[0, 1]).as_matrix()

    # Remove column with policy ids (history_lengths is a numpy.ndarray)
    seq_lengths = policy_histories_lengths[:, 1]

    # Reshape labels from (overall_no_of_histories) to (overall_no_of_policies, maximum_history_length) and take maximum of each row (0 or 1)
    labels = labels.reshape(policy_histories.index.levels[0].shape[0], -1).max(axis=1)

    # Reshape features from (overall_no_of_histories, 2) to (overall_no_of_policies, maximum_history_length, 2)
    features = features.reshape((policy_histories.index.levels[0].shape[0], max_policy_history_length, -1))

    return labels, features, seq_lengths

class TrainDataSet:
    def __init__(self, train_labels, train_features, train_seq_lengths):
        self._labels = train_labels
        self._features = train_features
        self._seq_lengths = train_seq_lengths
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = train_features.shape[0]

    @property
    def features(self):
        return self._features

    @property
    def labels(self):
        return self._labels

    @property
    def seq_lengths(self):
        return self._seq_lengths

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, shuffle=True):
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._labels = self.labels[perm0]
            self._features = self.features[perm0]
            self._seq_lengths = self.seq_lengths[perm0]

        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            labels_rest_part = self._labels[start:self._num_examples]
            features_rest_part = self._features[start:self._num_examples]
            seq_lengths_rest_part = self._seq_lengths[start:self._num_examples]

            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._labels = self.labels[perm]
                self._features = self.features[perm]
                self._seq_lengths = self.seq_lengths[perm]

            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch

            labels_new_part = self._labels[start:end]
            features_new_part = self._features[start:end]
            seq_lengths_new_part = self._seq_lengths[start:end]

            return np.concatenate((labels_rest_part, labels_new_part), axis=0), \
                   np.concatenate((features_rest_part, features_new_part), axis=0), \
                   np.concatenate((seq_lengths_rest_part, seq_lengths_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch

            return self._labels[start:end], self._features[start:end], self._seq_lengths[start:end]