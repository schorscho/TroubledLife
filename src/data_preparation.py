# Common imports
import random

import numpy as np
import pandas as pd

from policy import Policy


def generate_life_policy_data(no_of_policies, runtime, trouble, file_path):
    with open(file_path, 'w') as csvfile:
        policies = [Policy(id, 100000 * random.uniform(1.0, 1.5), random.randint(1, 4)) for id in range(1001, 1001 + no_of_policies)]

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
                policy.add_yearly_interest(trouble=trouble)
                policy.write_to_csv(csvfile)

            print("generate: ", year)


def load_life_policy_data(file_path):
    policy_histories = pd.read_csv(filepath_or_buffer=file_path, header=0, index_col=[0, 1, 2])

    return policy_histories.sort_index()


def get_policy_history_lengths(policy_histories):
    # Collect the length of each policy's history
    policy_histories_lengths = np.array([(id, policy_histories.loc[id].shape[0]) for id in policy_histories.index.levels[0]])
    max_policy_history_length = policy_histories_lengths.max(axis=0)[1]

    return policy_histories_lengths, max_policy_history_length


def pad_life_policy_histories(policy_histories, policy_histories_lengths, max_policy_history_length):
    # Let's zero-padd all policy histories to the maximum number of versions
    for id, policy_history_length in policy_histories_lengths:
        # Append at the end, set id to current group and current_year to 10000+
        # such that sorting will definitely move all padded rows to the end of the group
        for i in range(max_policy_history_length - policy_history_length):
            policy_histories.loc[id, 10000 + i, 0] = ["pad", 0, 0, 0, 0, 0, False]

        print("pad: ", id)

    return policy_histories.sort_index()


def prepare_labels_features_lengths(policy_histories, policy_histories_lengths, max_policy_history_length, binary_classification=False):
    # Extract features and labels from dataset as numpy.ndarray(s)
    features = policy_histories[['premium', 'current_capital']].as_matrix()
    labels = policy_histories['troubled'].replace(to_replace=[False, True], value=[0, 1]).as_matrix()

    # Remove column with policy ids (history_lengths is a numpy.ndarray)
    seq_lengths = policy_histories_lengths[:, 1]

    # Reshape labels from (overall_no_of_histories) to (overall_no_of_policies, maximum_history_length) and take maximum of each row (0 or 1)
    labels = labels.reshape(policy_histories.index.levels[0].shape[0], -1).argmax(axis=1)

    if (binary_classification):
        labels = (labels > 0).astype(int)

    # Reshape features from (overall_no_of_histories, 2) to (overall_no_of_policies, maximum_history_length, 2)
    features = features.reshape((policy_histories.index.levels[0].shape[0], max_policy_history_length, -1))

    return labels, features, seq_lengths


def generate_Z_batch(size_batch, max_length_policy_history, n_inputs, runtime):
    #Z_batch = np.random.normal(10, 1, size=[size_batch, max_length_policy_history, n_inputs])
    #seq_length_z_batch = np.random.randint(low=1 + runtime * 2, high=max_length_policy_history + 1, size=size_batch)
    Z_batch = np.full(shape=[size_batch, max_length_policy_history, n_inputs], fill_value=0.0)
    seq_length_z_batch = np.full(shape=size_batch, fill_value=1 + runtime * 2)

    init_z_batch = np.full(shape=[size_batch, 5], fill_value=0.0)
    init_z_batch[:, 0] = np.random.uniform(low=1000.0, high=1500.0, size=size_batch)
    
    s = np.random.randint(low=1, high=2, size=size_batch)
    
    for i in range(size_batch):
        for j in range(1, 5):
            init_z_batch[i, j] = (s[i] == j)
    
    #init_z_batch[:, 0] = 1250.0
    #init_z_batch[:, 1] = 0.03

    
    Z_batch[:, 0, 0] = init_z_batch[:, 0]
    Z_batch[:, 0, 1] = init_z_batch[:, 1]
    Z_batch[:, 0, 2] = init_z_batch[:, 2]
    Z_batch[:, 0, 3] = init_z_batch[:, 3]
    Z_batch[:, 0, 4] = init_z_batch[:, 4]

    #for i in range(size_batch):
     #   Z_batch[i, seq_length_z_batch[i]:max_length_policy_history, :] = 0
    
    return Z_batch, seq_length_z_batch, init_z_batch


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


def numpy_correct(init_z, gData):
    correct = np.full(shape=(size_batch, max_length_policy_history, 2), fill_value=0.0)

    premium = init_z[:, 0]
    interest = init_z[:, 1] * 0.03 + init_z[:, 2] * 0.04 + init_z[:, 3] * 0.05 + init_z[:, 4] * 0.06

    k1 = premium
    k1v = k1 * (1 + interest)
    k2 = premium + k1v
    k2v = k2 * (1 + interest)
    k3 = premium + k2v
    k3v = k3 * (1 + interest)
    k4 = premium + k3v
    k4v = k4 * (1 + interest)
    k5 = premium + k4v
    k5v = k5 * (1 + interest)

    correct[:, 0, 0] = premium
    correct[:, 0, 1] = 0.0
    correct[:, 1, 0] = premium
    correct[:, 1, 1] = k1
    correct[:, 2, 0] = premium
    correct[:, 2, 1] = k1v
    correct[:, 3, 0] = premium
    correct[:, 3, 1] = k2
    correct[:, 4, 0] = premium
    correct[:, 4, 1] = k2v
    correct[:, 5, 0] = premium
    correct[:, 5, 1] = k3
    correct[:, 6, 0] = premium
    correct[:, 6, 1] = k3v
    correct[:, 7, 0] = premium
    correct[:, 7, 1] = k4
    correct[:, 8, 0] = premium
    correct[:, 8, 1] = k4v
    correct[:, 9, 0] = premium
    correct[:, 9, 1] = k5
    correct[:, 10, 0] = premium
    correct[:, 10, 1] = k5v

    np_loss = (np.fabs(gData[:, 0, 0] - premium) + np.fabs(gData[:, 0, 1]) +
               np.fabs(gData[:, 1, 0] - premium) + np.fabs(gData[:, 1, 1] - k1) +
               np.fabs(gData[:, 2, 0] - premium) + np.fabs(gData[:, 2, 1] - k1v) +
               np.fabs(gData[:, 3, 0] - premium) + np.fabs(gData[:, 3, 1] - k2) +
               np.fabs(gData[:, 4, 0] - premium) + np.fabs(gData[:, 4, 1] - k2v) +
               np.fabs(gData[:, 5, 0] - premium) + np.fabs(gData[:, 5, 1] - k3) +
               np.fabs(gData[:, 6, 0] - premium) + np.fabs(gData[:, 6, 1] - k3v) +
               np.fabs(gData[:, 7, 0] - premium) + np.fabs(gData[:, 7, 1] - k4) +
               np.fabs(gData[:, 8, 0] - premium) + np.fabs(gData[:, 8, 1] - k4v) +
               np.fabs(gData[:, 9, 0] - premium) + np.fabs(gData[:, 9, 1] - k5) +
               np.fabs(gData[:, 10, 0] - premium) + np.fabs(gData[:, 10, 1] - k5v)
               )

    return np_loss
