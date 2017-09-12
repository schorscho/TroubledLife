# Common imports
import os
import random

import numpy as np
import pandas as pd

from src.policy import Policy

PROJECT_ROOT_DIR = "/Users/gopora/MyStuff/Dev/Workspaces/Sandbox/TroubledLife"
DATASETS_DIR = os.path.join(PROJECT_ROOT_DIR, "data")
TRAINING_SET_DATA = "troubled_life_policy_data.csv"


def generate_very_simple_portfolio_history(no_of_policies, runtime):
    outfile_name = os.path.join(DATASETS_DIR, TRAINING_SET_DATA)

    if not os.path.isdir(DATASETS_DIR):
        os.makedirs(DATASETS_DIR)

    with open(outfile_name, 'w') as csvfile:
        policies = [Policy(id, 100000) for id in range(1001, 1001 + no_of_policies)]

        Policy.write_header_to_csv(csvfile)

        for policy in policies:
            policy.write_to_csv(csvfile)

        for year in range(runtime):
            for policy in policies:
                policy.collect_premium(year)
                policy.write_to_csv(csvfile)

            for policy in policies:
                policy.add_yearly_interest()
                policy.write_to_csv(csvfile)

def generate_portfolio_history(no_of_policies, runtime):
    random.seed(42)

    outfile_name = os.path.join(DATASETS_DIR, TRAINING_SET_DATA)

    if not os.path.isdir(DATASETS_DIR):
        os.makedirs(DATASETS_DIR)

    with open(outfile_name, 'w') as csvfile:
        policies = [Policy(id, 100000) for id in range(1001, 1001 + no_of_policies)]

        Policy.write_header_to_csv(csvfile)

        for policy in policies:
            policy.write_to_csv(csvfile)

        for year in range(runtime):
            for policy in policies:
                policy.collect_premium(year)
                policy.write_to_csv(csvfile)

            for policy in random.sample(policies, no_of_policies // 10):
                policy.adapt_premium(random.uniform(-0.2, 0.2))
                policy.write_to_csv(csvfile)

            for policy in policies:
                policy.add_yearly_interest()
                policy.write_to_csv(csvfile)

def load_troubled_life_policy_data():
    return pd.read_csv(filepath_or_buffer=os.path.join(DATASETS_DIR, TRAINING_SET_DATA), header=0, index_col=[0, 1, 2])

def prepare_troubled_life_policy_data(policy_histories):
    policy_histories["troubled"].replace(to_replace=[False, True], value=[0, 1], inplace=True)

    # Collect the lenght of each policy's history
    history_lengths = np.array([(id, policy_histories.loc[id].shape[0]) for id in policy_histories.index.levels[0]])
    max_history_length = history_lengths.max(axis=0)[1]

    # Let's zero-padd all policy histories to the maximum number of versions
    for id, history_length in history_lengths:
        # Append at the end, set id to current group and current_year to 10000+
        # such that sorting will definitely move all padded rows to the end of the group
        for i in range(max_history_length - history_length):
            policy_histories.loc[id, 10000 + i, 0] = [0, 0, 0]

    return policy_histories.sort_index(), history_lengths, max_history_length