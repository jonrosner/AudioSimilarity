from collections import defaultdict
import os

def calculate_splits(extract_path, iden_splits_path, n=-1, k=-1):
    data_splits = {
        "train": defaultdict(list),
        "validation": defaultdict(list),
        "test": defaultdict(list)
    }
    with open(iden_splits_path, "r") as f:
        for line in f:
            group, path = line.strip().split()
            speaker = path.split("/")[-3]
            label = int(speaker[3:]) - 1 # there is no label 0
            full_path = os.path.join(extract_path, path)
            if not os.path.isfile(full_path):
                continue
            split_name = {1: "train", 2: "train", 3: "test"}[int(group)]
            # n-way
            if n == -1 or label < n:
                # k-shot
                if k == -1 or split_name != "train" or len(data_splits["train"][label]) < k:
                    data_splits[split_name][label].append(full_path.strip())
    final_data_splits = {
        "train": [],
        "validation": [],
        "test": []
    }
    for split, split_values in data_splits.items():
        for label, paths in split_values.items():
            final_data_splits[split] += paths
    return final_data_splits
