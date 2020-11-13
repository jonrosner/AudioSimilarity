from collections import defaultdict
import os

def calculate_splits(iden_splits_path, n=-1, k=-1):
    data_splits = {
        "train": collections.defaultdict(list),
        "validation": collections.defaultdict(list),
        "test": collections.defaultdict(list)
    }
    with open(iden_splits_path, "r") as f:
        for line in f:
            group,label,filename = line.rstrip().split(",")
            label = int(label)
            split_name = {0: "train", 1: "test"}[int(group)]
            # n-way
            if n == -1 or label < n:
                # k-shot
                if k == -1 or split_name != "train" or len(data_splits["train"][label]) < k:
                    data_splits[split_name][label].append(filename.strip())
    final_data_splits = {
        "train": [],
        "test": []
    }
    for split, split_values in data_splits.items():
        for label, paths in split_values.items():
            final_data_splits[split] += paths
    return final_data_splits
