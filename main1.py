# dt_utils/utils.py
import numpy as np
import pandas as pd
from collections import Counter

# ---------------------------
# A1: Entropy
# ---------------------------
def entropy(y):
    y = pd.Series(y).astype("category")
    probs = y.value_counts(normalize=True)
    return -np.sum(probs * np.log2(probs))

# ---------------------------
# A2: Gini index
# ---------------------------
def gini(y):
    y = pd.Series(y).astype("category")
    probs = y.value_counts(normalize=True)
    return 1 - np.sum(probs ** 2)

# ---------------------------
# Equal-width binning
# ---------------------------
def equal_width_binning(series, n_bins=4):
    bins = pd.cut(series, bins=n_bins, duplicates='drop')
    return bins.astype("category")

# ---------------------------
# Equal-frequency binning
# ---------------------------
def equal_freq_binning(series, n_bins=4):
    bins = pd.qcut(series, q=n_bins, duplicates='drop')
    return bins.astype("category")

# ---------------------------
# Information Gain
# ---------------------------
def information_gain(y, y_subsets):
    H_before = entropy(y)
    total = len(y)
    H_after = 0
    for subset in y_subsets:
        H_after += (len(subset) / total) * entropy(subset)
    return H_before - H_after

# ---------------------------
# Select root feature
# ---------------------------
def select_root_feature(X, y, binning="equal_width", n_bins=4):
    best_feature = None
    best_ig = -1
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            if binning == "equal_freq":
                binned = equal_freq_binning(X[col], n_bins)
            else:
                binned = equal_width_binning(X[col], n_bins)
        else:
            binned = X[col].astype("category")

        subsets = [y[binned == val] for val in binned.cat.categories]
        ig = information_gain(y, subsets)
        if ig > best_ig:
            best_ig = ig
            best_feature = col
    return best_feature, best_ig

# ---------------------------
# ID3 Algorithm
# ---------------------------
def id3_train(X, y, binning="equal_width", n_bins=4, max_depth=None, min_samples_split=2, depth=0):
    y = y.astype("category")

    # If all labels are same
    if len(y.unique()) == 1:
        return y.iloc[0]

    # Stopping conditions
    if len(X.columns) == 0 or (max_depth is not None and depth >= max_depth) or len(y) < min_samples_split:
        return y.mode()[0]

    # Select best feature
    best_feature, _ = select_root_feature(X, y, binning=binning, n_bins=n_bins)
    if best_feature is None:
        return y.mode()[0]

    # Bin feature if numeric
    if pd.api.types.is_numeric_dtype(X[best_feature]):
        if binning == "equal_freq":
            feature_values = equal_freq_binning(X[best_feature], n_bins)
        else:
            feature_values = equal_width_binning(X[best_feature], n_bins)
    else:
        feature_values = X[best_feature].astype("category")

    tree = {best_feature: {}}

    for val in feature_values.cat.categories:
        mask = (feature_values == val)
        if mask.sum() == 0:
            tree[best_feature][val] = y.mode()[0]
        else:
            subtree = id3_train(
                X.loc[mask, X.columns != best_feature],
                y[mask],
                binning=binning,
                n_bins=n_bins,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                depth=depth + 1
            )
            tree[best_feature][val] = subtree
    return tree

# ---------------------------
# Prediction with category alignment
# ---------------------------
def id3_predict(tree, X, y_categories=None):
    def majority_label(subtree):
        if not isinstance(subtree, dict):
            return subtree
        labels = []
        for branch in subtree.values():
            labels.append(majority_label(branch))
        return Counter(labels).most_common(1)[0][0]

    def predict_one(tree, row):
        if not isinstance(tree, dict):
            return tree
        feature = next(iter(tree))
        val = row[feature]
        if val in tree[feature]:
            return predict_one(tree[feature][val], row)
        else:
            return majority_label(tree[feature])

    preds = [predict_one(tree, row) for _, row in X.iterrows()]
    if y_categories is not None:
        return pd.Series(preds, dtype=pd.CategoricalDtype(categories=y_categories))
    return pd.Series(preds)

# ---------------------------
# Print tree
# ---------------------------
def print_tree(tree, indent=""):
    if not isinstance(tree, dict):
        print(indent + "→ " + str(tree))
        return
    for feature, branches in tree.items():
        for val, subtree in branches.items():
            print(f"{indent}{feature} = {val}")
            print_tree(subtree, indent + "    ")

