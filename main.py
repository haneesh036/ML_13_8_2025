import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from dt_utils import (
    entropy, gini, equal_width_binning, equal_freq_binning,
    select_root_feature, id3_train, id3_predict, print_tree
)


# Load features (pick spectral or cepstral)
spectral = pd.read_csv("20231225_dfall_obs_data_and_spectral_features_revision1_n469.csv")
# Or cepstral:
# cepstral = pd.read_csv("20240106_dfall_obs_data_and_cepstral_features_revision1_n469.csv")

# Keep only rows with certain receiver labels
df = spectral[spectral["Cert_ID_Elic1_Bin"] == 1].copy()

# Target
y = df["Elicitor1_ID"].astype("category")

# Choose features: start with a few components for clarity
feature_cols = [c for c in df.columns if c.startswith("V")][:6]
# Optionally include sparse metrics or F/M derived metrics
# feature_cols += ["sprsMed", "sprsMbw", "sprsEqbw", "sprsMc"]
X = df[feature_cols].copy()

# Optionally limit to most frequent receivers to avoid extreme class imbalance
topK = 5
top_receivers = y.value_counts().index[:topK]
mask = y.isin(top_receivers)
X = X.loc[mask].reset_index(drop=True)
y = y.loc[mask].reset_index(drop=True)

# A1: Entropy
H_y = entropy(y)
print("A1) Entropy (Elicitor1_ID):", H_y)

# A2: Gini
G_y = gini(y)
print("A2) Gini (Elicitor1_ID):", G_y)

# A3: Root feature using IG (with binning for numeric features)
best_feat, best_ig = select_root_feature(X, y, binning="equal_freq", n_bins=4)
print("A3) Root feature:", best_feat, "IG=", best_ig)

# A5: Train ID3 tree
tree = id3_train(X, y, binning="equal_freq", n_bins=4, max_depth=4, min_samples_split=5)

# A6: Visualize the tree (text)
print("\nA6) Decision tree (text):")
print_tree(tree)


# A7: Decision boundary for two features
feat2d = feature_cols[:2]  # e.g., V1 and V2
tree2d = id3_train(X[feat2d], y, binning="equal_freq", n_bins=4, max_depth=3)

# Grid over raw feature space
x_min, x_max = X[feat2d[0]].min(), X[feat2d[0]].max()
y_min, y_max = X[feat2d[1]].min(), X[feat2d[1]].max()
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))
grid = pd.DataFrame({
    feat2d[0]: xx.ravel(),
    feat2d[1]: yy.ravel()
})

Z = id3_predict(tree2d, grid, y_categories=y.cat.categories)
Z_codes = pd.Series(Z, dtype="category").cat.codes.values.reshape(xx.shape)

plt.figure(figsize=(7,6))
plt.contourf(xx, yy, Z_codes, alpha=0.25, cmap="tab20")
plt.scatter(X[feat2d[0]], X[feat2d[1]], c=y.cat.codes, cmap="tab20", s=12, edgecolor="k", linewidth=0.2)
plt.xlabel(feat2d[0])
plt.ylabel(feat2d[1])
plt.title("A7) Decision boundary (ID3, two features)")
plt.tight_layout()
plt.show()

