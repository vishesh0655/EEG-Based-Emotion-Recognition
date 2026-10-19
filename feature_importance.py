import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

CSV_PATH = "DREAMER_features.csv"
OUTPUT_DIR = "dreamer_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHANNELS_L = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1"]
CHANNELS_R = ["AF4", "F8", "F4", "FC6", "T8", "P8", "O2"]
BANDS = ["theta", "alpha", "beta", "gamma"]

META_COLS = ["subject", "trial", "window", "valence", "arousal", "valence_label", "arousal_label"]

df = pd.read_csv(CSV_PATH)
print("Dataset Shape:", df.shape)

base_feat_cols = [c for c in df.columns if c not in META_COLS]
print("Base Features:", len(base_feat_cols))

new_features = {}

# Log Features
for band in BANDS:
    for ch in CHANNELS_L + CHANNELS_R:
        new_features[f"log_{band}_{ch}"] = np.log1p(df[f"{band}_{ch}"])

# Hemisphere Asymmetry
for band in BANDS:
    for l, r in zip(CHANNELS_L, CHANNELS_R):
        new_features[f"asym_{band}_{l}_{r}"] = np.log1p(df[f"{band}_{r}"]) - np.log1p(df[f"{band}_{l}"])

# Band Ratios
for ch in CHANNELS_L + CHANNELS_R:
    new_features[f"theta_alpha_{ch}"] = np.log1p(df[f"theta_{ch}"]) - np.log1p(df[f"alpha_{ch}"])
    new_features[f"beta_alpha_{ch}"] = np.log1p(df[f"beta_{ch}"]) - np.log1p(df[f"alpha_{ch}"])
    new_features[f"gamma_beta_{ch}"] = np.log1p(df[f"gamma_{ch}"]) - np.log1p(df[f"beta_{ch}"])

# FAA Features (Frontal Alpha Symmetry)
new_features["FAA"] = np.log1p(df["alpha_AF4"]) - np.log1p(df["alpha_AF3"])
new_features["FAA_F4_F3"] = np.log1p(df["alpha_F4"]) - np.log1p(df["alpha_F3"])
new_features["FAA_FC6_FC5"] = np.log1p(df["alpha_FC6"]) - np.log1p(df["alpha_FC5"])

# Band Statistics
for band in BANDS:
    all_ch = [f"{band}_{ch}" for ch in CHANNELS_L + CHANNELS_R]
    left_ch = [f"{band}_{ch}" for ch in CHANNELS_L]
    right_ch = [f"{band}_{ch}" for ch in CHANNELS_R]

    new_features[f"total_{band}"] = np.log1p(df[all_ch].sum(axis=1))
    new_features[f"mean_{band}"] = np.log1p(df[all_ch].mean(axis=1))
    new_features[f"total_{band}_L"] = np.log1p(df[left_ch].sum(axis=1))
    new_features[f"total_{band}_R"] = np.log1p(df[right_ch].sum(axis=1))
    new_features[f"LR_asym_{band}"] = new_features[f"total_{band}_R"] - new_features[f"total_{band}_L"]

# Subject Normalization
for col in base_feat_cols:
    means = df.groupby("subject")[col].transform("mean")
    stds = df.groupby("subject")[col].transform("std").replace(0, 1)
    new_features[f"snorm_{col}"] = (df[col] - means) / stds

# Trial Normalization
for col in base_feat_cols:
    means = df.groupby(["subject", "trial"])[col].transform("mean")
    stds = df.groupby(["subject", "trial"])[col].transform("std").replace(0, 1)
    new_features[f"tnorm_{col}"] = (df[col] - means) / stds

print("Before", len(base_feat_cols))
print("New Features Created:", len(new_features))

df_fe = pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)

feat_cols = [c for c in df_fe.columns if c not in META_COLS]
print("Total Features:", len(feat_cols))

X = df_fe[feat_cols].values
X = np.nan_to_num(X)

y_valence = (df_fe["valence_label"] == "High").astype(int)
y_arousal = (df_fe["arousal_label"] == "High").astype(int)

#XGBoost Feature Importance Valence

X_train, _, y_train, _ = train_test_split(
    X, y_valence, test_size=0.20, random_state=42, stratify=y_valence
)

xgb_valence = XGBClassifier(
    n_estimators=300, max_depth=7, learning_rate=0.08,
    subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42
)
xgb_valence.fit(X_train, y_train)

imp_valence = pd.DataFrame({
    "feature": feat_cols,
    "importance": xgb_valence.feature_importances_
}).sort_values("importance", ascending=False).reset_index(drop=True)

print("Top 20 Features - Valence:")
print(imp_valence.head(20))

plt.figure(figsize=(8, 8))
sns.barplot(data=imp_valence.head(20), x="importance", y="feature", color="steelblue")
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("XGBoost Feature Importance - Valence")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "xgboost_valence_feature_importance.png"))
plt.show()

#Feature Importance Arousal

X_train, _, y_train, _ = train_test_split(
    X, y_arousal, test_size=0.20, random_state=42, stratify=y_arousal
)

xgb_arousal = XGBClassifier(
    n_estimators=300, max_depth=7, learning_rate=0.08,
    subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42
)
xgb_arousal.fit(X_train, y_train)

imp_arousal = pd.DataFrame({
    "feature": feat_cols,
    "importance": xgb_arousal.feature_importances_
}).sort_values("importance", ascending=False).reset_index(drop=True)

print("Top 20 Features - Arousal:")
print(imp_arousal.head(20))

plt.figure(figsize=(8, 8))
sns.barplot(data=imp_arousal.head(20), x="importance", y="feature", color="steelblue")
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("XGBoost Feature Importance - Arousal")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "xgboost_arousal_feature_importance.png"))
plt.show()