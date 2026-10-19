import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split
)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


CSV_PATH = "DREAMER_features.csv"

OUTPUT_DIR = "dreamer_outputs"

N_ESTIMATORS = 300
CV_FOLDS = 3
TEST_SIZE = 0.20
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
CHANNELS_L = ["AF3","F7","F3","FC5","T7","P7","O1"]
CHANNELS_R = ["AF4","F8","F4","FC6","T8","P8","O2"]

BANDS = ["theta","alpha","beta","gamma"]

META_COLS = [
    "subject",
    "trial",
    "window",
    "valence",
    "arousal",
    "valence_label",
    "arousal_label"
]
df = pd.read_csv(CSV_PATH)

print("Dataset Shape:", df.shape)


base_feat_cols = [
    c for c in df.columns
    if c not in META_COLS
]

print("Base Features:", len(base_feat_cols))

df.head()


new_features = {}


# Log Features

for band in BANDS:
    for ch in CHANNELS_L + CHANNELS_R:
        new_features[f"log_{band}_{ch}"] = np.log1p(
            df[f"{band}_{ch}"]
        )


# Hemisphere Asymmetry

for band in BANDS:
    for l, r in zip(CHANNELS_L, CHANNELS_R):

        new_features[f"asym_{band}_{l}_{r}"] = (
            np.log1p(df[f"{band}_{r}"])
            -
            np.log1p(df[f"{band}_{l}"])
        )


# Band Ratios

for ch in CHANNELS_L + CHANNELS_R:

    new_features[f"theta_alpha_{ch}"] = (
        np.log1p(df[f"theta_{ch}"])
        -
        np.log1p(df[f"alpha_{ch}"])
    )

    new_features[f"beta_alpha_{ch}"] = (
        np.log1p(df[f"beta_{ch}"])
        -
        np.log1p(df[f"alpha_{ch}"])
    )

    new_features[f"gamma_beta_{ch}"] = (
        np.log1p(df[f"gamma_{ch}"])
        -
        np.log1p(df[f"beta_{ch}"])
    )


# FAA Features(Frontal Alpha Symmetry)

new_features["FAA"] = (
    np.log1p(df["alpha_AF4"])
    -
    np.log1p(df["alpha_AF3"])
)

new_features["FAA_F4_F3"] = (
    np.log1p(df["alpha_F4"])
    -
    np.log1p(df["alpha_F3"])
)

new_features["FAA_FC6_FC5"] = (
    np.log1p(df["alpha_FC6"])
    -
    np.log1p(df["alpha_FC5"])
)

# Band Statistics

for band in BANDS:

    all_ch = [f"{band}_{ch}" for ch in CHANNELS_L + CHANNELS_R]
    left_ch = [f"{band}_{ch}" for ch in CHANNELS_L]
    right_ch = [f"{band}_{ch}" for ch in CHANNELS_R]

    new_features[f"total_{band}"] = np.log1p(
        df[all_ch].sum(axis=1)
    )

    new_features[f"mean_{band}"] = np.log1p(
        df[all_ch].mean(axis=1)
    )

    new_features[f"total_{band}_L"] = np.log1p(
        df[left_ch].sum(axis=1)
    )

    new_features[f"total_{band}_R"] = np.log1p(
        df[right_ch].sum(axis=1)
    )

    new_features[f"LR_asym_{band}"] = (
        new_features[f"total_{band}_R"]
        -
        new_features[f"total_{band}_L"]
    )


# Subject Normalization (df[col] - mean) / std

for col in base_feat_cols:

    means = (
        df.groupby("subject")[col]
        .transform("mean")
    )

    stds = (
        df.groupby("subject")[col]
        .transform("std")
        .replace(0, 1)
    )

    new_features[f"snorm_{col}"] = (
        (df[col] - means) / stds
    )


# Trial Normalization (df[col] - mean) / std

for col in base_feat_cols:

    means = (
        df.groupby(
            ["subject", "trial"]
        )[col].transform("mean")
    )

    stds = (
        df.groupby(
            ["subject", "trial"]
        )[col].transform("std")
        .replace(0, 1)
    )

    new_features[f"tnorm_{col}"] = (
        (df[col] - means) / stds
    )
print("Before", len(base_feat_cols))
print("New Features Created:", len(new_features))

df_fe = pd.concat(
    [
        df,
        pd.DataFrame(
            new_features,
            index=df.index
        )
    ],
    axis=1
)

feat_cols = [
    c for c in df_fe.columns
    if c not in META_COLS
]

print("Total Features:", len(feat_cols))
X = df_fe[feat_cols].values

X = np.nan_to_num(X)

y_valence = (
    df_fe["valence_label"] == "High"
).astype(int)

y_arousal = (
    df_fe["arousal_label"] == "High"
).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_arousal,
    test_size=0.20,
    random_state=42,
    stratify=y_arousal
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

print(X_train.shape)

cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

print("Arousal Classification using LightGBM, Random Forest, XGBoost, SVM and Logistic Regression")

from lightgbm import LGBMClassifier

lgbm = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.08,
    max_depth=7,
    random_state=42,
    verbose=-1,
    n_jobs=-1
)

lgbm_cv = cross_val_score(
    lgbm,
    X,
    y_arousal,
    cv=cv,
    scoring="accuracy"
)

lgbm.fit(X_train, y_train)

lgbm_pred = lgbm.predict(X_test)

lgbm_acc = accuracy_score(
    y_test,
    lgbm_pred
)

print("LightGBM CV Accuracy:",
      lgbm_cv.mean()*100)

print("LightGBM Test Accuracy:",
      lgbm_acc*100)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf_cv = cross_val_score(
    rf,
    X,
    y_arousal,
    cv=cv,
    scoring="accuracy"
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

rf_acc = accuracy_score(
    y_test,
    rf_pred
)

print("RF CV Accuracy:",
      rf_cv.mean()*100)

print("RF Test Accuracy:",
      rf_acc*100)

from xgboost import XGBClassifier

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=7,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

xgb_cv = cross_val_score(
    xgb,
    X,
    y_arousal,
    cv=cv,
    scoring="accuracy"
)

xgb.fit(X_train, y_train)

xgb_pred = xgb.predict(X_test)

xgb_acc = accuracy_score(
    y_test,
    xgb_pred
)

print("XGB CV Accuracy:",
      xgb_cv.mean()*100)

print("XGB Test Accuracy:",
      xgb_acc*100)

from sklearn.svm import SVC

svm = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale"
)

svm_cv = cross_val_score(
    svm,
    X,
    y_arousal,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1

)

svm.fit(X_train, y_train)

svm_pred = svm.predict(X_test)

svm_acc = accuracy_score(
    y_test,
    svm_pred
)

print("SVM CV Accuracy:",
      svm_cv.mean()*100)

print("SVM Test Accuracy:",
      svm_acc*100)


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(
    max_iter=2000,
    random_state=42,
    n_jobs=-1
)

lr_cv = cross_val_score(
    lr,
    X,
    y_arousal,
    cv=cv,
    scoring="accuracy"
)

lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

lr_acc = accuracy_score(
    y_test,
    lr_pred
)

print("LR CV Accuracy:",
      lr_cv.mean()*100)

print("LR Test Accuracy:",
      lr_acc*100)


#confusion matrix & deatiled report for arousal


# deatiled report of lightgbm

print(classification_report(y_test, lgbm_pred))

print(confusion_matrix(y_test, lgbm_pred))

cm = confusion_matrix(
    y_test,
    lgbm_pred
)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("LightGBM Arousal Confusion Matrix")
plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "lightgbm_arousal_confusion_matrix.png"
    )
)

plt.show()

# deatiled report of random forest 

print(classification_report(y_test, rf_pred))

print(confusion_matrix(y_test, rf_pred))

cm = confusion_matrix(
    y_test,
    rf_pred
)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest Arousal Confusion Matrix")
plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "random_forest_arousal_confusion_matrix.png"
    )
)

plt.show()

# deatiled report of xgboost

print(classification_report(y_test, xgb_pred))

print(confusion_matrix(y_test, xgb_pred))

cm = confusion_matrix(
    y_test,
    xgb_pred
)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("XGBoost Arousal Confusion Matrix")
plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "xgboost_arousal_confusion_matrix.png"
    )
)
plt.show()

# deatiled report of SVM


print(classification_report(y_test, svm_pred))

print(confusion_matrix(y_test, svm_pred))

cm = confusion_matrix(
    y_test,
    svm_pred
)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("SVM Arousal Confusion Matrix")
plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "svm_arousal_confusion_matrix.png"
    )
)
plt.show()

# deatiled report of Logistic Regression



print(classification_report(y_test, lr_pred))

print(confusion_matrix(y_test, lr_pred))

cm = confusion_matrix(
    y_test,
    lr_pred
)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Logistic Regression Arousal Confusion Matrix")
plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "logistic_regression_arousal_confusion_matrix.png"
    )
)
plt.show()

