```python
"""
Week 5 - Predictive Modeling and Algorithm Selection (Beauty & Wellness, simulated data)
Author: Chetan Kumar Gauttam
Run:  python predictive_modeling.py      (clean_salon_visits.csv must be in the same folder)
Needs: pandas numpy matplotlib seaborn scikit-learn
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, mean_squared_error, r2_score,
                             silhouette_score)

df = pd.read_csv("clean_salon_visits.csv")

# ---------------- features (from Week 4) ----------------
df["log_spend"] = np.log1p(df["avg_spend"])
df["long_wait"] = (df["wait_time_min"] > 25).astype(int)
df["festive_season"] = df["visit_month"].isin([10, 11, 12, 2]).astype(int)
df["loyalty_score"] = df["visits_last_year"] * df["satisfaction"] + 3 * df["membership"]

# =============== PART A: classification - will the customer return in 90 days? ===============
feat = ["age", "membership", "discount_used", "wait_time_min", "visits_last_year",
        "satisfaction", "long_wait", "festive_season", "loyalty_score", "log_spend"]
X = pd.get_dummies(df[feat + ["service"]], columns=["service"], drop_first=True)
y = df["returned_90d"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=1, stratify=y)
cv = StratifiedKFold(5, shuffle=True, random_state=1)

models = {
    "Baseline (majority)": DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Decision Tree (depth 4)": DecisionTreeClassifier(max_depth=4, random_state=1),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=1),
}
rows = []
for name, m in models.items():
    cv_auc = cross_val_score(m, Xtr, ytr, cv=cv, scoring="roc_auc")
    m.fit(Xtr, ytr)
    pred = m.predict(Xte)
    proba = m.predict_proba(Xte)[:, 1]
    rows.append({"model": name,
                 "cv_auc_mean": cv_auc.mean(), "cv_auc_std": cv_auc.std(),
                 "accuracy": accuracy_score(yte, pred),
                 "precision": precision_score(yte, pred, zero_division=0),
                 "recall": recall_score(yte, pred),
                 "f1": f1_score(yte, pred),
                 "test_auc": roc_auc_score(yte, proba)})
res = pd.DataFrame(rows).round(3)
print(res.to_string(index=False))
res.to_csv("model_results.csv", index=False)

# confusion matrix of the logistic regression
lr = models["Logistic Regression"]
cm = confusion_matrix(yte, lr.predict(Xte))
print("Confusion matrix (logistic regression):\n", cm)

# ---------------- overfitting check: tree depth vs accuracy ----------------
depths = range(1, 13)
tr_acc, cv_acc = [], []
for d in depths:
    t = DecisionTreeClassifier(max_depth=d, random_state=1)
    t.fit(Xtr, ytr)
    tr_acc.append(t.score(Xtr, ytr))
    cv_acc.append(cross_val_score(t, Xtr, ytr, cv=cv).mean())
print("Best depth by CV:", list(depths)[int(np.argmax(cv_acc))],
      "| train acc at depth 12:", round(tr_acc[-1], 3), "cv acc at depth 12:", round(cv_acc[-1], 3))

# ---------------- hyperparameter tuning ----------------
gs = GridSearchCV(RandomForestClassifier(random_state=1),
                  {"max_depth": [3, 5, 8], "n_estimators": [100, 200], "min_samples_leaf": [1, 5, 10]},
                  cv=cv, scoring="roc_auc").fit(Xtr, ytr)
print("Best random forest params:", gs.best_params_, "CV AUC:", round(gs.best_score_, 3),
      "Test AUC:", round(roc_auc_score(yte, gs.predict_proba(Xte)[:, 1]), 3))

# feature importance
rf = models["Random Forest"]
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Top features:\n", imp.head(5).round(3))
coef = pd.Series(lr[-1].coef_[0], index=X.columns).sort_values(key=abs, ascending=False)
print("Logistic coefficients (scaled):\n", coef.head(5).round(2))


# ---------------- class imbalance: can we catch customers who will NOT return? ----------------
for cw in [None, "balanced"]:
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight=cw)).fit(Xtr, ytr)
    p = m.predict(Xte)
    print(f"class_weight={cw}: accuracy={accuracy_score(yte, p):.3f}, "
          f"recall for non-returners={recall_score(yte, p, pos_label=0):.3f}, "
          f"precision for non-returners={precision_score(yte, p, pos_label=0):.3f}")

# =============== PART B: regression - predict average spend ===============
Xr = pd.get_dummies(df[["age", "membership", "visits_last_year", "satisfaction", "service"]],
                    columns=["service"], drop_first=True)
yr = df["avg_spend"]
Xrtr, Xrte, yrtr, yrte = train_test_split(Xr, yr, test_size=0.25, random_state=1)
lin = LinearRegression().fit(Xrtr, yrtr)
rmse = mean_squared_error(yrte, lin.predict(Xrte)) ** 0.5
print(f"Linear regression: RMSE={rmse:.0f}, R2={r2_score(yrte, lin.predict(Xrte)):.3f}, "
      f"baseline RMSE (predict mean)={yrte.std():.0f}")

# =============== PART C: clustering - customer segments ===============
Z = StandardScaler().fit_transform(df[["avg_spend", "visits_last_year", "satisfaction", "wait_time_min"]])
for k in range(2, 7):
    km = KMeans(n_clusters=k, n_init=10, random_state=1).fit(Z)
    print(f"k={k}: silhouette={silhouette_score(Z, km.labels_):.3f}")
km = KMeans(n_clusters=3, n_init=10, random_state=1).fit(Z)
df["segment"] = km.labels_
print(df.groupby("segment")[["avg_spend", "visits_last_year", "satisfaction", "wait_time_min",
                              "returned_90d"]].mean().round(2).assign(n=df.groupby("segment").size()))

# =============== figures ===============
fig, ax = plt.subplots(figsize=(7, 3.8))
sub = res[res.model != "Baseline (majority)"]
xs = np.arange(len(sub)); w = 0.2
for i, col in enumerate(["accuracy", "precision", "recall", "test_auc"]):
    ax.bar(xs + i * w, sub[col], w, label=col)
ax.set_xticks(xs + 1.5 * w); ax.set_xticklabels(sub.model, fontsize=8); ax.set_ylim(0, 1)
ax.axhline(res.loc[0, "accuracy"], color="grey", ls="--", lw=1); ax.legend(fontsize=8, ncol=4)
ax.set_title("Model comparison on the test set (dashed line = baseline accuracy)", fontsize=10)
plt.tight_layout(); plt.savefig("fig5_model_comparison.png", dpi=130); plt.close()

plt.figure(figsize=(6, 3.8))
plt.plot(list(depths), tr_acc, marker="o", label="Training accuracy")
plt.plot(list(depths), cv_acc, marker="o", label="Cross-validation accuracy")
plt.xlabel("Decision tree depth"); plt.ylabel("Accuracy"); plt.legend()
plt.title("Overfitting check: tree depth"); plt.tight_layout()
plt.savefig("fig6_overfitting.png", dpi=130); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
imp.head(8)[::-1].plot.barh(ax=ax[0]); ax[0].set_title("Random forest: top features", fontsize=10)
ax[1].imshow(cm, cmap="Blues")
for (i, j), v in np.ndenumerate(cm): ax[1].text(j, i, v, ha="center", va="center", fontsize=14)
ax[1].set_xticks([0, 1]); ax[1].set_yticks([0, 1])
ax[1].set_xticklabels(["Pred: no", "Pred: yes"]); ax[1].set_yticklabels(["Actual: no", "Actual: yes"])
ax[1].set_title("Confusion matrix (logistic regression)", fontsize=10)
plt.tight_layout(); plt.savefig("fig7_importance_confusion.png", dpi=130); plt.close()
```