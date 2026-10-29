```python
"""
Week 6 - Model Evaluation, Reporting and Business Impact (Beauty & Wellness, simulated data)
Author: Chetan Kumar Gauttam
Run:  python model_evaluation.py     (clean_salon_visits.csv must be in the same folder)
Needs: pandas numpy matplotlib scikit-learn
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import (StratifiedKFold, RepeatedStratifiedKFold, cross_val_predict,
                                     cross_validate, train_test_split)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix, precision_score,
                             recall_score, f1_score, accuracy_score)

df = pd.read_csv("clean_salon_visits.csv")
df["log_spend"] = np.log1p(df["avg_spend"])
df["long_wait"] = (df["wait_time_min"] > 25).astype(int)
df["festive_season"] = df["visit_month"].isin([10, 11, 12, 2]).astype(int)
df["loyalty_score"] = df["visits_last_year"] * df["satisfaction"] + 3 * df["membership"]
feat = ["age", "membership", "discount_used", "wait_time_min", "visits_last_year", "satisfaction",
        "long_wait", "festive_season", "loyalty_score", "log_spend"]
X = pd.get_dummies(df[feat + ["service"]], columns=["service"], drop_first=True)
y = df["returned_90d"]
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

# ---- 1. repeated cross-validation (5 folds x 5 repeats) vs baseline ----
rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=7)
scoring = {"accuracy": "accuracy", "precision": "precision", "recall": "recall", "f1": "f1", "auc": "roc_auc"}
rows = []
for name, m in {"Baseline (always 'returns')": DummyClassifier(strategy="most_frequent"), "Logistic Regression": model}.items():
    r = cross_validate(m, X, y, cv=rcv, scoring=scoring)
    rows.append({"model": name, **{k: (r["test_" + k].mean(), r["test_" + k].std()) for k in scoring}})
cvres = pd.DataFrame(rows)
for _, r in cvres.iterrows():
    print(r["model"], {k: f"{r[k][0]:.3f}+-{r[k][1]:.3f}" for k in scoring})

# ---- 2. out-of-fold risk scores for every customer (risk = chance of NOT returning) ----
oof = cross_val_predict(model, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=7), method="predict_proba")[:, 1]
df["p_return"] = oof
df["risk"] = 1 - oof
leaver = (y == 0).astype(int)
print("AUC (out-of-fold):", round(roc_auc_score(y, oof), 3), " base leaver rate:", round(leaver.mean(), 3))

# ---- 3. gain / lift: contact the riskiest k% ----
order = np.argsort(-df["risk"].values)
n = len(df); rows = []
for pct in [10, 20, 30, 40, 50, 60, 80, 100]:
    k = int(n * pct / 100); idx = order[:k]
    caught = leaver.values[idx].sum()
    rows.append({"contacted_pct": pct, "contacted": k, "leavers_reached": int(caught),
                 "share_of_all_leavers": caught / leaver.sum(), "precision": caught / k,
                 "lift": (caught / k) / leaver.mean()})
gain = pd.DataFrame(rows); print(gain.round(3).to_string(index=False))
gain.to_csv("gain_table.csv", index=False)

# ---- 4. confusion matrix when the riskiest 30% are flagged ----
k30 = int(n * .30); flag = np.zeros(n, int); flag[order[:k30]] = 1
cm = confusion_matrix(leaver, flag); print("Flag riskiest 30%: [[TN FP],[FN TP]] =", cm.tolist())
print("precision", round(precision_score(leaver, flag), 3), "recall", round(recall_score(leaver, flag), 3),
      "f1", round(f1_score(leaver, flag), 3), "accuracy", round(accuracy_score(leaver, flag), 3))

# ---- 5. performance by service (consistency check) ----
by = []
for s, g in df.groupby("service"):
    by.append({"service": s, "customers": len(g), "auc": roc_auc_score(g["returned_90d"], g["p_return"])})
byservice = pd.DataFrame(by).round(3); print(byservice.to_string(index=False))
byservice.to_csv("auc_by_service.csv", index=False)

# ---- 6. hypothetical cost-benefit (ASSUMPTIONS, not real data) ----
def net_benefit(pct, offer_cost, win_back=0.25):
    k = int(n * pct / 100); idx = order[:k]
    gain_rs = (win_back * df["avg_spend"].values[idx] * leaver.values[idx]).sum()
    return gain_rs - offer_cost * k
nb = pd.DataFrame({"contact_top_pct": [10, 20, 30, 40, 60, 100]})
for oc in [50, 100, 150]:
    nb[f"offer Rs.{oc}"] = [round(net_benefit(p, oc)) for p in nb["contact_top_pct"]]
print(nb.to_string(index=False)); nb.to_csv("cost_benefit.csv", index=False)

# ---- 7. sample 'dashboard' rows ----
def level(r): return "High" if r >= .6 else ("Medium" if r >= .4 else "Low")
df["risk_level"] = df["risk"].apply(level)
print(df["risk_level"].value_counts().to_dict())
dash = df.sort_values("risk", ascending=False).head(5)[["customer_id", "service", "satisfaction", "wait_time_min", "visits_last_year", "risk"]]
print(dash.round(2).to_string(index=False))
print(df.groupby("risk_level")["returned_90d"].agg(["count", lambda s: 1 - s.mean()]).round(3))

# ---- figures ----
fpr, tpr, _ = roc_curve(y, oof)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].plot(fpr, tpr, label=f"Model (AUC = {roc_auc_score(y, oof):.2f})"); ax[0].plot([0, 1], [0, 1], "--", c="grey", label="Guessing (AUC = 0.50)")
ax[0].set_xlabel("False alarms (share of returners wrongly flagged)"); ax[0].set_ylabel("Share of leavers caught")
ax[0].set_title("ROC curve", fontsize=10); ax[0].legend(fontsize=8)
ax[1].plot([0] + list(gain.contacted_pct), [0] + list(gain.share_of_all_leavers * 100), marker="o", label="Using the model")
ax[1].plot([0, 100], [0, 100], "--", c="grey", label="Contacting at random")
ax[1].set_xlabel("% of customers contacted (riskiest first)"); ax[1].set_ylabel("% of leavers reached")
ax[1].set_title("Gain chart", fontsize=10); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig("fig8_roc_gain.png", dpi=130); plt.close()

fig, ax = plt.subplots(figsize=(4.6, 3.6))
ax.imshow(cm, cmap="Blues")
for (i, j), v in np.ndenumerate(cm): ax.text(j, i, v, ha="center", va="center", fontsize=15, color="white" if v > cm.max() / 2 else "black")
ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_xticklabels(["Not flagged", "Flagged at-risk"], fontsize=8)
ax.set_yticklabels(["Returned", "Did not return"], fontsize=8); ax.set_title("Flagging the riskiest 30%", fontsize=10)
plt.tight_layout(); plt.savefig("fig9_confusion.png", dpi=130); plt.close()

plt.figure(figsize=(6.2, 3.6))
for oc in [50, 100, 150]: plt.plot(nb.contact_top_pct, nb[f"offer Rs.{oc}"], marker="o", label=f"Offer costs Rs. {oc}")
plt.axhline(0, c="grey", lw=1); plt.xlabel("% of customers contacted (riskiest first)"); plt.ylabel("Net benefit (Rs.)")
plt.title("Hypothetical net benefit of a win-back offer", fontsize=10); plt.legend(fontsize=8); plt.tight_layout()
plt.savefig("fig10_cost_benefit.png", dpi=130); plt.close()
df[["customer_id", "service", "risk", "risk_level"]].round(3).to_csv("customer_risk_scores.csv", index=False)
```