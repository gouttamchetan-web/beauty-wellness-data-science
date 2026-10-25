```python
"""
Week 4 - EDA and Feature Engineering (Beauty & Wellness, simulated data)
Author: Chetan Kumar Gauttam
Run:  python eda_feature_engineering.py
Needs: pandas numpy matplotlib seaborn scipy scikit-learn
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

rng = np.random.default_rng(42)

# ---------------- 1. simulated, already-cleaned data (output of Week 3) ----------------
def make_data(n=600):
    services = ["hair", "skin", "spa", "nails", "wellness"]
    df = pd.DataFrame({
        "customer_id": np.arange(1, n + 1),
        "age": rng.integers(18, 60, n),
        "service": rng.choice(services, n, p=[.30, .25, .15, .15, .15]),
        "visit_month": rng.integers(1, 13, n),
        "membership": rng.choice([0, 1], n, p=[.7, .3]),
        "discount_used": rng.choice([0, 1], n, p=[.6, .4]),
        "wait_time_min": np.clip(rng.gamma(3, 6, n), 0, 90).round(),
        "visits_last_year": rng.poisson(4, n) + 1,
    })
    base = {"hair": 700, "skin": 1100, "spa": 1800, "nails": 600, "wellness": 1400}
    df["avg_spend"] = (df["service"].map(base) * rng.lognormal(0, .35, n)).round()
    sat = 4.3 - 0.025 * df["wait_time_min"] + 0.25 * df["membership"] + rng.normal(0, .6, n)
    df["satisfaction"] = np.clip(sat.round(), 1, 5)
    logit = (-2.2 + 0.6 * df["satisfaction"] + 0.15 * df["visits_last_year"]
             + 0.5 * df["membership"] - 0.01 * df["wait_time_min"])
    df["returned_90d"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df

df = make_data()
df.to_csv("clean_salon_visits.csv", index=False)

# ---------------- 2. basic EDA ----------------
print(df.shape)
print(df.describe().round(2).T[["mean", "std", "min", "50%", "max"]])
print("Return rate overall:", round(df["returned_90d"].mean(), 3))
print("Skewness of avg_spend:", round(stats.skew(df["avg_spend"]), 2))

fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
sns.histplot(df["age"], bins=15, ax=ax[0]); ax[0].set_title("Age")
sns.histplot(df["avg_spend"], bins=25, ax=ax[1]); ax[1].set_title("Average spend (Rs.)")
sns.countplot(x="satisfaction", data=df, ax=ax[2]); ax[2].set_title("Satisfaction rating")
plt.tight_layout(); plt.savefig("fig1_distributions.png", dpi=130); plt.close()

plt.figure(figsize=(7, 4))
sns.boxplot(x="service", y="avg_spend", data=df)
plt.title("Spend by service type"); plt.tight_layout()
plt.savefig("fig2_spend_by_service.png", dpi=130); plt.close()

plt.figure(figsize=(7, 5))
sns.heatmap(df.drop(columns=["customer_id"]).select_dtypes("number").corr(),
            annot=True, fmt=".2f", cmap="coolwarm", annot_kws={"size": 7})
plt.title("Correlation heatmap"); plt.tight_layout()
plt.savefig("fig3_correlation.png", dpi=130); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
sns.barplot(x="satisfaction", y="returned_90d", data=df, ax=ax[0], errorbar=None)
ax[0].set_ylabel("Return rate"); ax[0].set_title("Return rate by satisfaction")
sns.barplot(x="service", y="returned_90d", data=df, ax=ax[1], errorbar=None)
ax[1].set_ylabel("Return rate"); ax[1].set_title("Return rate by service")
plt.tight_layout(); plt.savefig("fig4_return_rate.png", dpi=130); plt.close()

# statistical tests
r, p = stats.pearsonr(df["satisfaction"], df["returned_90d"])
print(f"Pearson satisfaction vs return: r={r:.2f}, p={p:.4f}")
ct = pd.crosstab(df["service"], df["returned_90d"])
chi2, p2, _, _ = stats.chi2_contingency(ct)
print(f"Chi-square service vs return: p={p2:.4f}")
g1 = df[df.membership == 1]["avg_spend"]; g0 = df[df.membership == 0]["avg_spend"]
print("t-test spend member vs non-member: p=%.4f" % stats.ttest_ind(g1, g0, equal_var=False).pvalue)

# ---------------- 3. feature engineering ----------------
def add_features(d):
    d = d.copy()
    d["log_spend"] = np.log1p(d["avg_spend"])                         # fix right skew
    d["spend_per_visit"] = d["avg_spend"] / d["visits_last_year"]
    d["age_group"] = pd.cut(d["age"], [17, 25, 35, 45, 60], labels=["18-25", "26-35", "36-45", "46-60"])
    d["long_wait"] = (d["wait_time_min"] > 25).astype(int)
    d["festive_season"] = d["visit_month"].isin([10, 11, 12, 2]).astype(int)   # Diwali, wedding season
    d["happy_member"] = ((d["satisfaction"] >= 4) & (d["membership"] == 1)).astype(int)  # interaction
    d["wait_x_sat"] = d["wait_time_min"] * d["satisfaction"]                          # interaction
    d["loyalty_score"] = d["visits_last_year"] * d["satisfaction"] + 3 * d["membership"]
    return d

fe = add_features(df)
print("New features added:", [c for c in fe.columns if c not in df.columns])

# ---------------- 4. does feature engineering help? (simple check) ----------------
def evaluate(data, cols, cat_cols):
    X = pd.get_dummies(data[cols + cat_cols], columns=cat_cols, drop_first=True)
    y = data["returned_90d"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=1, stratify=y)
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(max_iter=1000).fit(sc.transform(Xtr), ytr)
    pr = m.predict_proba(sc.transform(Xte))[:, 1]
    return accuracy_score(yte, pr > .5), roc_auc_score(yte, pr)

base_cols = ["age", "visit_month", "membership", "discount_used", "wait_time_min",
             "visits_last_year", "avg_spend", "satisfaction"]
a0, u0 = evaluate(fe, base_cols, ["service"])
new_cols = base_cols + ["log_spend", "spend_per_visit", "long_wait", "festive_season",
                        "happy_member", "wait_x_sat", "loyalty_score"]
a1, u1 = evaluate(fe, new_cols, ["service", "age_group"])
print(f"Baseline model  : accuracy={a0:.3f}, AUC={u0:.3f}")
print(f"With new features: accuracy={a1:.3f}, AUC={u1:.3f}")
```