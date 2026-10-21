```python
"""
Data cleaning and preprocessing pipeline - Beauty & Wellness (simulated data)
Author: Chetan Kumar Gauttam
Run:  python cleaning_pipeline.py
"""
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ---------- Step 0: simulated raw data (what messy public data usually looks like) ----------
def make_raw_data():
    return pd.DataFrame({
        "customer_id": [101, 102, 102, 103, 104, 105, 106, 107, 108, 109],
        "product":     ["Aloe Vera Gel", "aloe vera gel ", "aloe vera gel ", "Vitamin C Serum",
                        "vit c serum", "Face Wash", "Hair Oil", "Sunscreen SPF50", "Face Wash", "Hair Oil"],
        "skin_type":   ["Oily", "oily", "oily", "Dry", "dry skin", "Combination", None, "Normal", "OILY", "n/a"],
        "age":         [24, 31, 31, 28, 250, 22, None, 35, 29, 27],
        "price":       ["Rs. 299", "₹299", "₹299", "Rs 850", "850", "199", "₹ 349", "1,200", "199", "-5"],
        "rating":      [4.5, 5, 5, 3.5, None, 4, 2, 4.5, 40, 3],
        "purchase_date": ["12/03/2026", "2026-03-15", "2026-03-15", "20 Mar 2026",
                          "2026/03/22", "25-03-2026", "30-03-2026", "01-04-2026", "03-04-2026", "05/04/2026"],
        "review":      ["Great product!!", "great product!!", "great product!!", "Nice, a bit sticky",
                        None, "Okay", "Smell is too strong", "Works well in summer", "Good", "Not bad"],
    })

# ---------- Step 1: inspect ----------
def inspect(df, label=""):
    print(f"\n--- {label} ---")
    print("Shape:", df.shape)
    print("Missing values per column:\n", df.isna().sum())

# ---------- Step 2: remove duplicates ----------
def drop_duplicates(df):
    # standardise text first, otherwise 'Aloe Vera Gel' and 'aloe vera gel ' look different
    df = df.copy()
    df["product"] = df["product"].str.strip().str.lower()
    return df.drop_duplicates(subset=["customer_id", "product", "purchase_date"]).reset_index(drop=True)

# ---------- Step 3: fix inconsistent categories ----------
PRODUCT_MAP = {"vit c serum": "vitamin c serum"}
SKIN_MAP = {"dry skin": "dry", "n/a": np.nan, "na": np.nan}

def standardise_text(df):
    df = df.copy()
    df["product"] = df["product"].replace(PRODUCT_MAP)
    df["skin_type"] = (df["skin_type"].str.strip().str.lower().replace(SKIN_MAP))
    df["review"] = df["review"].str.strip().str.lower()
    return df

# ---------- Step 4: price -> numeric ----------
def clean_price(df):
    df = df.copy()
    df["price"] = (df["price"].astype(str)
                   .str.replace(r"(?i)rs\.?", "", regex=True)      # remove 'Rs.' / 'Rs' first
                   .str.replace(r"[^\d.\-]", "", regex=True)   # then ₹, commas, spaces
                   .replace("", np.nan).astype(float))
    df.loc[df["price"] <= 0, "price"] = np.nan                 # negative / zero price is invalid
    return df

# ---------- Step 5: dates -> one format ----------
def clean_dates(df):
    df = df.copy()
    df["purchase_date"] = pd.to_datetime(df["purchase_date"], dayfirst=True,
                                         errors="coerce", format="mixed")
    return df

# ---------- Step 6: impossible values and outliers ----------
def handle_outliers(df):
    df = df.copy()
    df.loc[~df["age"].between(10, 90), "age"] = np.nan        # rule based: age 250 is impossible
    df.loc[~df["rating"].between(1, 5), "rating"] = np.nan    # rating must be 1 to 5
    # IQR method for price (statistical outliers are capped, not deleted)
    q1, q3 = df["price"].quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    df["price"] = df["price"].clip(lower=max(low, 0), upper=high)
    return df

# ---------- Step 7: missing values ----------
def fill_missing(df):
    df = df.copy()
    df["age"] = df["age"].fillna(df["age"].median())
    df["price"] = df["price"].fillna(df.groupby("product")["price"].transform("median"))
    df["price"] = df["price"].fillna(df["price"].median())
    df["rating"] = df["rating"].fillna(df["rating"].median())
    df["skin_type"] = df["skin_type"].fillna("unknown")
    df["review"] = df["review"].fillna("no review")
    return df

# ---------- Step 8: normalisation and encoding ----------
def transform(df):
    df = df.copy()
    scaler = MinMaxScaler()
    df[["age_scaled", "price_scaled"]] = scaler.fit_transform(df[["age", "price"]])
    df = pd.get_dummies(df, columns=["skin_type"], prefix="skin")
    return df

# ---------- Step 9: validation ----------
def validate(df):
    assert df[["age", "price", "rating"]].isna().sum().sum() == 0, "missing values left"
    assert df["rating"].between(1, 5).all(), "rating out of range"
    assert (df["price"] > 0).all(), "invalid price"
    print("\nAll validation checks passed.")

def run_pipeline():
    df = make_raw_data()
    inspect(df, "RAW DATA")
    df = drop_duplicates(df)
    df = standardise_text(df)
    df = clean_price(df)
    df = clean_dates(df)
    df = handle_outliers(df)
    df = fill_missing(df)
    df = transform(df)
    validate(df)
    inspect(df, "CLEAN DATA")
    return df

if __name__ == "__main__":
    raw = make_raw_data()
    raw.to_csv("raw_beauty_data.csv", index=False)
    clean = run_pipeline()
    clean.to_csv("clean_beauty_data.csv", index=False)
    print(clean.head(10).to_string())
```