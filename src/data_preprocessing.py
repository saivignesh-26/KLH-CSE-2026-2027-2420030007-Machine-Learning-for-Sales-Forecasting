import pandas as pd
import numpy as np
import os

print("===== E-COMMERCE SALES FORECASTING PROJECT =====")
print("Phase 1: Data Collection and Preprocessing")

# --------------------------------------------------
# 1. DATASET PATHS
# --------------------------------------------------

TRAIN_PATH = "data/raw/rossmann/train.csv"
STORE_PATH = "data/raw/rossmann/store.csv"
OUTPUT_PATH = "data/processed/rossmann_processed.csv"

# --------------------------------------------------
# 2. LOAD DATASETS
# --------------------------------------------------

print("\nLoading Rossmann datasets...")

train_df = pd.read_csv(
    TRAIN_PATH,
    low_memory=False,
    dtype={"StateHoliday": str}
)

store_df = pd.read_csv(STORE_PATH)

print("\nTrain dataset shape:", train_df.shape)
print("Store dataset shape:", store_df.shape)

# --------------------------------------------------
# 3. MERGE TRAIN AND STORE DATA
# --------------------------------------------------

print("\nMerging datasets...")

df = pd.merge(
    train_df,
    store_df,
    on="Store",
    how="left"
)

print("Merged dataset shape:", df.shape)

# --------------------------------------------------
# 4. HANDLE MISSING VALUES
# --------------------------------------------------

print("\nMissing values before cleaning:")
print(df.isnull().sum())

# Numerical columns
df["CompetitionDistance"] = df["CompetitionDistance"].fillna(
    df["CompetitionDistance"].median()
)

df["CompetitionOpenSinceMonth"] = df[
    "CompetitionOpenSinceMonth"
].fillna(0)

df["CompetitionOpenSinceYear"] = df[
    "CompetitionOpenSinceYear"
].fillna(0)

df["Promo2SinceWeek"] = df["Promo2SinceWeek"].fillna(0)

df["Promo2SinceYear"] = df["Promo2SinceYear"].fillna(0)

# Categorical column
df["PromoInterval"] = df["PromoInterval"].fillna("None")

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# --------------------------------------------------
# 5. REMOVE DUPLICATES
# --------------------------------------------------

duplicates_before = df.duplicated().sum()

df = df.drop_duplicates()

duplicates_after = df.duplicated().sum()

print("\nDuplicates before removal:", duplicates_before)
print("Duplicates after removal:", duplicates_after)

# --------------------------------------------------
# 6. CONVERT DATE COLUMN
# --------------------------------------------------

df["Date"] = pd.to_datetime(df["Date"])

# --------------------------------------------------
# 7. FEATURE ENGINEERING
# --------------------------------------------------

print("\nCreating time-based features...")

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)

# Weekend: 1 for Saturday/Sunday, otherwise 0
df["IsWeekend"] = df["DayOfWeek"].isin([6, 7]).astype(int)

# Avoid division by zero
df["SalesPerCustomer"] = np.where(
    df["Customers"] > 0,
    df["Sales"] / df["Customers"],
    0
)

# --------------------------------------------------
# 8. SAVE PROCESSED DATASET
# --------------------------------------------------

os.makedirs("data/processed", exist_ok=True)

df.to_csv(OUTPUT_PATH, index=False)

print("\n===== PREPROCESSING COMPLETED =====")

print("\nFinal dataset shape:", df.shape)

print("\nNew features created:")
print([
    "Year",
    "Month",
    "Day",
    "WeekOfYear",
    "IsWeekend",
    "SalesPerCustomer"
])

print("\nProcessed dataset saved successfully at:")
print(OUTPUT_PATH)