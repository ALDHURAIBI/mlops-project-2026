
"""
prepare.py — Network Intrusion Detection
MLOps Project 2026 | Chapter 1.2

Purpose
-------
Prepare the NF-UNSW-NB15-v3 dataset for machine learning.

Steps:
1. Read parameters from params.yaml.
2. Load a reproducible sample of network flows.
3. Convert the target into binary labels.
4. Remove identifiers and potentially leaking features.
5. Split data into train, validation and test datasets.
6. Save prepared datasets for the training pipeline.

The script is designed to run independently from Jupyter.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import yaml

from sklearn.model_selection import GroupShuffleSplit


# ============================================================
# 1. PROJECT PATHS AND CONFIGURATION
# ============================================================

# Determine the project root automatically.
# This works regardless of the terminal's current directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data/raw/NF-UNSW-NB15-v3.csv"
OUTPUT_DIR = PROJECT_ROOT / "data/prepared"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Read experiment parameters from YAML.
with open(PROJECT_ROOT / "params.yaml") as file:
    params = yaml.safe_load(file)

config = params["prepare"]

SEED = int(config["seed"])
SAMPLE_SIZE = int(config["sample_size"])
SAMPLE_FRACTION = float(config["sample_fraction"])
TEST_SIZE = float(config["test_size"])
VALIDATION_SIZE = float(config["validation_size"])
TARGET = config["target"]

print("Starting data preparation...")
print("Dataset:", DATA_PATH)
print("Sample size:", SAMPLE_SIZE)


# ============================================================
# 2. LOAD A REPRODUCIBLE SAMPLE
# ============================================================

# The original CSV has approximately 2.36 million rows.
# Reading it in chunks avoids loading all rows into memory.
#
# We use the same sampling approach as our initial notebook:
# sample a fraction of each chunk, then limit the result
# to the requested sample size.

sample_parts = []
total_rows = 0

for chunk in pd.read_csv(DATA_PATH, chunksize=100_000):

    sampled = chunk.sample(
        frac=SAMPLE_FRACTION,
        random_state=SEED
    )

    sample_parts.append(sampled)
    total_rows += len(chunk)

    print(f"Processed {total_rows:,} rows", end="\r")

df = pd.concat(sample_parts, ignore_index=True)

if len(df) > SAMPLE_SIZE:
    df = df.sample(
        n=SAMPLE_SIZE,
        random_state=SEED
    ).reset_index(drop=True)

print(f"\nOriginal rows: {total_rows:,}")
print(f"Sample rows: {len(df):,}")


# ============================================================
# 3. PREPARE BINARY TARGET
# ============================================================

# Binary classification:
# 0 = Benign (normal network traffic)
# 1 = Attack (malicious network traffic)

label_mapping = {
    "0": 0,
    "0.0": 0,
    "benign": 0,
    "normal": 0,
    "1": 1,
    "1.0": 1,
    "attack": 1,
    "malicious": 1,
}

label_text = (
    df[TARGET]
    .astype(str)
    .str.strip()
    .str.lower()
)

y = label_text.map(label_mapping)

# Stop execution if target values are unexpected.
# This prevents silently training on incorrect labels.
if y.isna().any():
    raise ValueError(
        f"Unexpected target labels: "
        f"{df.loc[y.isna(), TARGET].unique()}"
    )

y = y.astype(int)

print("\nClass distribution:")
print(y.value_counts())
print(f"Attack percentage: {y.mean() * 100:.2f}%")


# ============================================================
# 4. SELECT FEATURES AND REDUCE LEAKAGE
# ============================================================

# Exclude classification targets and network identifiers.
#
# Some variables could allow the model to recognize
# specific hosts, traffic scenarios or known attack patterns.
#
# This is our initial conservative feature-selection policy.
# It does not guarantee complete protection from leakage.

excluded_columns = {
    TARGET,
    "Attack",
    "FLOW_START_MILLISECONDS",
    "FLOW_END_MILLISECONDS",
    "IPV4_SRC_ADDR",
    "IPV4_DST_ADDR",
    "L4_SRC_PORT",
    "L4_DST_PORT",
    "PROTOCOL",
    "L7_PROTO",
    "IN_BYTES",
    "OUT_BYTES",
}

# Exclude additional IP, timestamp and TTL identifiers.
for column in df.columns:

    name = column.upper()

    if any(token in name for token in [
        "IPV4_", "IPV6_", "IP_ADDR",
        "TIMESTAMP", "MILLISECONDS", "TTL"
    ]) and any(token in name for token in [
        "ADDR", "TIMESTAMP", "MILLISECONDS", "TTL"
    ]):
        excluded_columns.add(column)

    if "TTL" in name:
        excluded_columns.add(column)

# Use numeric columns only for the initial experiment.
numeric_columns = df.select_dtypes(
    include=[np.number]
).columns.tolist()

feature_columns = [
    col for col in numeric_columns
    if col not in excluded_columns
]

X = df[feature_columns].copy()

# Replace infinity with missing values.
# Imputation will be fitted using training data only.
X = X.replace([np.inf, -np.inf], np.nan)

# Match our notebook's initial feature-selection method.
empty_columns = X.columns[X.isna().all()].tolist()
X = X.drop(columns=empty_columns)

feature_columns = X.columns.tolist()

if not feature_columns:
    raise ValueError("No usable features found.")

print(f"\nSelected features: {len(feature_columns)}")


# ============================================================
# 5. GROUP-AWARE DATA SPLITTING
# ============================================================

# Identical selected-feature vectors should not appear
# in different dataset partitions.
#
# Hash each feature vector and split by these group IDs.
# This reduces duplicate leakage, but it does not replace
# a future chronological or session-based evaluation.

groups = pd.util.hash_pandas_object(
    X, index=False
).to_numpy()

outer_split = GroupShuffleSplit(
    n_splits=1,
    test_size=TEST_SIZE,
    random_state=SEED
)

train_val_idx, test_idx = next(
    outer_split.split(X, y, groups=groups)
)

inner_split = GroupShuffleSplit(
    n_splits=1,
    test_size=VALIDATION_SIZE,
    random_state=SEED
)

inner_train, inner_val = next(
    inner_split.split(
        X.iloc[train_val_idx],
        y.iloc[train_val_idx],
        groups=groups[train_val_idx]
    )
)

train_idx = train_val_idx[inner_train]
val_idx = train_val_idx[inner_val]

# Verify that groups do not overlap across partitions.
train_groups = set(groups[train_idx])
val_groups = set(groups[val_idx])
test_groups = set(groups[test_idx])

assert train_groups.isdisjoint(val_groups)
assert train_groups.isdisjoint(test_groups)
assert val_groups.isdisjoint(test_groups)

# Verify all three datasets contain both classes.
for name, idx in [
    ("train", train_idx),
    ("validation", val_idx),
    ("test", test_idx)
]:
    if y.iloc[idx].nunique() != 2:
        raise ValueError(
            f"{name} partition contains only one class."
        )


# ============================================================
# 6. SAVE PREPARED DATASETS
# ============================================================

# Save features and target together.
# train.py and evaluate.py will load these CSV files.

for name, idx in [
    ("train", train_idx),
    ("val", val_idx),
    ("test", test_idx)
]:

    dataset = X.iloc[idx].copy()
    dataset[TARGET] = y.iloc[idx].to_numpy()

    output_path = OUTPUT_DIR / f"{name}.csv"

    dataset.to_csv(output_path, index=False)

    print(
        f"Saved {name}: {len(dataset):,} rows "
        f"({dataset[TARGET].mean() * 100:.2f}% attacks)"
    )

# Save feature column order.
# This is important for consistent model inference.
with open(OUTPUT_DIR / "feature_columns.json", "w") as file:
    json.dump(feature_columns, file, indent=2)

# Save metadata for reproducibility and documentation.
metadata = {
    "original_rows": total_rows,
    "sample_rows": len(df),
    "feature_count": len(feature_columns),
    "train_rows": len(train_idx),
    "validation_rows": len(val_idx),
    "test_rows": len(test_idx),
    "seed": SEED,
    "split_method": "GroupShuffleSplit on feature hashes"
}

with open(OUTPUT_DIR / "metadata.json", "w") as file:
    json.dump(metadata, file, indent=2)

print("\nData preparation completed successfully!")
print("Prepared files saved in:", OUTPUT_DIR)
