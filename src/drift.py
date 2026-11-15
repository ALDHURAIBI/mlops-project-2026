"""
drift.py — Network Intrusion Detection
MLOps Project 2026 | Iteration 1

Purpose
-------
Replay NF-CICIDS2018-v3 day by day to measure drift.

Usage: python src/drift.py <model>
dvc.yaml runs this script once per replayed model.

The model built by train.py was trained on NF-UNSW-NB15, another
network. Two versions of it are replayed on each CIC day:

- frozen: models/<model>/model.joblib, deployed as is;
- retrained: retrained every night on the UNSW training data plus
  all previous CIC days (the labels of a day arrive the next day).

On the first day there is no CIC history yet, so both are the same.
Main metric: MCC per day, with recall and false positive rate to
explain it. Two drift signals need no label: the share of flows
flagged, and the PSI of each feature against the UNSW data and
against the previous day.

Inputs:
    data/raw/NF-CICIDS2018-v3.parquet
    data/prepared/train.csv, val.csv, feature_columns.json
    models/<model>/model.joblib, model_info.json
    params.yaml

Outputs:
    evaluation/<model>/drift/days.csv       one row per day
    evaluation/<model>/drift/families.csv   recall per attack family and day
    evaluation/<model>/drift/summary.json   metrics compared with dvc metrics show
"""

from pathlib import Path
import json
import sys
import time

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from sklearn.metrics import matthews_corrcoef, roc_auc_score

# Model definitions: src/models/, one file per model.
from models import attack_score, build_model


# ============================================================
# 1. PROJECT PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL = sys.argv[1]

DATA_PATH = PROJECT_ROOT / "data/raw/NF-CICIDS2018-v3.parquet"
PREPARED_DIR = PROJECT_ROOT / "data/prepared"
MODEL_DIR = PROJECT_ROOT / "models" / MODEL
OUTPUT_DIR = PROJECT_ROOT / "evaluation" / MODEL / "drift"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(PROJECT_ROOT / "params.yaml") as file:
    params = yaml.safe_load(file)

SEED = int(params["prepare"]["seed"])
TARGET = params["prepare"]["target"]
TRAIN_CONFIG = params["train"]
config = params["drift"]

SAMPLE_FRACTION = float(config["sample_fraction"])
MIN_FLOWS_PER_DAY = int(config["min_flows_per_day"])
RETRAIN_CAP = int(config["retrain_cap"])
PSI_SAMPLE = int(config["psi_sample"])
PSI_ALERT = float(config["psi_alert"])

with open(PREPARED_DIR / "feature_columns.json") as file:
    FEATURES = json.load(file)

print("Starting drift replay...")
print("Dataset:", DATA_PATH)
print(f"Features: {len(FEATURES)}")


# ============================================================
# 2. LOAD A SAMPLE OF NF-CICIDS2018
# ============================================================

# 20 M flows do not fit in memory with pandas.
# A fixed share of each batch is kept: the rates per day
# stay the same and the sample is reproducible.

start_time = time.perf_counter()
rng = np.random.default_rng(SEED)
parts = []

for batch in pq.ParquetFile(DATA_PATH).iter_batches(
    batch_size=1_000_000,
    columns=FEATURES + ["FLOW_START_MILLISECONDS", "Label", "Attack"]
):
    part = batch.to_pandas()
    parts.append(part[rng.random(len(part)) < SAMPLE_FRACTION])

df = pd.concat(parts, ignore_index=True)
del parts

df[FEATURES] = df[FEATURES].astype("float32")
df["y"] = df["Label"].astype(int)
df["day"] = pd.to_datetime(
    df["FLOW_START_MILLISECONDS"], unit="ms"
).dt.normalize()

print(
    f"{len(df):,} flows loaded in "
    f"{time.perf_counter() - start_time:.0f}s"
)

# Readable attack family names.
NAMES = {
    "FTP-BruteForce": "FTP brute force",
    "SSH-Bruteforce": "SSH brute force",
    "DoS_attacks-GoldenEye": "DoS GoldenEye",
    "DoS_attacks-Slowloris": "DoS Slowloris",
    "DoS_attacks-SlowHTTPTest": "DoS SlowHTTPTest",
    "DoS_attacks-Hulk": "DoS Hulk",
    "DDoS_attacks-LOIC-HTTP": "DDoS LOIC-HTTP",
    "DDOS_attack-HOIC": "DDoS HOIC",
    "DDOS_attack-LOIC-UDP": "DDoS LOIC-UDP",
    "Brute_Force_-Web": "web brute force",
    "Brute_Force_-XSS": "XSS",
    "SQL_Injection": "SQL injection",
    "Infilteration": "infiltration",
    "Bot": "botnet",
}
df["family"] = df["Attack"].map(NAMES).fillna(df["Attack"])

# Families grouped by kind, for short day labels.
KIND = {
    "FTP brute force": "brute force",
    "SSH brute force": "brute force",
    "web brute force": "web attacks",
    "XSS": "web attacks",
    "SQL injection": "web attacks",
    "infiltration": "infiltration",
    "botnet": "botnet",
}


def kind(family):
    # DoS ..., DDoS ...
    return KIND.get(family, family.split()[0])


# Days with almost no traffic are skipped (27 Feb).
flows_per_day = df.groupby("day").size()
DAYS = flows_per_day.index[
    flows_per_day >= MIN_FLOWS_PER_DAY
].tolist()

print(f"Days replayed: {len(DAYS)} of {len(flows_per_day)}")


# ============================================================
# 3. FROZEN MODEL AND UNSW TRAINING DATA
# ============================================================

frozen = joblib.load(MODEL_DIR / "model.joblib")

with open(MODEL_DIR / "model_info.json") as file:
    model_info = json.load(file)

if model_info["feature_columns"] != FEATURES:
    raise ValueError("Model and prepared data use different features.")

# The retrained model starts from the UNSW data of prepare.py.
unsw = pd.concat(
    [pd.read_csv(PREPARED_DIR / f"{name}.csv") for name in ("train", "val")],
    ignore_index=True
)
unsw = unsw.rename(columns={TARGET: "y"})
unsw[FEATURES] = unsw[FEATURES].astype("float32")

print(f"Frozen model: {model_info['model_name']}")
print(f"UNSW flows for retraining: {len(unsw):,}")


# ============================================================
# 4. REPLAY DAY BY DAY
# ============================================================

def rates(y, pred, score):
    attack, benign = y == 1, y == 0
    both = attack.any() and benign.any()
    return {
        "recall": pred[attack].mean() if attack.any() else np.nan,
        "false_positive_%": (
            100 * pred[benign].mean() if benign.any() else np.nan
        ),
        # Undefined on a day without attack (23 Feb).
        "mcc": matthews_corrcoef(y, pred) if both else np.nan,
        # Known right away, without labels.
        "flagged_%": 100 * pred.mean(),
        # Ranking of attacks above benign flows, whatever the threshold.
        "auc": roc_auc_score(y, score) if both else np.nan,
    }


results, families = [], []

for i, day in enumerate(DAYS):

    d = df[df["day"] == day]
    y = d["y"].to_numpy()

    pred_frozen = frozen.predict(d[FEATURES])
    score_frozen = attack_score(frozen, d[FEATURES])
    pred_retrained, score_retrained = pred_frozen, score_frozen

    if i > 0:
        history = df[df["day"].isin(DAYS[:i])]
        if len(history) > RETRAIN_CAP:
            history = history.sample(RETRAIN_CAP, random_state=SEED)
        history = pd.concat(
            [unsw[FEATURES + ["y"]], history[FEATURES + ["y"]]],
            ignore_index=True
        )
        retrained = build_model(MODEL, TRAIN_CONFIG, SEED).fit(
            history[FEATURES], history["y"]
        )
        pred_retrained = retrained.predict(d[FEATURES])
        score_retrained = attack_score(retrained, d[FEATURES])

    row = {"day": day, "flows": len(d), "attack_%": 100 * y.mean()}
    row.update({
        f"frozen_{k}": v
        for k, v in rates(y, pred_frozen, score_frozen).items()
    })
    row.update({
        f"retrained_{k}": v
        for k, v in rates(y, pred_retrained, score_retrained).items()
    })
    results.append(row)

    attacks = d.assign(
        frozen=pred_frozen, retrained=pred_retrained
    )[y == 1]
    for family, g in attacks.groupby("family"):
        families.append({
            "day": day,
            "family": family,
            "flows": len(g),
            "recall_frozen": g["frozen"].mean(),
            "recall_retrained": g["retrained"].mean(),
        })

    print(
        f"{day:%d %b}: MCC frozen {row['frozen_mcc']:.2f}, "
        f"retrained {row['retrained_mcc']:.2f}",
        flush=True
    )

results = pd.DataFrame(results)
families = pd.DataFrame(families)


# ============================================================
# 5. DRIFT SIGNAL WITHOUT LABELS (PSI)
# ============================================================

# Highest feature PSI between the traffic of the day and
# two references: the UNSW training data and the previous day.

def psi(ref, cur, bins=10, eps=1e-4):
    """Population Stability Index of one feature, bins = deciles of the reference."""
    ref, cur = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2 or len(cur) == 0:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.clip(np.histogram(ref, edges)[0] / len(ref), eps, None)
    c = np.clip(np.histogram(cur, edges)[0] / len(cur), eps, None)
    return float(np.sum((c - r) * np.log(c / r)))


def max_psi(ref, cur):
    values = pd.Series({
        f: psi(ref[f].to_numpy(), cur[f].to_numpy()) for f in FEATURES
    })
    return values.max(), values.idxmax()


def sample(d):
    return d.sample(min(PSI_SAMPLE, len(d)), random_state=SEED)


drift = []

for i, day in enumerate(DAYS):

    today = df[df["day"] == day]
    seen = set(df.loc[df["day"].isin(DAYS[:i]) & (df["y"] == 1), "family"])

    row = {
        "day": day,
        "new_family": bool(
            set(today.loc[today["y"] == 1, "family"]) - seen
        ),
    }
    row["psi_vs_unsw"], row["top_feature_unsw"] = max_psi(
        sample(unsw), today
    )
    if i > 0:
        previous = df[df["day"] == DAYS[i - 1]]
        (row["psi_vs_previous_day"],
         row["top_feature_previous_day"]) = max_psi(sample(previous), today)

    drift.append(row)

drift = pd.DataFrame(drift)


# ============================================================
# 6. SAVE RESULTS
# ============================================================

day_kind = (
    df[df["y"] == 1]
    .groupby("day")["family"]
    .agg(lambda s: " + ".join(sorted({kind(f) for f in s})))
)

days = results.merge(drift, on="day")
days["attacks"] = [day_kind.get(d, "no attack") for d in days["day"]]
days["day"] = days["day"].dt.strftime("%Y-%m-%d")
days.to_csv(OUTPUT_DIR / "days.csv", index=False)

families["day"] = families["day"].dt.strftime("%Y-%m-%d")
families.to_csv(OUTPUT_DIR / "families.csv", index=False)

# Summary compared across iterations with `dvc metrics diff`.
# The retrained model only differs from the frozen one from
# the second day on, so its figures start there.
later = days.iloc[1:]
new_days = later[later["new_family"]]
control = later[later["attacks"] == "no attack"]

summary = {
    "days": len(days),
    "frozen": {
        "mean_mcc": float(days["frozen_mcc"].mean()),
        "mean_false_positive_%": float(
            days["frozen_false_positive_%"].mean()
        ),
    },
    "retrained": {
        "mean_mcc": float(later["retrained_mcc"].mean()),
        "mean_mcc_new_family_days": float(new_days["retrained_mcc"].mean()),
        "mean_false_positive_%": float(
            later["retrained_false_positive_%"].mean()
        ),
    },
    "psi_vs_previous_day": {
        "new_family_days": len(new_days),
        "alerts_on_new_family_days": int(
            (new_days["psi_vs_previous_day"] > PSI_ALERT).sum()
        ),
        "alerts_on_control_days": int(
            (control["psi_vs_previous_day"] > PSI_ALERT).sum()
        ),
    },
}

with open(OUTPUT_DIR / "summary.json", "w") as file:
    json.dump(summary, file, indent=2)

print("\nSummary:")
print(json.dumps(summary, indent=2))
print("\nDrift replay completed successfully!")
print("Results saved in:", OUTPUT_DIR)
