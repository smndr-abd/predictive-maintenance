"""
preprocessing.py
-----------------
Loads the NASA C-MAPSS (FD001) turbofan degradation dataset, computes
Remaining Useful Life (RUL) labels, normalizes sensor readings, and
builds fixed-length sliding-window sequences suitable for LSTM /
Transformer training.

Dataset columns (26 total, whitespace separated, no header):
    unit_number, time_cycles, op_setting_1, op_setting_2, op_setting_3,
    sensor_1 ... sensor_21
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------
INDEX_COLS = ["unit_number", "time_cycles"]
SETTING_COLS = ["op_setting_1", "op_setting_2", "op_setting_3"]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
ALL_COLS = INDEX_COLS + SETTING_COLS + SENSOR_COLS

# Sensors that are constant (or near-constant) in FD001 and carry no
# predictive signal — dropping them is standard practice in the literature.
DROP_SENSORS = [
    "sensor_1", "sensor_5", "sensor_6", "sensor_10",
    "sensor_16", "sensor_18", "sensor_19",
]

FEATURE_COLS = [c for c in SETTING_COLS + SENSOR_COLS if c not in DROP_SENSORS]

# Standard RUL clipping used in most C-MAPSS papers: early-life degradation
# is roughly flat, so capping RUL prevents the model from over-fitting to
# an unrealistically precise "far from failure" signal.
RUL_CAP = 125

# Sliding window length (timesteps per training sequence)
SEQUENCE_LENGTH = 30


def load_raw(path: str) -> pd.DataFrame:
    """Load a whitespace-delimited C-MAPSS file (train or test) into a DataFrame."""
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.iloc[:, : len(ALL_COLS)]  # drop trailing empty columns if present
    df.columns = ALL_COLS
    return df


def load_rul(path: str) -> np.ndarray:
    """Load the ground-truth RUL file (one value per test engine)."""
    return pd.read_csv(path, sep=r"\s+", header=None).iloc[:, 0].values


def add_train_rul(df: pd.DataFrame) -> pd.DataFrame:
    """
    For the TRAINING set, every engine runs to failure, so RUL at each row
    is simply (max_cycle_for_this_engine - current_cycle), clipped at RUL_CAP.
    """
    max_cycle = df.groupby("unit_number")["time_cycles"].transform("max")
    df = df.copy()
    df["RUL"] = (max_cycle - df["time_cycles"]).clip(upper=RUL_CAP)
    return df


def fit_scaler(train_df: pd.DataFrame) -> MinMaxScaler:
    scaler = MinMaxScaler()
    scaler.fit(train_df[FEATURE_COLS])
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: MinMaxScaler) -> pd.DataFrame:
    df = df.copy()
    df[FEATURE_COLS] = scaler.transform(df[FEATURE_COLS])
    return df


def make_train_sequences(df: pd.DataFrame, seq_len: int = SEQUENCE_LENGTH):
    """
    Build sliding-window sequences for training.
    For each engine, every window of `seq_len` consecutive cycles becomes
    one training example; the label is the RUL at the LAST cycle in the window.
    Engines shorter than seq_len are skipped (rare in FD001).
    """
    X, y = [], []
    for _, unit_df in df.groupby("unit_number"):
        unit_df = unit_df.sort_values("time_cycles")
        feats = unit_df[FEATURE_COLS].values
        ruls = unit_df["RUL"].values
        n = len(unit_df)
        if n < seq_len:
            continue
        for start in range(n - seq_len + 1):
            end = start + seq_len
            X.append(feats[start:end])
            y.append(ruls[end - 1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def make_test_sequences(df: pd.DataFrame, seq_len: int = SEQUENCE_LENGTH):
    """
    Build ONE sequence per test engine: the last `seq_len` cycles available.
    If an engine has fewer than seq_len cycles, left-pad by repeating the
    first row (keeps the model's expected input shape).
    """
    X = []
    for _, unit_df in df.groupby("unit_number"):
        unit_df = unit_df.sort_values("time_cycles")
        feats = unit_df[FEATURE_COLS].values
        if len(feats) < seq_len:
            pad = np.repeat(feats[0:1], seq_len - len(feats), axis=0)
            feats = np.concatenate([pad, feats], axis=0)
        else:
            feats = feats[-seq_len:]
        X.append(feats)
    return np.array(X, dtype=np.float32)


def prepare_datasets(train_path: str, test_path: str, rul_path: str,
                      seq_len: int = SEQUENCE_LENGTH):
    """
    End-to-end pipeline: load raw files -> label RUL -> normalize -> window.
    Returns (X_train, y_train, X_test, y_test, scaler, feature_cols)
    """
    train_df = load_raw(train_path)
    test_df = load_raw(test_path)
    test_rul = load_rul(rul_path)

    train_df = add_train_rul(train_df)

    scaler = fit_scaler(train_df)
    train_df = apply_scaler(train_df, scaler)
    test_df = apply_scaler(test_df, scaler)

    X_train, y_train = make_train_sequences(train_df, seq_len)
    X_test = make_test_sequences(test_df, seq_len)
    y_test = np.clip(test_rul, a_max=RUL_CAP, a_min=None).astype(np.float32)

    return X_train, y_train, X_test, y_test, scaler, FEATURE_COLS


if __name__ == "__main__":
    X_train, y_train, X_test, y_test, scaler, feats = prepare_datasets(
        "../data/train_FD001.txt", "../data/test_FD001.txt", "../data/RUL_FD001.txt"
    )
    print("Feature columns:", feats)
    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_test:", X_test.shape, "y_test:", y_test.shape)
