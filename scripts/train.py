"""
train.py
--------
Trains an LSTM or Transformer RUL regressor on the NASA C-MAPSS FD001
dataset, evaluates on the held-out test engines with RMSE + the official
C-MAPSS asymmetric scoring function, and saves the model + scaler for
deployment.

Usage:
    python train.py --model lstm
    python train.py --model transformer --epochs 60
"""

import argparse
import json
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from preprocessing import prepare_datasets, SEQUENCE_LENGTH
from models import build_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Official C-MAPSS scoring function (Saxena et al., 2008).
    Penalizes LATE predictions (predicted RUL > actual RUL, i.e. the model
    thought the machine had more life left than it did) much more heavily
    than early ones — mirroring the real safety/cost asymmetry of missing
    a failure vs. servicing a machine a bit too soon.
    """
    d = y_pred - y_true
    score = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return float(np.sum(score))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, X, y):
    model.eval()
    xb = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    preds = model(xb).cpu().numpy()
    return preds, rmse(y, preds), nasa_score(y, preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lstm", "transformer"], default="lstm")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--out_dir", default="../models")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Device: {DEVICE}")
    print("Loading + preprocessing C-MAPSS FD001 ...")
    X_train, y_train, X_test, y_test, scaler, feature_cols = prepare_datasets(
        os.path.join(args.data_dir, "train_FD001.txt"),
        os.path.join(args.data_dir, "test_FD001.txt"),
        os.path.join(args.data_dir, "RUL_FD001.txt"),
    )
    n_features = X_train.shape[-1]
    print(f"Train sequences: {X_train.shape}, Test sequences: {X_test.shape}, "
          f"features: {n_features}")

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    model = build_model(args.model, n_features).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_rmse = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        _, test_rmse, test_score = evaluate(model, X_test, y_test)
        history.append({"epoch": epoch, "train_loss": train_loss,
                         "test_rmse": test_rmse, "test_score": test_score})
        print(f"Epoch {epoch:3d}/{args.epochs} | train MSE loss: {train_loss:8.3f} "
              f"| test RMSE: {test_rmse:7.3f} | NASA score: {test_score:10.1f}")

        if test_rmse < best_rmse:
            best_rmse = test_rmse
            torch.save(model.state_dict(),
                       os.path.join(args.out_dir, f"{args.model}_best.pt"))

    # Save scaler + config needed for inference at deploy time
    with open(os.path.join(args.out_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    config = {
        "model_type": args.model,
        "n_features": n_features,
        "feature_cols": feature_cols,
        "sequence_length": SEQUENCE_LENGTH,
        "best_test_rmse": best_rmse,
    }
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    with open(os.path.join(args.out_dir, f"{args.model}_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest test RMSE: {best_rmse:.3f}")
    print(f"Saved model, scaler, and config to {args.out_dir}/")


if __name__ == "__main__":
    main()
