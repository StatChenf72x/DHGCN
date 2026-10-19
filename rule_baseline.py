# -*- coding: utf-8 -*-
import os
import collections
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


class Config:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(current_dir, "Data")
        self.out_csv = os.path.join(current_dir, "RuleBaseline_Results.csv")
        self.out_per_class = os.path.join(current_dir, "RuleBaseline_per_class.csv")
        self.num_classes = 10


args = Config()

CLASS_TO_PAIR = {
    0: (1, 1),
    1: (0, 0),
    2: (0, 1),
    3: (0, 2),
    4: (1, 0),
    5: (1, 2),
    6: (2, 0),
    7: (2, 1),
    8: (2, 2),
}
PAIR_TO_CLASS = {v: k for k, v in CLASS_TO_PAIR.items()}


def majority_polarity(counter):
    max_cnt = max(counter.values())
    return sorted([p for p, c in counter.items() if c == max_cnt])[0]


def build_tables(df_train):
    dg_counts = collections.defaultdict(collections.Counter)
    gd_counts = collections.defaultdict(collections.Counter)
    df_pos = df_train[df_train["label"] <= 8]
    for d, g, s, y in zip(df_pos["drug"], df_pos["gene"], df_pos["disease"], df_pos["label"]):
        dg_pol, gd_pol = CLASS_TO_PAIR[int(y)]
        dg_counts[(d, g)][dg_pol] += 1
        gd_counts[(g, s)][gd_pol] += 1
    dg_table = {k: majority_polarity(v) for k, v in dg_counts.items()}
    gd_table = {k: majority_polarity(v) for k, v in gd_counts.items()}
    dg_conflict = sum(1 for v in dg_counts.values() if len(v) > 1)
    gd_conflict = sum(1 for v in gd_counts.values() if len(v) > 1)
    return dg_table, gd_table, dg_conflict, gd_conflict


def predict(df, dg_table, gd_table):
    preds, covered = [], []
    for d, g, s in zip(df["drug"], df["gene"], df["disease"]):
        dg_pol = dg_table.get((d, g))
        gd_pol = gd_table.get((g, s))
        if dg_pol is None or gd_pol is None:
            preds.append(9)
            covered.append(0)
        else:
            preds.append(PAIR_TO_CLASS[(dg_pol, gd_pol)])
            covered.append(1)
    return np.array(preds), np.array(covered)


def report(name, df, preds, covered):
    y = df["label"].values
    acc = accuracy_score(y, preds)
    f1m = f1_score(y, preds, average="macro", zero_division=0)
    prem = precision_score(y, preds, average="macro", zero_division=0)
    recm = recall_score(y, preds, average="macro", zero_division=0)
    print(f"[{name}] n={len(y)} coverage={covered.mean():.4f} ACC={acc:.4f} "
          f"Macro-PRE={prem:.4f} Macro-REC={recm:.4f} Macro-F1={f1m:.4f}")
    rows = []
    for c in range(10):
        mask = y == c
        if mask.sum() == 0:
            continue
        rows.append({"true_class": c, "n": int(mask.sum()),
                     "coverage": float(covered[mask].mean()),
                     "acc": float((preds[mask] == y[mask]).mean())})
    per_class = pd.DataFrame(rows)
    print(per_class.to_string(index=False))
    return {"split": name, "n": len(y), "coverage": covered.mean(), "ACC": acc,
            "Macro-PRE": prem, "Macro-REC": recm, "Macro-F1": f1m}, per_class


def main():
    df_train = pd.read_csv(os.path.join(args.data_dir, "train_14400.csv"))
    df_val = pd.read_csv(os.path.join(args.data_dir, "val_14400.csv"))
    df_test = pd.read_csv(os.path.join(args.data_dir, "test_14400.csv"))

    dg_table, gd_table, dg_conf, gd_conf = build_tables(df_train)
    print(f"training functional triplets: {(df_train['label'] <= 8).sum()}")
    print(f"unique (drug,gene) pairs: {len(dg_table)} (conflicting: {dg_conf}); "
          f"unique (gene,disease) pairs: {len(gd_table)} (conflicting: {gd_conf})")

    results, per_class_test = [], None
    for name, df in [("val", df_val), ("test", df_test)]:
        preds, covered = predict(df, dg_table, gd_table)
        res, per_class = report(name, df, preds, covered)
        results.append(res)
        if name == "test":
            per_class_test = per_class
            cm = confusion_matrix(df["label"].values, preds, labels=list(range(10)))
            print("\nConfusion matrix (rows=true, cols=pred):")
            print(pd.DataFrame(cm, index=list(range(10)), columns=list(range(10))).to_string())

    pd.DataFrame(results).to_csv(args.out_csv, index=False)
    per_class_test.to_csv(args.out_per_class, index=False)
    print(f"\nSaved: {args.out_csv}\n        {args.out_per_class}")


if __name__ == "__main__":
    main()
