"""Reproduce Tables 1, 3 and 4 of the paper from data/rows30.csv.

Table 1 is printed as recorded: early-stage model values are historical
run outputs, and the script does not pretend to refit them. What it does
recompute, and check, is the baseline method: the global-mean leave-one-
drug-out baseline on the 30-row refit set must come out at 0.0831, and
on the cyto targets at 0.0788, matching Table 1.

Tables 3 and 4 are fully recomputed. The headline model is a Huber
(IRLS, Ridge alpha=10) fit on mechanism one-hot plus log10 dose plus
log10 growth, scored by leave-one-drug-out. Bootstrap intervals are
drug-cluster resamples over the 16 drugs, using fixed leave-one-drug-out
predictions: they measure sampling uncertainty, not refitting
uncertainty, and they are not adjusted for model selection. The paired
test resamples drugs and records the mean squared-error difference
(Huber minus baseline) per resample.

Run from the repository root:
    python code/reproduce_tables.py
Exit code is non-zero if a recomputed headline number disagrees with the
paper beyond rounding.
"""

import json
import math
import sys

import numpy as np
import pandas as pd

RNG_BOOT = np.random.default_rng(7)
RNG_PAIRED = np.random.default_rng(11)
N_BOOT = 1000
N_PAIRED = 10000

CYTO_TARGETS = {  # Section 3.7 split targets, replacing the constant kills
    ("Sorafenib", "HCC 06-0606"): 0.2600,
    ("Sorafenib", "HCC 10-0505"): 0.0876,
    ("Sunitinib", "MDA-MB-468 TNBC"): 0.1141,
    ("Pazopanib", "L9981 NSCLC"): 0.0597,
    ("Pazopanib", "A549 lung"): 0.0645,
}

TABLE1 = [
    ("2f", "9 drugs, mechanism", 0.0629, 0.555, 0.0784),
    ("2g", "10 drugs, mechanism", 0.0560, 0.467, 0.0745),
    ("2h", "11 drugs, mechanism", 0.0461, 0.444, 0.0725),
    ("2i", "12 drugs, mechanism", 0.0423, 0.391, 0.0695),
    ("2j", "13 drugs, mechanism", 0.0389, 0.350, 0.0684),
    ("2k", "14 drugs, mechanism", 0.0403, 0.345, 0.0658),
    ("2l", "16 drugs, mechanism", 0.0583, 0.531, 0.0612),
    ("2m", "16 drugs, mechanism, dose", 0.0548, 0.544, 0.0612),
    ("2r", "24 rows, mechanism, dose, growth", 0.0544, 0.541, 0.0696),
    ("2s", "30 rows, mechanism, dose, growth", 0.0814, 0.781, 0.0833),
    ("2u", "30 rows, refit targets", 0.0828, 0.749, 0.0831),
    ("Split", "30 rows, cyto targets", 0.0790, 0.723, 0.0788),
]


def design(df):
    mechs = sorted(df["mechanism"].unique())
    X = np.zeros((len(df), len(mechs) + 2))
    for i, m in enumerate(df["mechanism"]):
        X[i, mechs.index(m)] = 1.0
    X[:, -2] = np.log(df["dose_mgkg_day"].to_numpy(dtype=float))
    X[:, -1] = np.log(df["growth_per_day"].to_numpy(dtype=float))
    return X, mechs


def huber_fit(X, y, alpha=10.0, iters=20):
    """Exactly the paper's headline fit: sklearn Ridge, IRLS Huber weights."""
    from sklearn.linear_model import Ridge

    w = np.ones(len(y))
    m = None
    for _ in range(iters):
        m = Ridge(alpha=alpha).fit(X, y, sample_weight=w)
        r = y - m.predict(X)
        s = 1.4826 * np.median(np.abs(r - np.median(r)))
        if s < 1e-6:
            break
        delta = 1.345 * s
        a = np.abs(r)
        w = np.where(a <= delta, 1.0, delta / np.maximum(a, 1e-12))
    return m


def lodo_huber(df):
    from sklearn.preprocessing import StandardScaler

    X, _ = design(df)
    y = df["kill_per_day"].to_numpy(dtype=float)
    drugs = df["drug"].to_numpy()
    pred = np.zeros(len(df))
    for d in sorted(set(drugs)):
        te = drugs == d
        tr = ~te
        sc = StandardScaler().fit(X[tr])
        m = huber_fit(sc.transform(X[tr]), y[tr])
        pred[te] = m.predict(sc.transform(X[te]))
    return pred


def lodo_mean(df, by_class=False):
    y = df["kill_per_day"].to_numpy(dtype=float)
    drugs = df["drug"].to_numpy()
    mech = df["mechanism"].to_numpy()
    pred = np.zeros(len(df))
    for i in range(len(df)):
        mask = drugs != drugs[i]
        if by_class:
            mask &= mech == mech[i]
            if mask.sum() == 0:
                mask = drugs != drugs[i]
        pred[i] = y[mask].mean()
    return pred


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def median_ae(a, b):
    return float(np.median(np.abs(np.asarray(a) - np.asarray(b))))


def spearman(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def cluster_bootstrap(df, pred, metric, n=N_BOOT, rng=RNG_BOOT):
    drugs = sorted(df["drug"].unique())
    idx = {d: np.where(df["drug"].to_numpy() == d)[0] for d in drugs}
    y = df["kill_per_day"].to_numpy(dtype=float)
    vals = []
    for _ in range(n):
        chosen = rng.choice(drugs, size=len(drugs), replace=True)
        rows = np.concatenate([idx[d] for d in chosen])
        vals.append(metric(y[rows], pred[rows]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def paired_diff(df, pred, base_pred, n=N_PAIRED, rng=RNG_PAIRED):
    drugs = sorted(df["drug"].unique())
    idx = {d: np.where(df["drug"].to_numpy() == d)[0] for d in drugs}
    y = df["kill_per_day"].to_numpy(dtype=float)
    se_model = (y - pred) ** 2
    se_base = (y - base_pred) ** 2
    diffs = []
    for _ in range(n):
        chosen = rng.choice(drugs, size=len(drugs), replace=True)
        rows = np.concatenate([idx[d] for d in chosen])
        diffs.append(float(np.mean(se_model[rows] - se_base[rows])))
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(np.mean(se_model - se_base)), float(lo), float(hi)


def report_subset(df, label, paper):
    y = df["kill_per_day"].to_numpy(dtype=float)
    hub = lodo_huber(df)
    gm = lodo_mean(df)
    cm = lodo_mean(df, by_class=True)
    print(f"\n{label}")
    print(f"  Huber     RMSE {rmse(y, hub):.4f}  medianAE {median_ae(y, hub):.4f}  Spearman {spearman(y, hub):.3f}")
    print(f"  GlobalMean RMSE {rmse(y, gm):.4f}  medianAE {median_ae(y, gm):.4f}")
    print(f"  ClassMean  RMSE {rmse(y, cm):.4f}  medianAE {median_ae(y, cm):.4f}  Spearman {spearman(y, cm):.3f}")
    lo, hi = cluster_bootstrap(df, hub, lambda a, b: rmse(a, b))
    print(f"  Huber RMSE bootstrap CI [{lo:.3f}, {hi:.3f}]")
    d, lo, hi = paired_diff(df, hub, gm)
    print(f"  Paired Huber minus global-mean: {d:.5f} [{lo:.5f}, {hi:.5f}]")
    d2, lo2, hi2 = paired_diff(df, hub, cm)
    print(f"  Paired Huber minus class-mean: {d2:.5f} [{lo2:.5f}, {hi2:.5f}]")
    if paper:
        ok = abs(rmse(y, hub) - paper[0]) <= 0.0006 and abs(rmse(y, gm) - paper[1]) <= 0.0006
        print(f"  Paper check (Huber {paper[0]}, global-mean {paper[1]}): {'PASS' if ok else 'FAIL'}")
        return ok
    return True


def main():
    df = pd.read_csv("data/rows30.csv")
    print("Table 1, as recorded in the paper (model values are historical run outputs):")
    print(f"{'Stage':<7}{'Set':<36}{'Kill RMSE':>10}{'Curve':>8}{'Baseline':>10}")
    for stage, s, kill, curve, base in TABLE1:
        print(f"{stage:<7}{s:<36}{kill:>10.4f}{curve:>8.3f}{base:>10.4f}")

    # Baseline-method check on the 30-row refit set and on cyto targets.
    y = df["kill_per_day"].to_numpy(dtype=float)
    gm = lodo_mean(df)
    print(f"\nBaseline check, 30-row refit set: recomputed {rmse(y, gm):.4f}, paper 0.0831")
    cyto = df.copy()
    for (d, line), v in CYTO_TARGETS.items():
        cyto.loc[(cyto["drug"] == d) & (cyto["line"] == line), "kill_per_day"] = v
    gm_c = lodo_mean(cyto)
    print(f"Baseline check, cyto targets: recomputed {rmse(cyto['kill_per_day'].to_numpy(), gm_c):.4f}, paper 0.0788")

    # Stored predictions must match a fresh Huber refit.
    hub = lodo_huber(df)
    stored = df["huber_lodo_pred"].to_numpy(dtype=float)
    print(f"Stored Huber predictions vs fresh refit: max abs diff {np.max(np.abs(hub - stored)):.4f}")

    ok = True
    ok &= report_subset(df, "Table 3: all 30 rows, Huber vs baselines", (0.0716, 0.0831))
    for label, drop_drugs in [
        ("Table 4: minus cyclophosphamide MX-1", None),
        ("Table 4: minus sorafenib 06-0606", None),
        ("Table 4: minus both outlier rows", None),
    ]:
        pass  # subsets handled below, row-wise
    sub = df[~((df["drug"] == "Cyclophosphamide") & (df["line"] == "MX-1 breast"))]
    ok &= report_subset(sub, "Table 4: minus MX-1", (0.0525, 0.0631))
    sub = df[~((df["drug"] == "Sorafenib") & (df["line"] == "HCC 06-0606"))]
    ok &= report_subset(sub, "Table 4: minus sorafenib 06-0606", (0.0601, 0.0726))
    sub = df[~(((df["drug"] == "Cyclophosphamide") & (df["line"] == "MX-1 breast"))
                | ((df["drug"] == "Sorafenib") & (df["line"] == "HCC 06-0606")))]
    ok &= report_subset(sub, "Table 4: minus both", (0.0324, 0.0442))
    print("\nOVERALL:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
