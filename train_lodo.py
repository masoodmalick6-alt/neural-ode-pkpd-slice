"""Train the conditional encoder and report main-split + LODO metrics.

Usage (from the multidrug directory, with data/ next to it):
    python train_lodo.py

Data layout expected under data/:
    GDSC1_fitted_dose_response_24Jul22.csv
    rnaseq_merged_rsem_tpm_20260323.csv
    drug_smiles.csv

Training: minibatch 1024, Adam lr 0.01, grad clip 1.0.
Main split: 400 steps on a random 80/20 pair split.
LODO folds: 300 steps each, holding out Erlotinib, Trametinib, Cisplatin.

READ THE CAVEAT: the random-pair main split leaks (the same cell lines
and drugs appear in train and test), so its numbers are inflated. The
leave-one-drug-out folds are the honest metric: a held-out drug was
never seen in training.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge

from data_loader import (load_gdsc1, load_tpm, build_pair_table,
                         build_expression_matrix, select_top_genes,
                         featurize, smiles_to_fingerprints)
from ode_engine import simulate
from conditional_encoder import ConditionalEncoder

DATA = "data"
LODO_DRUGS = ["Erlotinib", "Trametinib", "Cisplatin"]
DOSE, V, K, T0 = 500.0, 50.0, 100.0, 10.0


def corr(p, q):
    p = p - p.mean()
    q = q - q.mean()
    return float((p * q).mean() / (p.std() * q.std()))


def rmse(p, q):
    return float(np.sqrt(np.mean((p - q) ** 2)))


def run_experiment(model, tr, te, steps, seed=0):
    torch.manual_seed(seed)
    (Xg_tr, Xf_tr, y_tr), (Xg_te, Xf_te, y_te) = tr, te
    Xg_tr = torch.tensor(Xg_tr)
    Xf_tr = torch.tensor(Xf_tr)
    Xg_te = torch.tensor(Xg_te)
    Xf_te = torch.tensor(Xf_te)
    y_tr_t = torch.tensor(y_tr)
    n = len(y_tr)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)

    def predict(Xg, Xf):
        CL, growth, kill = model.parameters_for(Xg, Xf)
        T30 = simulate(CL, growth, kill, dose=DOSE, V=V, K=K, T0=T0)[-1, :, 1]
        return model.predict_ln_ic50(Xg, Xf, T30, T0=T0)

    for step in range(steps + 1):
        idx = torch.randint(0, n, (min(1024, n),))
        opt.zero_grad()
        loss = ((predict(Xg_tr[idx], Xf_tr[idx]) - y_tr_t[idx]) ** 2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0:
            print("step", step, "loss", round(loss.item(), 4))
    with torch.no_grad():
        pred = predict(Xg_te, Xf_te).numpy()
    ridge = Ridge(alpha=1.0).fit(
        np.concatenate([Xg_tr.numpy(), Xf_tr.numpy()], axis=1), y_tr)
    pred_r = ridge.predict(
        np.concatenate([Xg_te.numpy(), Xf_te.numpy()], axis=1))
    return pred, pred_r, y_te


def main():
    gdsc = load_gdsc1(f"{DATA}/GDSC1_fitted_dose_response_24Jul22.csv")
    expr = load_tpm(f"{DATA}/rnaseq_merged_rsem_tpm_20260323.csv")
    smiles = pd.read_csv(f"{DATA}/drug_smiles.csv")
    drugs, fp, drug_index = smiles_to_fingerprints(smiles)
    pairs = build_pair_table(gdsc, drugs)
    pairs = pairs[pairs["SANGER_MODEL_ID"].isin(expr.columns)]
    mat = build_expression_matrix(expr, pairs["SANGER_MODEL_ID"])
    print("pairs:", len(pairs), "drugs:", pairs["DRUG_NAME"].nunique(),
          "models:", mat.shape[0], "genes:", mat.shape[1])

    rng = np.random.RandomState(0)
    perm = rng.permutation(len(pairs))
    test_idx = set(perm[:int(0.2 * len(pairs))].tolist())
    train_df = pairs[[i not in test_idx for i in range(len(pairs))]]
    test_df = pairs[[i in test_idx for i in range(len(pairs))]]

    def prep(tr, te):
        top, med, mu, sd = select_top_genes(
            mat.loc[sorted(set(tr["SANGER_MODEL_ID"]))])
        return (featurize(tr, mat, top, med, mu, sd, fp, drug_index),
                featurize(te, mat, top, med, mu, sd, fp, drug_index))

    pred, pred_r, y_te = run_experiment(
        ConditionalEncoder(), *prep(train_df, test_df), steps=400)
    print("MAIN split: Neural ODE corr", round(corr(pred, y_te), 3),
          "RMSE", round(rmse(pred, y_te), 3))
    print("MAIN split: Ridge corr", round(corr(pred_r, y_te), 3),
          "RMSE", round(rmse(pred_r, y_te), 3))

    for drug in LODO_DRUGS:
        te_d = pairs[pairs["DRUG_NAME"] == drug]
        tr_d = pairs[pairs["DRUG_NAME"] != drug]
        pred_d, pred_r_d, y_d = run_experiment(
            ConditionalEncoder(), *prep(tr_d, te_d), steps=300)
        print(f"LODO {drug}: Neural ODE corr {corr(pred_d, y_d):.3f} "
              f"RMSE {rmse(pred_d, y_d):.3f} | Ridge corr "
              f"{corr(pred_r_d, y_d):.3f} RMSE {rmse(pred_r_d, y_d):.3f}")


if __name__ == "__main__":
    main()
