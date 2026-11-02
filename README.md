# Neural ODE PK/PD, multi-drug branch

Gene expression plus a drug fingerprint goes into a neural encoder. The
encoder outputs PK/PD parameters (clearance, growth, kill), a
differentiable ODE solver turns them into tumour curves, and the
day-30 tumour size is mapped to the GDSC drug-response endpoint.

## What is honest here

- GDSC gives ONE endpoint per cell line (log IC50), not a measured
  time course. The curves are the model's internal predictions.
- The random-pair main split leaks: the same cell lines and drugs
  appear in train and test, so its numbers are inflated.
- The leave-one-drug-out (LODO) folds are the honest metric: a
  held-out drug was never seen in training.
- The erlotinib treated curves are digitised by eye from a published
  figure (see below), approximate to about +/-0.15 cm3, means only.

## Results

Single-drug slice (Erlotinib only, Phase 3): Neural ODE test
correlation 0.404 vs Ridge 0.322. The mechanistic model beats Ridge
on held-out cell lines.

Multi-drug conditional encoder (77 drugs, 67,927 pairs, 945 models):

| Split | Neural ODE corr / RMSE | Ridge corr / RMSE |
|---|---|---|
| Main, random pairs (leaky) | 0.685 / 1.891 | 0.881 / 1.229 |
| LODO Erlotinib | 0.240 / 3.036 | 0.198 / 1.518 |
| LODO Trametinib | -0.168 / 4.257 | 0.153 / 2.893 |
| LODO Cisplatin | 0.132 / 1.148 | 0.513 / 0.988 |

Verdict: on unseen drugs the multi-drug encoder does NOT beat Ridge
yet. Erlotinib is roughly tied on correlation but worse on error,
Trametinib fails outright, Cisplatin is worse on both. Loss plateaued
around 3-4.5 (vs 0.303 in the single-drug phase), suggesting
undertraining at 300 steps per fold and/or a single shared endpoint
head that is too rigid across drug mechanisms.

Time series (real curves, separate notebooks on main):
- Untreated Zenodo curves: LM2-4LUC forecast corr 0.770 (286 held-out
  points), LLC forecast corr 0.352 (91 points). Longer curves forecast
  decently, short fast curves poorly.
- Erlotinib xenograft, digitised: full fit corr 0.978-0.996; kill
  0.0844/day at 4 mg/kg vs 0.1120/day at 50 mg/kg, dose ordering holds;
  forecast corr 0.798-0.989, with the late control surge
  underestimated.

## Files

- `data_loader.py` — GDSC1 + Sanger TPM + PubChem SMILES/fingerprints.
  Top-500 variable genes selected on train models only; row indexing
  by position (never expand duplicated model ids over all genes).
- `ode_engine.py` — one-compartment drug + logistic growth minus kill,
  RK4 via torchdiffeq, single bolus and dosed variants.
- `conditional_encoder.py` — drug MLP (2048-128-64) + encoder
  (564-64-3) with the soft bounded tanh mapping for CL, growth, kill.
- `train_lodo.py` — main split + LODO training with Ridge baselines.
- `erlotinib_digitised_fig2E.csv` — digitised means from Wu et al.,
  Acta Pharmacologica Sinica 2013, Figure 2E (SPC-A-1 xenografts,
  n=5 per group, daily erlotinib, every 48 h). Estimates, not measured
  data.

## Run

    python train_lodo.py

Expects `data/` next to the scripts with
`GDSC1_fitted_dose_response_24Jul22.csv`,
`rnaseq_merged_rsem_tpm_20260323.csv` and `drug_smiles.csv`.

## Data sources

- GDSC1 fitted dose responses, Sanger FTP current release.
- Sanger merged RNA-seq TPM (Sanger + Broad/CCLE), cog.sanger.ac.uk.
- PubChem PUG REST for SMILES; RDKit Morgan fingerprints, radius 2,
  2048 bits. Six internal tool codes had no resolvable structure and
  were dropped from the fingerprint model.
