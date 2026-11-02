# Research prototype: multi-omics-informed QSP/PK-PD Neural ODE

## Question

Can gene expression predict how a tumour responds to a drug, through a
mechanistic model rather than a black box? The idea: an encoder reads a
cell line's expression profile plus a drug's chemical fingerprint and
outputs PK/PD parameters (clearance, growth, kill). A differential
equation turns those parameters into a tumour curve. The endpoint of
the curve is compared against the measured drug response.

## Data

- GDSC1 fitted dose responses: 378 drugs, 970 models, 333,161 rows.
  Endpoint per drug-model pair: log IC50.
- Sanger merged RNA-seq TPM: 41,144 genes, 1,898 models. DepMap's
  portal is bot-walled and GDSC's own expression file is retired, so
  expression comes from Sanger's merged table instead.
- Drug structures: PubChem lookups for 83 candidate drugs resolved 77;
  six internal tool codes had no public structure and were dropped.
  Fingerprints are 2048-bit Morgan, radius 2.
- Time series: Zenodo untreated growth curves (LM2-4LUC breast, 66
  animals; Lewis Lung, 20 animals) and digitised erlotinib xenograft
  means (Wu et al. 2013, Figure 2E; 3 groups, 9 timepoints each).

## Model

One-compartment drug, logistic growth, kill proportional to drug
concentration times tumour size. Single-drug phase: encoder over 500
top variable genes. Multi-drug phase: gene vector plus a 64-dim drug
embedding from fingerprints, one shared endpoint head
(a + b * log(T30/T0)). Parameters use a soft bounded mapping
(CL = 7.5*exp(tanh(z)) and analogues) after a hard clamp killed
gradients and a free exponential went NaN in the single-drug phase.

## Experiments and results

1. Toy ODE (Phase 1): dose 500, CL 7.5 gives tumour 10 down to 4.27
   then back to 55.9 at day 30; CL 3.0 gives 3.3. The machinery behaves.
2. Fake-gene encoder (Phase 2): recovers a hidden rule almost
   perfectly (correlation 0.995). The encoder can learn when signal
   exists.
3. Single drug, real data (Phase 3): Erlotinib, 392 lines. Neural ODE
   test correlation 0.404 vs Ridge 0.322. The mechanistic model wins
   on held-out lines.
4. Multi-drug, real data (Day 2): 77 drugs, 67,927 pairs. Random-pair
   split (leaky): 0.685 vs Ridge 0.881. Leave-one-drug-out, the honest
   test: Erlotinib 0.240 vs 0.198, Trametinib -0.168 vs 0.153,
   Cisplatin 0.132 vs 0.513. The multi-drug model does NOT beat Ridge
   on unseen drugs yet.
5. Time series, untreated: fitting early points and forecasting later
   ones gives correlation 0.770 on the longer breast curves and 0.352
   on the short lung curves.
6. Time series, treated (digitised): the kill term reproduces the
   dose ordering (0.0844/day at 4 mg/kg, 0.1120/day at 50 mg/kg) with
   full-fit correlations 0.978-0.996, and forecasts at 0.798-0.989,
   underestimating the late control surge.

## Interpretation

The single-drug result shows the approach can work: expression carries
signal about drug response that a mechanistic model can use. The
multi-drug result shows the hard part: generalising to a drug the
model never saw. Fingerprints alone carry little of what makes a drug
behave differently in a new cell line, and one shared endpoint head
forces very different mechanisms through the same mapping. The
time-series results show the ODE itself is sound on real growth
curves, and that drug effect over time can be recovered when treated
curves exist.

## Limitations (stated, not buried)

- GDSC endpoints are single numbers, not curves; the model's curves
  are internal predictions, not observations.
- LODO is the honest metric and it currently loses to Ridge.
- Treated time-series values are eye-digitised estimates, means only,
  27 points, and daily dosing was approximated as a constant kill
  rate rather than full mouse pharmacokinetics.
- Training was short (300 steps per LODO fold) on CPU; the loss had
  plateaued but not converged.
- Six drugs were dropped for lack of a public structure.

## Next steps

- Longer training and a per-drug or per-mechanism endpoint head.
- A richer drug representation (beyond fingerprints) or drug-class
  conditioning.
- Full mouse PK for the treated fits instead of constant kill.
- More treated time-series datasets, ideally with published tables
  rather than digitised figures.
- A held-out drug-class test, harder than leave-one-drug-out.
