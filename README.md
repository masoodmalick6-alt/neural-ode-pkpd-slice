# Neural ODE PK/PD Slice

A small, honest slice of a bigger idea: predict drug sensitivity from gene
expression with a neural network that outputs pharmacokinetic/pharmacodynamic
(PK/PD) parameters, which then run through a differentiable ordinary
differential equation (ODE) to produce a tumour response curve.

Built by Masood Ur Rehman, MSc Drug Discovery and Development student at
Uppsala University, Pharmacist (PharmD) working on AI for drug discovery.

## The idea in one paragraph

Gene expression goes into a small neural encoder. The encoder outputs PK/PD
parameters (clearance and tumour growth rate). Those parameters drive a
one-compartment drug model plus logistic tumour growth with drug kill, solved
with a differentiable ODE solver. On fake patients the encoder recovers hidden
parameters almost exactly. On real GDSC data it predicts Erlotinib sensitivity
better than a Ridge baseline on held-out cell lines.

## Notebooks

- `phase0_setup.ipynb` — install torchdiffeq and check the setup.
- `phase1_ode.ipynb` — the ODE alone, no neural net: concentration and tumour
  curves, change clearance by hand.
- `phase2_encoder.ipynb` — fake patients: the encoder recovers hidden
  clearance and growth from genes alone.
- `phase3_real_data.ipynb` — real data: GDSC Erlotinib response plus Sanger
  merged RNA-seq expression.

## Results

- Phase 2 (fake patients, 200 lines): guessed vs true clearance correlation
  0.995, growth 0.995.
- Phase 3 (real data, 392 cell lines, 79 held out): Neural ODE test
  correlation 0.404, Ridge baseline test correlation 0.322. Train
  correlation 0.875.

## Honest scope

- GDSC gives one endpoint number per cell line (log IC50), not a measured
  time course. The ODE curves in this slice are the model's internal
  predictions, not measured curves.
- The ODE runs at one fixed teaching dose; what varies per cell line is
  clearance and growth from expression.
- This is a prototype and teaching project, not a validated clinical tool
  and not a treatment predictor.
- Data files are not in this repo (too big). The Phase 3 notebook downloads
  them itself:
  - GDSC1 fitted dose response (Sanger FTP)
  - Sanger merged RNA-seq (Cell Model Passports)
- Expression source note: the DepMap portal is bot-walled and GDSC's own
  basal-expression file is retired, so Phase 3 uses Sanger's merged RNA-seq
  file, which combines Sanger and Broad (CCLE) data with Sanger preferred
  where a model was analysed twice, matched to GDSC by model ID.

## Run it

Colab (CPU is fine) or a local Python environment with torch, torchdiffeq,
pandas, scikit-learn and matplotlib. Phase 0 checks the setup; each notebook
runs top to bottom.

## Next steps

Multi-drug integration and fitting to true time-series data (small
literature xenograft sets) are the planned research-prototype steps.
