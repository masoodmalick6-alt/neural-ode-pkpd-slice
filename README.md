# Why tumour drug kill is hard to predict: a 30-row case study

## Read this first: the limitations

This is a negative result, and it is meant to be read as one.

- **30 rows, 16 drugs.** Small, and leave-one-drug-out on 16 drugs has
  wide uncertainty of its own.
- **Digitised by eye, by one person.** Every curve was read from a
  published figure. Errors are stated per row in `data/rows30.csv`
  (about 0.05 in relative volume, 15 to 150 mm3 in absolute volume
  depending on axis scale, about 30 percent on one old log-scale axis).
  The values are digitised estimates, not measured data.
- **No external validation.** All tests are cross-validation inside one
  small, hand-built collection, assembled by the same person who fitted
  it. That is a bias risk, stated here rather than hidden.
- **Dose is not exposure.** Dose intensity in mg/kg/day is an
  administered amount normalised by schedule. Exposure would be plasma
  concentration over time, free fraction and tumour concentration, with
  a potency number matched to the line. The source papers give dose and
  schedule only, so exposure was not modelled at all. Any exposure
  analysis on these rows would have to invent pharmacokinetics.
- **The headline model barely beats the mean.** On the 30 rows, the
  Huber headline model gives drug-out kill error 0.0716 per day against
  a global-mean baseline of 0.0831 and a class-mean baseline of 0.0846.
  The paired interval excludes zero, so the win over the mean is small
  but real on these rows, subject to a selection caveat: Huber was
  chosen after seeing the results, so it is the best of the models
  tried on this set, not a pre-specified headline.
- **Two rows carry 81 percent of the squared error**
  (cyclophosphamide in MX-1 and sorafenib in HCC 06-0606). Without both
  rows the error falls to 0.0324 per day, still only modestly ahead of
  the mean baseline of 0.0442.

Do not use this as a predictor for a new drug. Use it as a benchmark,
a reusable table, and a record of what failed and why.

## What is in here

- `data/rows30.csv` — the 30-row table: drug, line, fitted kill and
  growth, dose, mechanism, source paper, figure or table, digitisation
  error, digitised file, fit note, and the leave-one-drug-out
  predictions of the headline model and both baselines. This is the
  most reusable part.
- `data/digitised/` — the digitised points behind each row, one file
  per source curve, with the reference for each in `rows30.csv`.
  No published figures are included: the numbers are shared with
  citations, the figures themselves are not copied.
- `data/rejected_figures.md` — the figures that failed the filter and
  why, so the filter is auditable.
- `code/stage1_fit.py` and `code/curves.json` — the stage 1 fitting
  code (logistic ODE, shared growth, kill per treated arm, two-phase
  mode where the curve dips then regrows) and the exact curve
  specification for every row. Run
  `python code/stage1_fit.py --spec code/curves.json --all` from the
  repository root to refit all rows. Full refits take a while; each
  fit is 800 Adam steps.
- **Stage 1 refit verification, in progress, not yet passed.** Re-running
  `code/stage1_fit.py` from `code/curves.json` does not yet reproduce
  every paper kill bit-exactly. As of 3 Nov 2026, 8 of 30 rows have been
  checked: 5 match exactly, 2 mismatch (erlotinib SPC-A-1: refit 0.0811
  vs paper 0.0982; HNSCC cisplatin/5-FU second row: 0.0530 vs 0.0609),
  and 2 fail on script key errors (cisplatin HNO97 two-phase,
  gemcitabine SW1990), not on paper values. The paper kills remain the
  notebook values. Do not claim stage 1 bit-exact reproduction until
  the full check lands and any mismatches are fixed or documented.
- `code/reproduce_tables.py` — regenerates Tables 1, 3 and 4 from
  `data/rows30.csv`, including the drug-cluster bootstrap and the
  paired test. Table 1 model values are historical run outputs and are
  printed as recorded; the script recomputes and checks the baseline
  method and fully recomputes Tables 3 and 4. Exit code is non-zero
  if a recomputed headline number disagrees with the paper beyond
  rounding.
- `gao_check_criteria.md` — the pre-registered thresholds for the
  Novartis PDX supplementary check, committed with its date before the
  supplement was opened.

## Why a logistic kill, and not a Simeoni-type model

A Simeoni transit-compartment model needs drug concentration over time
as an input, and these figures contain no concentration data. Inventing
a pharmacokinetic model for each of 16 old drugs, across species and
routes, would be fabrication dressed as modelling. The logistic kill is
what volume curves alone can identify. Simeoni is parked until a
dataset with concentration data exists, not rejected as an idea. The
domain fix this work did not try is a dataset designed around exposure
from the start: plasma concentration, free fraction and line-matched
potency collected up front, judged by the same leave-one-drug-out test
with the same baselines.

## Licences

- Code in `code/` is under the MIT licence (`LICENSE`).
- Data in `data/` (the digitised numbers and the 30-row table) is
  under CC BY 4.0 (`LICENSE-DATA`). The numbers are derived from
  published figures; cite the source papers listed in `rows30.csv`
  alongside this repository.
- No published figures, no author manuscripts and no supplementary
  files from the source papers are included.

## Reproducibility

Install from `requirements.txt`, then run
`python code/reproduce_tables.py` from the repository root. It checks
itself against the paper and fails loudly if a headline number moves.
