# Gao (Novartis PDX screen, 2015) supplementary-tables check: decision criteria

Fixed 3 November 2026, BEFORE opening or inspecting the supplementary
files themselves. Web searches were used only to confirm that catalogues
(PDCM Finder, NCI PDMR, JAX) exist; no Gao supplementary table content
has been inspected. These thresholds decide Phase 3, so that the result
cannot steer them.

Audit rule (added 3 November 2026): when the supplement is first opened,
record the date here, and the file version or hash if available, before
reading any results. The order of events must stay auditable: criteria
first (3 November 2026), opening date second, verdict third.

Date supplement first opened: 3 November 2026. Method: Nature Medicine
article page (DOI 10.1038/nm.3954) accessed the same day; its text
confirms supplementary information exists but does not expose the
supplementary file links, so a live browser check of the supplementary
files was launched 3 November 2026. The browser downloaded Supplementary
Table 1 freely, no login, the same evening (file
41591_2015_BFnm3954_MOESM10_ESM.xlsx, 122,352,817 bytes, SHA-256
c4b9a6903a4d1f76e3ddca4199039776d56bb99970aa5b7abe4f3abd732a0c6d),
and the sheets were inspected directly from that downloaded file on
3 November 2026. Interim fact, from the paper itself, not from the
supplement: the design is one animal per model per treatment (1x1x1).

## Verdict, logged 3 November 2026: all three PASS

- Criterion 1, PASS: the sheet "PCT raw data" holds day-by-day volumes
  (columns Model, Tumor Type, Treatment, Volume (mm3), body weight (g),
  Days Post T0, % TVol Difference, % BW Difference), 67,445 rows,
  4,706 model-treatment series, median 11 time points per series,
  4,699 series with at least 3 time points, including "untreated"
  series. Caveat, stated not hidden: there is no animal-ID column, and
  only 58 rows duplicate a model-treatment-day, so each series is a
  single animal's trajectory with no replicate mice and no error bars.
  The criterion as written is met (volumes over time, not only
  summaries), but the 1x1x1 structure stays a limitation on any model
  fitted to it.
- Criterion 2, PASS: of 4,476 usable treated series (at least 3 time
  points), 3,962 have a matched untreated series for the same model:
  88.5 percent, above the 80 percent bar. 223 of 277 models have an
  untreated series.
- Criterion 3, PASS: 37 single-agent treatments have at least 5 usable
  matched models each (61 treatments including combinations). Largest:
  LEE011 with 217 models, binimetinib 200, BKM120 197, CLR457 188,
  BYL719 184.
- Access route: free, no login, no subscription. Not a blocker.
- Other sheets in the file, for the record: "RNAseq_fpkm" (22,666 rows,
  gene expression per model), "copy number" (23,855 rows), "pdxe_mut_and_cn2"
  (262,092 rows, mutations and copy number), "PCT curve metrics" (4,759
  rows, BestResponse and ResponseCategory summaries per model-treatment).
- Design note, not a criterion: the raw-data sheet carries no dose or
  schedule column, so a stage-2 model on this source has mechanism and
  growth but no dose feature unless doses are recovered from the paper
  or another source. That shapes Phase 3 design; it does not change
  this verdict.

Decision table outcome: Phase 3 proceeds as hierarchical shrinkage on
the bigger set. Simeoni stays parked.

## Criterion 1: time courses, not summaries
- PASS: the supplement gives per-mouse tumour volumes over time, with
  at least 3 time points per mouse, for treated and untreated arms.
- FAIL: it gives only summary responses (best response, tumour growth
  inhibition at one endpoint, response category, or a single end volume).
- If FAIL, Phase 3 is dead. The reframed negative-result paper is the
  deliverable, and no mixed-effects model is attempted on this source.

## Criterion 2: an untreated arm for every model used
- The control-plus-treated stage-1 fit needs a matched untreated arm
  for each model-treatment pair it uses.
- PASS: at least 80 percent of usable model-treatment pairs have a
  matched untreated arm with its own time course.
- FAIL: below 80 percent. The pipeline does not run unchanged, and the
  source cannot be used as a drop-in replacement for the digitised set.

## Criterion 3: enough drugs
- A drug counts only if it has at least 5 models with usable time
  courses (Criterion 1) and a matched untreated arm (Criterion 2).
- PASS: at least 30 such drugs (the stated goal is 30 to 50 drugs).
- FAIL: fewer than 30 such drugs. Phase 3 is dead as a scale-up; the
  source may still serve as a small external check, but not as the
  Phase 3 dataset.

## Also recorded, not a kill criterion on its own
- Access route: freely downloadable, or gated (application, materials
  transfer agreement, commercial licence, institutional login). A gate
  is recorded as a blocker with its expected delay, and the decision
  above is reported separately from access.

## Decision table
- All three PASS: Phase 3 proceeds as hierarchical shrinkage on the
  bigger set. Simeoni stays parked unless the bigger set proves dense
  enough (many points per curve, many mice per arm) to identify it.
- Any FAIL: Phase 3 on this source is dead, and the reframed paper
  stands as the deliverable. The failure and the numbers behind it are
  recorded here, not argued away.
