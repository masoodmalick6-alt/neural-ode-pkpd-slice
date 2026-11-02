"""Data loading for the multi-drug Neural ODE prototype.

Sources (downloaded once, reused from disk when present):
- GDSC1 fitted dose responses (Sanger FTP, current release):
  GDSC1_fitted_dose_response_24Jul22.csv
- Sanger merged RNA-seq TPM (Sanger + Broad/CCLE, Sanger preferred):
  rnaseq_merged_20260323.zip -> rnaseq_merged_rsem_tpm_20260323.csv
  Layout: header row = model_id + SIDM columns, rows 1-3 metadata,
  gene data from row 4. Values are already on a log2 scale.
- Drug SMILES via PubChem (fixed-key lookup: CanonicalSMILES,
  IsomericSMILES, SMILES, ConnectivitySMILES, in that order),
  turned into 2048-bit Morgan fingerprints (radius 2) with RDKit.

Memory rule (learned the hard way): NEVER index the full gene matrix
with a duplicated list of model ids and then select columns. Pandas
expands the duplicates first (54k rows x 41k genes = 16.7 GB) and the
run dies. Select the top genes FIRST, then index rows by position.
"""

import numpy as np
import pandas as pd

GDSC1_URL = ("https://ftp.sanger.ac.uk/pub/project/cancerrxgene/releases/"
              "current_release/GDSC1_fitted_dose_response_24Jul22.csv")
TPM_ZIP_URL = "https://cog.sanger.ac.uk/cmp/download/rnaseq_merged_20260323.zip"


def load_gdsc1(path):
    """GDSC1 fitted responses, one row per drug-model pair."""
    gdsc = pd.read_csv(path, usecols=["SANGER_MODEL_ID", "DRUG_NAME", "LN_IC50"])
    return gdsc.dropna(subset=["LN_IC50"]).drop_duplicates(
        ["DRUG_NAME", "SANGER_MODEL_ID"])


def load_tpm(path):
    """Sanger merged TPM table -> expression DataFrame (genes x models).

    Rows 1-3 are metadata and are skipped; duplicate gene rows are
    dropped, keeping the first.
    """
    tpm = pd.read_csv(path, header=0, low_memory=False)
    tdata = tpm.iloc[3:].copy().set_index(tpm.columns[0])
    model_cols = [c for c in tdata.columns if str(c).startswith("SIDM")]
    expr = tdata[model_cols].apply(pd.to_numeric, errors="coerce")
    return expr[~expr.index.duplicated(keep="first")]


def build_pair_table(gdsc, drugs):
    """Pairs for the fingerprint drugs only, matched to expression models."""
    pairs = gdsc[gdsc["DRUG_NAME"].isin(drugs)]
    return pairs.reset_index(drop=True)


def build_expression_matrix(expr, model_ids):
    """Models x genes matrix for the models present in the pair table."""
    models = sorted(set(model_ids))
    mat = expr[models].T
    return mat  # index = model id, columns = genes


def select_top_genes(mat_train, k=500):
    """Top-k variable genes using TRAIN models only.

    Returns (gene_index, medians, means, sds). Genes missing in more
    than half the train lines are excluded before ranking.
    """
    values = mat_train.values.astype(np.float32)
    nanfrac = np.isnan(values).mean(axis=0)
    var = np.nanvar(values, axis=0)
    var = np.where(nanfrac < 0.5, var, -1)
    top = np.argsort(var)[-k:]
    top_values = values[:, top]
    med = np.nanmedian(top_values, axis=0)
    mu = np.nanmean(top_values, axis=0)
    sd = np.nanstd(top_values, axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    return top, med, mu, sd


def featurize(pairs_df, mat, top, med, mu, sd, fp, drug_index):
    """Turn pair rows into (gene_tensor, fingerprint_tensor, targets).

    Row indexing is by integer position into the pre-selected top-gene
    matrix, so duplicated model ids cost nothing (see module docstring).
    """
    mat_top = mat.values[:, top].astype(np.float32)
    row_of = {m: i for i, m in enumerate(mat.index)}
    rows = np.array([row_of[m] for m in pairs_df["SANGER_MODEL_ID"]])
    genes = mat_top[rows]
    genes = np.where(np.isnan(genes), med, genes)
    genes = (genes - mu) / sd
    fps = fp[[drug_index[d] for d in pairs_df["DRUG_NAME"]]]
    targets = pairs_df["LN_IC50"].values.astype(np.float32)
    return genes, fps, targets


def smiles_to_fingerprints(smiles_df):
    """SMILES table (DRUG_NAME, SMILES) -> (drug list, fingerprint array)."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    fp_map = {}
    for _, row in smiles_df.iterrows():
        mol = Chem.MolFromSmiles(row["SMILES"])
        if mol is not None:
            fp_map[row["DRUG_NAME"]] = np.array(
                AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048),
                dtype=np.float32)
    drugs = sorted(fp_map)
    fp = np.stack([fp_map[d] for d in drugs])
    return drugs, fp, {d: i for i, d in enumerate(drugs)}
