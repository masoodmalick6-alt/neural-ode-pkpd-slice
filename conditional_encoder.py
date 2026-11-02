"""Conditional encoder: gene expression + drug fingerprint -> PK/PD params.

Two branches:
- Drug branch: 2048-bit Morgan fingerprint -> MLP 2048 -> 128 -> 64.
- Encoder: 500 top genes + 64-dim drug embedding -> 64 -> 3 outputs.

Outputs use the SOFT bounded mapping from Phase 3 (a hard clamp killed
gradients, a free exp went NaN):

    CL     = 7.5  * exp(tanh(z0))   (range 2.76 - 20.4)
    growth = 0.2  * exp(tanh(z1))   (range 0.074 - 0.544)
    kill   = 0.05 * exp(tanh(z2))   (range 0.018 - 0.136)

The endpoint head maps the day-30 tumour size to the GDSC endpoint:

    predicted LN_IC50 = a + b * log(T30 / T0)

with a and b learned. Note the honest label: GDSC gives ONE endpoint
per cell line, not a measured time course, so the curves are the
model's internal predictions, not observed trajectories.
"""

import torch


class DrugMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2048, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, 64))

    def forward(self, fp):
        return self.net(fp)


class ConditionalEncoder(torch.nn.Module):
    def __init__(self, n_genes=500):
        super().__init__()
        self.drug_mlp = DrugMLP()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(n_genes + 64, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 3))
        self.a = torch.nn.Parameter(torch.tensor(2.0))
        self.b = torch.nn.Parameter(torch.tensor(-1.0))

    def parameters_for(self, genes, fp):
        """Genes + fingerprints -> (CL, growth, kill) tensors."""
        z = self.encoder(torch.cat([genes, self.drug_mlp(fp)], dim=1))
        CL = 7.5 * torch.exp(torch.tanh(z[:, 0]))
        growth = 0.2 * torch.exp(torch.tanh(z[:, 1]))
        kill = 0.05 * torch.exp(torch.tanh(z[:, 2]))
        return CL, growth, kill

    def predict_ln_ic50(self, genes, fp, T30, T0=10.0):
        return self.a + self.b * torch.log(T30 / T0)
