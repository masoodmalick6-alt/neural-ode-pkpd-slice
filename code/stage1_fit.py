"""Stage 1: fit a logistic tumour-growth ODE per curve and estimate kill/day.

Model, identical to the paper, Section 2.1:
    dV/dt = g * V * (1 - V / K) - kill * V
Growth g, carrying capacity K and starting volume T0 are shared across the
arms of one fit. Kill is 0 for the control arm and one fitted value per
treated arm. The fit minimises mean squared error in log space, using Adam
(lr 0.03, 800 steps) with parameters in log space, RK4 with step size 0.1.

Two-phase mode: the treated arm uses kill1 before a switch time and kill2
after it. The reported single kill is the duration-weighted mean
(switch * kill1 + (end - switch) * kill2) / end, where end is the last
treated observation time in the fit window.

Arms may cover different days: no arm is invented past its own data. The
loss is pooled over all observed points of all arms.

Usage:
    python stage1_fit.py --spec curves.json --fit sorafenib_0606
    python stage1_fit.py --spec curves.json --all
"""

import argparse
import json
import math
import sys

import numpy as np
import pandas as pd
import torch
from torchdiffeq import odeint


def load_arms(spec, base_dir):
    """Return [(arm_name, times, values, is_control), ...] for one fit."""
    path = f"{base_dir}/{spec['file']}"
    df = pd.read_csv(path)
    time_col = spec["time_col"]
    shift = spec.get("time_shift", 0)
    tmin = spec.get("day_min")
    tmax = spec.get("day_max")
    arms = []
    if spec["format"] == "long":
        gcol, vcol = spec["group_col"], spec["value_col"]
        filt = spec.get("filter", {})
        sub = df
        for col, val in filt.items():
            sub = sub[sub[col] == val]
        for name, label in spec["arms"].items():
            rows = sub[sub[gcol] == label]
            t = rows[time_col].to_numpy(dtype=float) - shift
            v = rows[vcol].to_numpy(dtype=float)
            keep = np.isfinite(t) & np.isfinite(v) & (v > 0)
            if tmin is not None:
                keep &= t >= tmin
            if tmax is not None:
                keep &= t <= tmax
            arms.append((name, t[keep], v[keep], name == "control"))
    elif spec["format"] == "wide":
        for name, col in spec["arms"].items():
            rows = df[np.isfinite(df[col].to_numpy(dtype=float))]
            t = rows[time_col].to_numpy(dtype=float) - shift
            v = rows[col].to_numpy(dtype=float)
            keep = (v > 0)
            if tmin is not None:
                keep &= t >= tmin
            if tmax is not None:
                keep &= t <= tmax
            arms.append((name, t[keep], v[keep], name == "control"))
    else:
        raise ValueError(f"unknown format {spec['format']}")
    for name, t, v, _ in arms:
        if len(t) < 2:
            raise ValueError(f"arm {name} in {spec['file']} has fewer than 2 points")
    return arms


def simulate(t, g, K, T0, kill1, kill2=None, switch=None):
    def ode(tt, y):
        if switch is None:
            k = kill1
        else:
            k = torch.where(tt < switch, kill1, kill2)
        return g * y * (1 - y / K) - k * y

    y0 = T0.reshape(1) if torch.is_tensor(T0) else torch.tensor([T0])
    sol = odeint(ode, y0, t, method="rk4", options={"step_size": 0.1})
    return sol[:, 0]


def fit_arms(arms, two_phase=None, steps=800, lr=0.03, seed=0):
    """Fit shared g/K/T0 and one kill per treated arm. Returns a dict."""
    torch.manual_seed(seed)
    max_v = max(float(np.max(v)) for _, _, v, _ in arms)
    first_c = next(v[0] for _, _, v, c in arms if c)
    lg = torch.nn.Parameter(torch.tensor(math.log(0.15)))
    lK = torch.nn.Parameter(torch.tensor(math.log(5.0 * max_v)))
    lT = torch.nn.Parameter(torch.tensor(math.log(first_c)))
    treated = [(name, t, v) for name, t, v, c in arms if not c]
    lk = {name: torch.nn.Parameter(torch.tensor(math.log(0.05)))
          for name, _, _ in treated}
    lk2 = {}
    if two_phase is not None:
        lk2 = {name: torch.nn.Parameter(torch.tensor(math.log(0.05)))
               for name, _, _ in treated}
    params = [lg, lK, lT] + list(lk.values()) + list(lk2.values())
    opt = torch.optim.Adam(params, lr=lr)
    t_tensors = {name: torch.tensor(t) for name, t, _, _ in arms}
    v_tensors = {name: torch.tensor(v) for name, _, v, _ in arms}
    switch = two_phase["switch"] if two_phase else None
    for _ in range(steps):
        g, K, T0 = torch.exp(lg), torch.exp(lK), torch.exp(lT)
        loss = torch.tensor(0.0)
        npts = 0
        for name, t, v, is_c in arms:
            if is_c:
                pred = simulate(t_tensors[name], g, K, T0, torch.tensor(0.0))
            elif two_phase is None:
                pred = simulate(t_tensors[name], g, K, T0, torch.exp(lk[name]))
            else:
                pred = simulate(t_tensors[name], g, K, T0,
                                torch.exp(lk[name]), torch.exp(lk2[name]), switch)
            loss = loss + torch.sum((torch.log(pred) - torch.log(v_tensors[name])) ** 2)
            npts += len(t)
        loss = loss / npts
        opt.zero_grad()
        loss.backward()
        opt.step()
    g, K, T0 = float(torch.exp(lg)), float(torch.exp(lK)), float(torch.exp(lT))
    out = {"g": g, "K": K, "T0": T0, "arms": {}}
    for name, t, v, is_c in arms:
        if is_c:
            pred = simulate(t_tensors[name], torch.tensor(g), torch.tensor(K),
                            torch.tensor(T0), torch.tensor(0.0)).detach().numpy()
            out["arms"][name] = {"kill": 0.0, "log_rmse": float(np.sqrt(np.mean((np.log(pred) - np.log(v)) ** 2)))}
        elif two_phase is None:
            k = float(torch.exp(lk[name]))
            pred = simulate(t_tensors[name], torch.tensor(g), torch.tensor(K),
                            torch.tensor(T0), torch.tensor(k)).detach().numpy()
            out["arms"][name] = {"kill": k, "log_rmse": float(np.sqrt(np.mean((np.log(pred) - np.log(v)) ** 2)))}
        else:
            k1, k2 = float(torch.exp(lk[name])), float(torch.exp(lk2[name]))
            end = float(np.max(t))
            kavg = (switch * k1 + (end - switch) * k2) / end
            pred = simulate(t_tensors[name], torch.tensor(g), torch.tensor(K),
                            torch.tensor(T0), torch.tensor(k1), torch.tensor(k2),
                            switch).detach().numpy()
            out["arms"][name] = {"kill1": k1, "kill2": k2, "kill_avg": kavg,
                                 "log_rmse": float(np.sqrt(np.mean((np.log(pred) - np.log(v)) ** 2)))}
    return out


def row_kill(fit_result, spec):
    """Combine arm kills into the row kill exactly as the paper does."""
    mode = spec.get("row_kill", "single")
    if mode == "single":
        arm = spec["row_arm"]
        a = fit_result["arms"][arm]
        return a.get("kill_avg", a["kill"])
    if mode == "mean":
        vals = []
        for arm in spec["row_arms"]:
            a = fit_result["arms"][arm]
            vals.append(a.get("kill_avg", a["kill"]))
        return float(np.mean(vals))
    raise ValueError(f"unknown row_kill mode {mode}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", required=True, help="curves.json path")
    ap.add_argument("--fit", help="one fit id")
    ap.add_argument("--all", action="store_true", help="fit every entry")
    ap.add_argument("--base", default=".", help="directory holding the data files")
    args = ap.parse_args()
    with open(args.spec) as fh:
        specs = json.load(fh)
    ids = list(specs) if args.all else [args.fit]
    for fid in ids:
        spec = specs[fid]
        arms = load_arms(spec, args.base)
        res = fit_arms(arms, two_phase=spec.get("two_phase"))
        scale = 7.0 if spec.get("time_unit") == "week" else 1.0
        rk = row_kill(res, spec) / scale
        print(f"{fid}: g={res['g'] / scale:.4f}/day K={res['K']:.1f} "
              f"T0={res['T0']:.3f} row_kill={rk:.4f}/day")
        for name, a in res["arms"].items():
            print(f"    {name}: {a}")


if __name__ == "__main__":
    main()
