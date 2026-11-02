"""ODE engine for the Neural ODE prototype.

One-compartment drug plus logistic tumour growth minus kill:

    dC/dt = -ke * C,  ke = CL / V
    dT/dt = growth * T * (1 - T / K) - kill * C * T

A single bolus sets C(0) = Dose / V. `simulate_with_doses` additionally
handles a list of (time, amount) dose events by stepping between events
and adding amount / V to C at each event time. Integration uses RK4
from torchdiffeq, matching the notebooks. Times are in days.
"""

import torch
from torchdiffeq import odeint


def _rhs(CL, V, growth, kill, K):
    ke = CL / V

    def ode(t, y):
        C, T = y[:, 0], y[:, 1]
        return torch.stack([-ke * C,
                            growth * T * (1 - T / K) - kill * C * T], dim=1)

    return ode


def simulate(CL, growth, kill, dose=500.0, V=50.0, K=100.0, T0=10.0,
             days=30, step=1.0):
    """Single bolus at t=0. Returns the full trajectory (time, batch, 2)."""
    t = torch.linspace(0, days, int(days / step) + 1)
    y0 = torch.stack([torch.full_like(CL, dose / V),
                      torch.full_like(CL, T0)], dim=1)
    return odeint(_rhs(CL, V, growth, kill, K), y0, t,
                  method="rk4", options={"step_size": step})


def simulate_with_doses(CL, growth, kill, doses, V=50.0, K=100.0, T0=10.0,
                        t_end=16.0, step=0.5):
    """Doses = list of (day, amount). State carries across dose events.

    Used for the treated time-series fits (daily dosing approximated by
    explicit dose events; see WRITEUP.md for the constant-kill variant
    used when only digitised means are available).
    """
    doses = sorted(doses)
    t = torch.linspace(0, t_end, int(t_end / step) + 1)
    y = torch.stack([torch.full_like(CL, 0.0),
                     torch.full_like(CL, T0)], dim=1)
    # Doses at time 0 apply before integration starts.
    for day, amount in doses:
        if abs(day) < 1e-9:
            y = torch.stack([y[:, 0] + amount / V, y[:, 1]], dim=1)
    out = [y]
    rhs = _rhs(CL, V, growth, kill, K)
    dose_idx = sum(1 for day, _ in doses if abs(day) < 1e-9)
    for i in range(1, len(t)):
        seg = torch.linspace(t[i - 1], t[i], 2)
        y = odeint(rhs, y, seg, method="rk4",
                   options={"step_size": step})[-1]
        while dose_idx < len(doses) and abs(doses[dose_idx][0] - float(t[i])) < 1e-9:
            y = torch.stack([y[:, 0] + doses[dose_idx][1] / V, y[:, 1]], dim=1)
            dose_idx += 1
        out.append(y)
    return torch.stack(out)
