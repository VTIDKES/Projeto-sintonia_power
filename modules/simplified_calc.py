"""Power System Studio - Simplified calculations.

Implements spreadsheet-like approximations to compare against full AC Newton-Raphson.

Included methods:
1) DC power flow (lossless, |V|=1 p.u.)
   - Uses susceptance matrix built from line reactances (x_ohm_per_km).
   - Works best on transmission-like networks; for distribution it is a rough guide.
2) Voltage drop estimate (radial-like, per-branch)
   - Approximate dV (p.u.) ~ (R*P + X*Q)/V for each branch.
3) Line loss estimate
   - I ~ S/(sqrt(3)*V_ll)
   - P_loss = 3 * I^2 * R_total

All outputs are returned as pandas DataFrames / dicts.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


def _build_bus_index(buses: pd.DataFrame) -> Dict[str, int]:
    names = buses['name'].astype(str).tolist()
    return {n: i for i, n in enumerate(names)}


def dc_power_flow(
    buses: pd.DataFrame,
    gens: pd.DataFrame,
    loads: pd.DataFrame,
    lines: pd.DataFrame,
    slack_bus_name: str | None = None,
) -> Dict[str, Any]:
    """Lossless DC power flow.

    Returns:
    - theta_rad per bus (slack = 0)
    - p_inj_mw per bus (generation - load)
    - p_from_to_mw per line (approx)

    Assumptions:
    - Voltage magnitudes are 1.0 p.u.
    - R ignored; only X used.
    """

    if buses.empty:
        return {"error": "No buses"}

    bus_idx = _build_bus_index(buses)
    n = len(bus_idx)

    # Determine slack
    if slack_bus_name is None:
        # Prefer explicit Slack type
        slack_candidates = buses[buses['type'].astype(str).str.upper() == 'SLACK']['name'].astype(str).tolist()
        slack_bus_name = slack_candidates[0] if slack_candidates else buses.iloc[0]['name']

    if slack_bus_name not in bus_idx:
        return {"error": f"Slack bus '{slack_bus_name}' not found"}

    slack = bus_idx[slack_bus_name]

    # Net injections (MW): gen P minus load P
    p_gen = np.zeros(n)
    if not gens.empty:
        for _, row in gens.iterrows():
            b = str(row.get('bus', ''))
            if b in bus_idx:
                p_gen[bus_idx[b]] += float(row.get('p_mw', 0.0) or 0.0)

    p_load = np.zeros(n)
    if not loads.empty:
        for _, row in loads.iterrows():
            b = str(row.get('bus', ''))
            if b in bus_idx:
                p_load[bus_idx[b]] += float(row.get('p_mw', 0.0) or 0.0)

    p_inj = p_gen - p_load

    # Build B' (susceptance matrix) from line X
    B = np.zeros((n, n), dtype=float)
    used_lines = 0
    for _, row in lines.iterrows():
        fb = str(row.get('from_bus', ''))
        tb = str(row.get('to_bus', ''))
        if fb not in bus_idx or tb not in bus_idx:
            continue
        i, j = bus_idx[fb], bus_idx[tb]
        length_km = float(row.get('length_km', 0.0) or 0.0)
        x = float(row.get('x_ohm_per_km', 0.0) or 0.0)
        if x == 0.0 or length_km == 0.0:
            continue
        x_total = x * length_km
        b_ij = -1.0 / x_total
        B[i, j] += b_ij
        B[j, i] += b_ij
        B[i, i] -= b_ij
        B[j, j] -= b_ij
        used_lines += 1

    if used_lines == 0:
        return {"error": "No valid lines with non-zero reactance for DC flow"}

    # Solve for theta excluding slack
    mask = np.ones(n, dtype=bool)
    mask[slack] = False
    B_red = B[mask][:, mask]
    p_red = p_inj[mask]

    # If singular (islands), will throw
    try:
        theta_red = np.linalg.solve(B_red, p_red)
    except Exception as e:
        return {"error": f"DC solve failed (check connectivity / islands): {e}"}

    theta = np.zeros(n)
    theta[mask] = theta_red
    theta[slack] = 0.0

    # Line flows: P_ij = (theta_i - theta_j) / X_total
    line_rows = []
    for _, row in lines.iterrows():
        fb = str(row.get('from_bus', ''))
        tb = str(row.get('to_bus', ''))
        if fb not in bus_idx or tb not in bus_idx:
            continue
        i, j = bus_idx[fb], bus_idx[tb]
        length_km = float(row.get('length_km', 0.0) or 0.0)
        x = float(row.get('x_ohm_per_km', 0.0) or 0.0)
        if x == 0.0 or length_km == 0.0:
            continue
        x_total = x * length_km
        p_ij = (theta[i] - theta[j]) / x_total
        line_rows.append({
            "name": row.get('name', ''),
            "from_bus": fb,
            "to_bus": tb,
            "p_from_to_mw": float(p_ij),
        })

    bus_rows = []
    for name, idx in bus_idx.items():
        bus_rows.append({
            "bus": name,
            "theta_deg": float(theta[idx] * 180.0 / np.pi),
            "p_inj_mw": float(p_inj[idx]),
        })

    return {
        "slack_bus": slack_bus_name,
        "bus_results": pd.DataFrame(bus_rows),
        "line_results": pd.DataFrame(line_rows),
    }


def voltage_drop_and_losses(
    buses: pd.DataFrame,
    loads: pd.DataFrame,
    lines: pd.DataFrame,
    base_v_pu: float = 1.0,
) -> Dict[str, Any]:
    """Very simplified per-line voltage drop and losses estimation.

    This is NOT a full load-flow; it's intended to mimic spreadsheet estimates.

    Approach:
    - For each line, take the downstream load as the load at the 'to_bus'.
      (This is a crude assumption that works best for radial networks.)
    - Compute per-phase current magnitude from S and V_ll at from_bus.
    - Compute losses: P_loss = 3*I^2*R_total
    - Estimate voltage drop (p.u.): dV ~= (R*P + X*Q) / (V^2)

    Returns:
    - line_results DataFrame with approximate drops and losses
    - totals
    """

    if buses.empty or lines.empty:
        return {"error": "Need buses and lines"}

    bus_v_kv = {str(r['name']): float(r.get('vn_kv', 0.0) or 0.0) for _, r in buses.iterrows()}

    # Aggregate loads per bus
    p_bus = {}
    q_bus = {}
    for _, row in loads.iterrows():
        b = str(row.get('bus', ''))
        p_bus[b] = p_bus.get(b, 0.0) + float(row.get('p_mw', 0.0) or 0.0)
        q_bus[b] = q_bus.get(b, 0.0) + float(row.get('q_mvar', 0.0) or 0.0)

    rows = []
    total_ploss = 0.0
    total_qloss = 0.0

    for _, row in lines.iterrows():
        fb = str(row.get('from_bus', ''))
        tb = str(row.get('to_bus', ''))
        length_km = float(row.get('length_km', 0.0) or 0.0)
        r = float(row.get('r_ohm_per_km', 0.0) or 0.0)
        x = float(row.get('x_ohm_per_km', 0.0) or 0.0)

        if length_km <= 0:
            continue

        R = r * length_km
        X = x * length_km

        P = float(p_bus.get(tb, 0.0))
        Q = float(q_bus.get(tb, 0.0))

        V_kv = bus_v_kv.get(fb, 0.0)
        V_ll = V_kv * 1e3
        if V_ll <= 0:
            continue

        S_mva = np.sqrt(P**2 + Q**2)
        I = (S_mva * 1e6) / (np.sqrt(3) * V_ll)  # A

        Ploss = 3.0 * (I**2) * R / 1e6  # MW
        # Rough reactive loss due to series reactance: Qloss = 3 I^2 X
        Qloss = 3.0 * (I**2) * X / 1e6  # MVAr

        # Voltage drop estimate in pu using MW/MVAr and ohms on 3ph base:
        # Convert P,Q to W/VAr, and use V_phase ~ V_ll/sqrt(3)
        V_phase = V_ll / np.sqrt(3)
        dV_phase = (R * (P * 1e6) + X * (Q * 1e6)) / (V_phase**2)
        dV_pu = dV_phase / base_v_pu

        total_ploss += Ploss
        total_qloss += Qloss

        rows.append({
            "name": row.get('name', ''),
            "from_bus": fb,
            "to_bus": tb,
            "P_to_mw": P,
            "Q_to_mvar": Q,
            "I_A": float(I),
            "dV_pu": float(dV_pu),
            "P_loss_mw": float(Ploss),
            "Q_loss_mvar": float(Qloss),
        })

    return {
        "line_results": pd.DataFrame(rows),
        "total_losses_mw": float(total_ploss),
        "total_losses_mvar": float(total_qloss),
    }
PY