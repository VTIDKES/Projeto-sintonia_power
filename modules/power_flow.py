"""Power System Studio - Power flow runner.

Wraps pandapower power flow execution (Newton-Raphson) and extracts results
into plain python/pandas structures suitable for Streamlit display and export.

Notes:
- pandapower uses Newton-Raphson by default; we explicitly set algorithm='nr'.
- Per-iteration mismatch history is not always exposed; we try to retrieve
  whatever pandapower stored in net._ppc['internal'].
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd
import pandapower as pp


def run_newton_raphson(net, tol: float = 1e-8, max_iter: int = 20, init: str = "auto") -> Dict[str, Any]:
    """Run pandapower Newton-Raphson power flow.

    Returns a dict with:
    - converged (bool)
    - iterations (int or None)
    - bus_results (DataFrame)
    - line_results (DataFrame)
    - trafo_results (DataFrame)
    - total_losses_mw (float)
    - total_losses_mvar (float)
    - convergence_trace (optional DataFrame)
    """

    # Clear old results if any
    try:
        net.res_bus.drop(net.res_bus.index, inplace=True)
    except Exception:
        pass

    converged = False
    err_msg = ""
    try:
        pp.runpp(
            net,
            algorithm="nr",
            tolerance_mva=tol,
            max_iteration=max_iter,
            init=init,
            enforce_q_lims=True,
            calculate_voltage_angles=True,
        )
        converged = bool(getattr(net, "converged", False))
    except Exception as e:
        converged = False
        err_msg = str(e)

    # Extract results (if converged, res_* are filled; if not, they may be empty)
    bus_results = getattr(net, "res_bus", pd.DataFrame()).copy()
    line_results = getattr(net, "res_line", pd.DataFrame()).copy()
    trafo_results = getattr(net, "res_trafo", pd.DataFrame()).copy()

    total_losses_mw = 0.0
    total_losses_mvar = 0.0
    if not line_results.empty:
        if "pl_mw" in line_results.columns:
            total_losses_mw += float(line_results["pl_mw"].sum())
        if "ql_mvar" in line_results.columns:
            total_losses_mvar += float(line_results["ql_mvar"].sum())
    if not trafo_results.empty:
        if "pl_mw" in trafo_results.columns:
            total_losses_mw += float(trafo_results["pl_mw"].sum())
        if "ql_mvar" in trafo_results.columns:
            total_losses_mvar += float(trafo_results["ql_mvar"].sum())

    iterations = None
    trace_df = None

    # Try to fetch iteration count from internal ppc
    try:
        ppc = net._ppc  # type: ignore[attr-defined]
        internal = ppc.get("internal", {}) if isinstance(ppc, dict) else {}
        if isinstance(internal, dict):
            iterations = internal.get("iterations")
            # Some pandapower versions may store a list/array under internal['convergence']
            conv = internal.get("convergence")
            if conv is not None:
                # Make a best-effort dataframe
                try:
                    trace_df = pd.DataFrame(conv)
                except Exception:
                    pass
    except Exception:
        pass

    out: Dict[str, Any] = {
        "converged": converged,
        "error": err_msg,
        "iterations": iterations,
        "bus_results": bus_results,
        "line_results": line_results,
        "trafo_results": trafo_results,
        "total_losses_mw": total_losses_mw,
        "total_losses_mvar": total_losses_mvar,
    }
    if trace_df is not None and not trace_df.empty:
        out["convergence_trace"] = trace_df

    return out
