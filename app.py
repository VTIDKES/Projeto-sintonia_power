"""
Power System Studio (Streamlit)

Run:
  streamlit run app.py

- Newton-Raphson AC (pandapower)
- Simplified methods (DC flow + voltage drop/loss estimates)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# -----------------------------
# OPTIONAL dependency: Streamlit Flow
# -----------------------------
try:
    from streamlit_flow import streamlit_flow
    from streamlit_flow.elements import StreamlitFlowEdge, StreamlitFlowNode
    from streamlit_flow.state import StreamlitFlowState
except Exception as e:
    streamlit_flow = None
    StreamlitFlowState = None
    StreamlitFlowNode = None
    StreamlitFlowEdge = None
    _FLOW_IMPORT_ERROR = str(e)

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Sintonia Power Studio (Visual)", layout="wide")


# =========================================================
# HELPERS
# =========================================================
def _uid(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:10]}"


def _safe_str(x: Any, default: str = "") -> str:
    try:
        return default if x is None else str(x)
    except Exception:
        return default


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or (isinstance(x, str) and x.strip() == ""):
            return default
        return float(x)
    except Exception:
        return default


def _safe_bool(x: Any, default: bool = False) -> bool:
    try:
        if isinstance(x, bool):
            return x
        if isinstance(x, str):
            return x.strip().lower() in ("1", "true", "yes", "y", "sim")
        if isinstance(x, (int, float)):
            return bool(x)
        return default
    except Exception:
        return default


# =========================================================
# DEFAULT FLOW (start empty-ish but usable)
# =========================================================
def default_flow_state() -> "StreamlitFlowState":
    # A small starter so the canvas isn't blank
    nodes = [
        StreamlitFlowNode(
            id="BUS_1",
            pos=(0, 0),
            data={
                "kind": "BUS",
                "name": "Bus 1",
                "vn_kv": 230.0,
                "bus_type": "Slack",
                "label": "Bus 1 (Slack)",
            },
            node_type="input",
        ),
        StreamlitFlowNode(
            id="BUS_2",
            pos=(260, 0),
            data={
                "kind": "BUS",
                "name": "Bus 2",
                "vn_kv": 115.0,
                "bus_type": "PQ",
                "label": "Bus 2 (PQ)",
            },
            node_type="default",
        ),
    ]
    edges = [
        StreamlitFlowEdge(
            id="LINE_1",
            source="BUS_1",
            target="BUS_2",
            data={
                "kind": "LINE",
                "name": "L12",
                "length_km": 10.0,
                "r_ohm_per_km": 0.08,
                "x_ohm_per_km": 0.32,
                "c_nf_per_km": 10.0,
                "max_i_ka": 0.40,
                "std_type": "",
            },
            animated=False,
        )
    ]
    return StreamlitFlowState(nodes, edges)


# =========================================================
# GRAPH -> TABLES
# =========================================================
def _nodes_by_kind(state: "StreamlitFlowState", kind: str) -> List["StreamlitFlowNode"]:
    out = []
    for n in getattr(state, "nodes", []) or []:
        data = getattr(n, "data", {}) or {}
        if _safe_str(data.get("kind", "")).upper() == kind.upper():
            out.append(n)
    return out


def _edges_by_kind(state: "StreamlitFlowState", kind: str) -> List["StreamlitFlowEdge"]:
    out = []
    for e in getattr(state, "edges", []) or []:
        data = getattr(e, "data", {}) or {}
        if _safe_str(data.get("kind", "")).upper() == kind.upper():
            out.append(e)
    return out


def _node_id_to_busname(state: "StreamlitFlowState") -> Dict[str, str]:
    m = {}
    for n in _nodes_by_kind(state, "BUS"):
        d = n.data or {}
        m[n.id] = _safe_str(d.get("name", n.id), n.id)
    return m


def _find_connected_bus_for_device(state: "StreamlitFlowState", device_node_id: str) -> Optional[str]:
    """
    GEN/LOAD nodes typically connect to exactly one BUS node via an edge.
    If found, returns BUS node id; else None.
    """
    for e in getattr(state, "edges", []) or []:
        if e.source == device_node_id and _safe_str((e.data or {}).get("kind", "")).upper() == "CONNECT":
            return e.target
        if e.target == device_node_id and _safe_str((e.data or {}).get("kind", "")).upper() == "CONNECT":
            return e.source
    return None


def graph_to_tables(state: "StreamlitFlowState") -> Dict[str, pd.DataFrame]:
    # Buses
    buses_rows = []
    for n in _nodes_by_kind(state, "BUS"):
        d = n.data or {}
        x, y = (n.pos or (0, 0))
        name = _safe_str(d.get("name", n.id), n.id)
        vn_kv = _safe_float(d.get("vn_kv", 0.0), 0.0)
        bus_type = _safe_str(d.get("bus_type", "PQ"), "PQ")
        buses_rows.append({"name": name, "vn_kv": vn_kv, "type": bus_type, "x": float(x), "y": float(y)})
    buses = pd.DataFrame(buses_rows).reindex(columns=["name", "vn_kv", "type", "x", "y"])

    bus_id2name = _node_id_to_busname(state)
    bus_name_set = set(buses["name"].astype(str).tolist()) if not buses.empty else set()

    # Lines (edges between BUS nodes)
    lines_rows = []
    for e in _edges_by_kind(state, "LINE"):
        d = e.data or {}
        fb = bus_id2name.get(e.source, "")
        tb = bus_id2name.get(e.target, "")
        if fb and tb and fb in bus_name_set and tb in bus_name_set:
            lines_rows.append(
                {
                    "name": _safe_str(d.get("name", e.id), e.id),
                    "from_bus": fb,
                    "to_bus": tb,
                    "length_km": _safe_float(d.get("length_km", 1.0), 1.0),
                    "r_ohm_per_km": _safe_float(d.get("r_ohm_per_km", 0.0), 0.0),
                    "x_ohm_per_km": _safe_float(d.get("x_ohm_per_km", 0.0001), 0.0001),
                    "c_nf_per_km": _safe_float(d.get("c_nf_per_km", 0.0), 0.0),
                    "max_i_ka": _safe_float(d.get("max_i_ka", 1.0), 1.0),
                    "std_type": _safe_str(d.get("std_type", ""), ""),
                }
            )
    lines = pd.DataFrame(lines_rows).reindex(
        columns=["name", "from_bus", "to_bus", "length_km", "r_ohm_per_km", "x_ohm_per_km", "c_nf_per_km", "max_i_ka", "std_type"]
    )

    # Trafos (edges between BUS nodes)
    trafos_rows = []
    for e in _edges_by_kind(state, "TRAFO"):
        d = e.data or {}
        hv = bus_id2name.get(e.source, "")
        lv = bus_id2name.get(e.target, "")
        if hv and lv and hv in bus_name_set and lv in bus_name_set:
            trafos_rows.append(
                {
                    "name": _safe_str(d.get("name", e.id), e.id),
                    "hv_bus": hv,
                    "lv_bus": lv,
                    "sn_mva": _safe_float(d.get("sn_mva", 100.0), 100.0),
                    "vn_hv_kv": _safe_float(d.get("vn_hv_kv", 230.0), 230.0),
                    "vn_lv_kv": _safe_float(d.get("vn_lv_kv", 115.0), 115.0),
                    "vk_percent": _safe_float(d.get("vk_percent", 10.0), 10.0),
                    "vkr_percent": _safe_float(d.get("vkr_percent", 0.4), 0.4),
                    "pfe_kw": _safe_float(d.get("pfe_kw", 0.0), 0.0),
                    "i0_percent": _safe_float(d.get("i0_percent", 0.0), 0.0),
                    "tap_side": _safe_str(d.get("tap_side", "hv"), "hv"),
                    "tap_neutral": int(_safe_float(d.get("tap_neutral", 0), 0)),
                    "tap_min": int(_safe_float(d.get("tap_min", -2), -2)),
                    "tap_max": int(_safe_float(d.get("tap_max", 2), 2)),
                    "tap_step_percent": _safe_float(d.get("tap_step_percent", 1.25), 1.25),
                }
            )
    trafos = pd.DataFrame(trafos_rows).reindex(
        columns=["name", "hv_bus", "lv_bus", "sn_mva", "vn_hv_kv", "vn_lv_kv", "vk_percent", "vkr_percent", "pfe_kw", "i0_percent",
                 "tap_side", "tap_neutral", "tap_min", "tap_max", "tap_step_percent"]
    )

    # Gens + Loads (nodes that reference a BUS either by connect edge or by property)
    gens_rows = []
    loads_rows = []

    # Make a mapping from BUS node id -> BUS name
    bus_nodeid2busname = {nid: nname for nid, nname in bus_id2name.items()}

    for n in _nodes_by_kind(state, "GEN"):
        d = n.data or {}
        name = _safe_str(d.get("name", n.id), n.id)
        # Try connection
        bus_node = _find_connected_bus_for_device(state, n.id)
        bus_name = bus_nodeid2busname.get(bus_node, "") if bus_node else _safe_str(d.get("bus", ""), "")
        if bus_name and bus_name in bus_name_set:
            gens_rows.append(
                {
                    "name": name,
                    "bus": bus_name,
                    "p_mw": _safe_float(d.get("p_mw", 0.0), 0.0),
                    "vm_pu": _safe_float(d.get("vm_pu", 1.0), 1.0),
                    "min_q_mvar": _safe_float(d.get("min_q_mvar", -999.0), -999.0),
                    "max_q_mvar": _safe_float(d.get("max_q_mvar", 999.0), 999.0),
                    "slack": _safe_bool(d.get("slack", False), False),
                }
            )

    for n in _nodes_by_kind(state, "LOAD"):
        d = n.data or {}
        name = _safe_str(d.get("name", n.id), n.id)
        bus_node = _find_connected_bus_for_device(state, n.id)
        bus_name = bus_nodeid2busname.get(bus_node, "") if bus_node else _safe_str(d.get("bus", ""), "")
        if bus_name and bus_name in bus_name_set:
            loads_rows.append(
                {
                    "name": name,
                    "bus": bus_name,
                    "p_mw": _safe_float(d.get("p_mw", 0.0), 0.0),
                    "q_mvar": _safe_float(d.get("q_mvar", 0.0), 0.0),
                }
            )

    gens = pd.DataFrame(gens_rows).reindex(columns=["name", "bus", "p_mw", "vm_pu", "min_q_mvar", "max_q_mvar", "slack"])
    loads = pd.DataFrame(loads_rows).reindex(columns=["name", "bus", "p_mw", "q_mvar"])

    return {"buses": buses, "gens": gens, "loads": loads, "lines": lines, "trafos": trafos}


# =========================================================
# TABLE VALIDATION
# =========================================================
def validate_tables(tables: Dict[str, pd.DataFrame]) -> Tuple[bool, List[str], List[str]]:
    warnings: List[str] = []
    errors: List[str] = []

    for k in ["buses", "gens", "loads", "lines", "trafos"]:
        if k not in tables:
            errors.append(f"Tabela ausente: {k}")

    if errors:
        return False, warnings, errors

    buses = tables["buses"]
    if buses.empty:
        errors.append("Sem barras (BUS). Crie pelo menos 1 barra.")
        return False, warnings, errors

    for c in ["name", "vn_kv", "type"]:
        if c not in buses.columns:
            errors.append(f"buses: faltando coluna '{c}'")

    if errors:
        return False, warnings, errors

    if buses["name"].astype(str).duplicated().any():
        errors.append("buses: nomes duplicados em 'name'.")

    slack_count = buses["type"].astype(str).str.upper().eq("SLACK").sum()
    if slack_count == 0:
        warnings.append("Nenhuma barra Slack definida. Recomendado 1 Slack.")
    if slack_count > 1:
        warnings.append("Mais de uma Slack definida; será usada a primeira (pandapower).")

    busset = set(buses["name"].astype(str).tolist())

    def check_ref(dfkey: str, col: str):
        df = tables[dfkey]
        if df.empty or col not in df.columns:
            return
        bad = df[~df[col].astype(str).isin(busset)]
        if len(bad) > 0:
            errors.append(f"{dfkey}: referência inválida de barra na coluna '{col}'.")

    check_ref("gens", "bus")
    check_ref("loads", "bus")
    check_ref("lines", "from_bus")
    check_ref("lines", "to_bus")
    check_ref("trafos", "hv_bus")
    check_ref("trafos", "lv_bus")

    return (len(errors) == 0), warnings, errors


# =========================================================
# PANDAPOWER BUILD + NR
# =========================================================
def build_network_pp(tables: Dict[str, pd.DataFrame]):
    import pandapower as pp

    net = pp.create_empty_network()

    buses = tables["buses"].copy()
    buses["name"] = buses["name"].astype(str)

    bus_map: Dict[str, int] = {}
    for _, r in buses.iterrows():
        bidx = pp.create_bus(net, vn_kv=float(r["vn_kv"]), name=str(r["name"]))
        bus_map[str(r["name"])] = int(bidx)

    # ext_grid at first Slack (if exists)
    slack_buses = buses[buses["type"].astype(str).str.upper().eq("SLACK")]["name"].astype(str).tolist()
    if slack_buses:
        pp.create_ext_grid(net, bus=bus_map[slack_buses[0]], vm_pu=1.0, name="Slack")

    # gens
    gens = tables["gens"]
    if not gens.empty:
        for _, g in gens.iterrows():
            pp.create_gen(
                net,
                bus=bus_map[str(g["bus"])],
                p_mw=float(g.get("p_mw", 0.0)),
                vm_pu=float(g.get("vm_pu", 1.0)),
                min_q_mvar=float(g.get("min_q_mvar", -999.0)),
                max_q_mvar=float(g.get("max_q_mvar", 999.0)),
                slack=bool(g.get("slack", False)),
                name=str(g.get("name", "gen")),
            )

    # loads
    loads = tables["loads"]
    if not loads.empty:
        for _, l in loads.iterrows():
            pp.create_load(
                net,
                bus=bus_map[str(l["bus"])],
                p_mw=float(l.get("p_mw", 0.0)),
                q_mvar=float(l.get("q_mvar", 0.0)),
                name=str(l.get("name", "load")),
            )

    # lines
    lines = tables["lines"]
    if not lines.empty:
        for _, ln in lines.iterrows():
            std = str(ln.get("std_type", "")).strip()
            if std:
                pp.create_line(
                    net,
                    from_bus=bus_map[str(ln["from_bus"])],
                    to_bus=bus_map[str(ln["to_bus"])],
                    length_km=float(ln.get("length_km", 1.0)),
                    std_type=std,
                    name=str(ln.get("name", "line")),
                )
            else:
                pp.create_line_from_parameters(
                    net,
                    from_bus=bus_map[str(ln["from_bus"])],
                    to_bus=bus_map[str(ln["to_bus"])],
                    length_km=float(ln.get("length_km", 1.0)),
                    r_ohm_per_km=float(ln.get("r_ohm_per_km", 0.0)),
                    x_ohm_per_km=float(ln.get("x_ohm_per_km", 0.0001)),
                    c_nf_per_km=float(ln.get("c_nf_per_km", 0.0)),
                    max_i_ka=float(ln.get("max_i_ka", 1.0)),
                    name=str(ln.get("name", "line")),
                )

    # trafos
    trafos = tables["trafos"]
    if not trafos.empty:
        for _, t in trafos.iterrows():
            pp.create_transformer_from_parameters(
                net,
                hv_bus=bus_map[str(t["hv_bus"])],
                lv_bus=bus_map[str(t["lv_bus"])],
                sn_mva=float(t.get("sn_mva", 100.0)),
                vn_hv_kv=float(t.get("vn_hv_kv", 230.0)),
                vn_lv_kv=float(t.get("vn_lv_kv", 115.0)),
                vk_percent=float(t.get("vk_percent", 10.0)),
                vkr_percent=float(t.get("vkr_percent", 0.5)),
                pfe_kw=float(t.get("pfe_kw", 0.0)),
                i0_percent=float(t.get("i0_percent", 0.0)),
                tap_side=str(t.get("tap_side", "hv")),
                tap_neutral=int(t.get("tap_neutral", 0)),
                tap_min=int(t.get("tap_min", -2)),
                tap_max=int(t.get("tap_max", 2)),
                tap_step_percent=float(t.get("tap_step_percent", 1.25)),
                name=str(t.get("name", "trafo")),
            )

    return net, bus_map


def run_newton_raphson(net, tol: float, max_iter: int, init: str) -> Dict[str, Any]:
    import pandapower as pp

    try:
        pp.runpp(net, algorithm="nr", tolerance_mva=tol, max_iteration=max_iter, init=init)
    except Exception as e:
        return {"converged": False, "error": str(e)}

    converged = bool(getattr(net, "converged", True))

    bus_res = net.res_bus.copy()
    line_res = net.res_line.copy() if hasattr(net, "res_line") and len(net.line) > 0 else pd.DataFrame()
    trafo_res = net.res_trafo.copy() if hasattr(net, "res_trafo") and len(net.trafo) > 0 else pd.DataFrame()

    total_losses_mw = float(net.res_line.pl_mw.sum()) if not line_res.empty and "pl_mw" in line_res.columns else 0.0
    total_losses_mvar = float(net.res_line.ql_mvar.sum()) if not line_res.empty and "ql_mvar" in line_res.columns else 0.0

    return {
        "converged": converged,
        "bus_results": bus_res,
        "line_results": line_res,
        "trafo_results": trafo_res,
        "total_losses_mw": total_losses_mw,
        "total_losses_mvar": total_losses_mvar,
        "iterations": int(net.get("iterations", max_iter)) if isinstance(net, dict) else max_iter,
    }


# =========================================================
# SIMPLIFIED METHODS
# =========================================================
def dc_power_flow(buses: pd.DataFrame, gens: pd.DataFrame, loads: pd.DataFrame, lines: pd.DataFrame, slack_bus_name: Optional[str]):
    """
    Lossless DC power flow (quick spreadsheet-like check).
    """
    try:
        if buses.empty:
            return {"error": "buses vazio."}

        bus_names = buses["name"].astype(str).tolist()
        n = len(bus_names)
        idx = {name: i for i, name in enumerate(bus_names)}

        if slack_bus_name is None:
            slack_bus_name = bus_names[0]
        if slack_bus_name not in idx:
            return {"error": f"Slack '{slack_bus_name}' não existe em buses."}

        P = np.zeros(n, dtype=float)

        if not gens.empty:
            for _, g in gens.iterrows():
                b = str(g["bus"])
                if b in idx:
                    P[idx[b]] += float(g.get("p_mw", 0.0))

        if not loads.empty:
            for _, l in loads.iterrows():
                b = str(l["bus"])
                if b in idx:
                    P[idx[b]] -= float(l.get("p_mw", 0.0))

        B = np.zeros((n, n), dtype=float)
        if lines.empty:
            return {"error": "lines vazio (DC precisa de linhas)."}

        for _, ln in lines.iterrows():
            fb = str(ln["from_bus"]); tb = str(ln["to_bus"])
            if fb not in idx or tb not in idx:
                continue
            x = float(ln.get("x_ohm_per_km", 0.0001)) * float(ln.get("length_km", 1.0))
            x = max(x, 1e-6)
            bval = 1.0 / x
            i = idx[fb]; j = idx[tb]
            B[i, i] += bval; B[j, j] += bval
            B[i, j] -= bval; B[j, i] -= bval

        s = idx[slack_bus_name]
        mask = [i for i in range(n) if i != s]
        Bred = B[np.ix_(mask, mask)]
        Pred = P[mask]

        theta = np.zeros(n, dtype=float)
        theta_red = np.linalg.solve(Bred, Pred)
        theta[mask] = theta_red
        theta[s] = 0.0

        line_rows = []
        for _, ln in lines.iterrows():
            fb = str(ln["from_bus"]); tb = str(ln["to_bus"])
            if fb not in idx or tb not in idx:
                continue
            x = float(ln.get("x_ohm_per_km", 0.0001)) * float(ln.get("length_km", 1.0))
            x = max(x, 1e-6)
            pij = (theta[idx[fb]] - theta[idx[tb]]) / x
            line_rows.append({"name": ln.get("name", ""), "from_bus": fb, "to_bus": tb, "p_flow_dc": pij})

        bus_df = pd.DataFrame({"bus": bus_names, "p_inj_mw": P, "theta_deg": theta * 180 / math.pi})
        return {"slack_bus": slack_bus_name, "bus_results": bus_df, "line_results": pd.DataFrame(line_rows)}
    except Exception as e:
        return {"error": str(e)}


def voltage_drop_and_losses(buses: pd.DataFrame, loads: pd.DataFrame, lines: pd.DataFrame):
    """
    Very rough per-line losses estimate (spreadsheet-like).
    """
    try:
        if buses.empty or lines.empty:
            return {"error": "buses/lines vazio."}

        vn = {str(r["name"]): float(r.get("vn_kv", 0.0)) for _, r in buses.iterrows()}
        p_load: Dict[str, float] = {}
        q_load: Dict[str, float] = {}
        if not loads.empty:
            for _, l in loads.iterrows():
                b = str(l["bus"])
                p_load[b] = p_load.get(b, 0.0) + float(l.get("p_mw", 0.0))
                q_load[b] = q_load.get(b, 0.0) + float(l.get("q_mvar", 0.0))

        rows = []
        total_p = 0.0
        total_q = 0.0

        for _, ln in lines.iterrows():
            fb = str(ln["from_bus"]); tb = str(ln["to_bus"])
            V_kv = vn.get(fb, 0.0) or vn.get(tb, 0.0) or 1.0

            Pmw = p_load.get(tb, 0.0)
            Qmvar = q_load.get(tb, 0.0)
            Smva = math.sqrt(Pmw**2 + Qmvar**2)

            V_ll = V_kv * 1e3
            I = (Smva * 1e6) / (math.sqrt(3) * V_ll) if V_ll > 0 else 0.0

            R = float(ln.get("r_ohm_per_km", 0.0)) * float(ln.get("length_km", 1.0))
            X = float(ln.get("x_ohm_per_km", 0.0)) * float(ln.get("length_km", 1.0))

            P_loss_mw = (3 * (I**2) * R) / 1e6
            Q_loss_mvar = (3 * (I**2) * X) / 1e6

            total_p += P_loss_mw
            total_q += Q_loss_mvar

            dV_kv = 0.0
            if V_ll > 0:
                dV_kv = (R * (Pmw * 1e6) + X * (Qmvar * 1e6)) / V_ll / 1e3

            rows.append(
                {
                    "name": ln.get("name", ""),
                    "from_bus": fb,
                    "to_bus": tb,
                    "V_kv_base": V_kv,
                    "P_to_mw": Pmw,
                    "Q_to_mvar": Qmvar,
                    "I_a": I,
                    "R_ohm": R,
                    "X_ohm": X,
                    "P_loss_mw": P_loss_mw,
                    "Q_loss_mvar": Q_loss_mvar,
                    "dV_kv_est": dV_kv,
                }
            )

        return {"line_results": pd.DataFrame(rows), "total_losses_mw": total_p, "total_losses_mvar": total_q}
    except Exception as e:
        return {"error": str(e)}


# =========================================================
# PLOTS
# =========================================================
def fig_voltage_profile(bus_res: pd.DataFrame):
    fig = go.Figure()
    if bus_res is None or bus_res.empty or "vm_pu" not in bus_res.columns:
        fig.update_layout(title="Perfil de tensão (vazio)")
        return fig
    x = list(range(len(bus_res)))
    fig.add_trace(go.Scatter(x=x, y=bus_res["vm_pu"], mode="lines+markers"))
    fig.update_layout(title="Perfil de tensão (pu)", xaxis_title="Índice da barra", yaxis_title="vm_pu")
    return fig


def fig_angle_profile(bus_res: pd.DataFrame):
    fig = go.Figure()
    if bus_res is None or bus_res.empty or "va_degree" not in bus_res.columns:
        fig.update_layout(title="Ângulo de tensão (vazio)")
        return fig
    x = list(range(len(bus_res)))
    fig.add_trace(go.Scatter(x=x, y=bus_res["va_degree"], mode="lines+markers"))
    fig.update_layout(title="Ângulo de tensão (graus)", xaxis_title="Índice da barra", yaxis_title="graus")
    return fig


def export_table_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# =========================================================
# UI: INIT
# =========================================================
def init_state():
    if streamlit_flow is None:
        return
    if "flow_state" not in st.session_state:
        st.session_state.flow_state = default_flow_state()
    if "tables" not in st.session_state:
        st.session_state.tables = None
    if "results" not in st.session_state:
        st.session_state.results = {}


def sidebar_nav() -> str:
    st.sidebar.title("Sintonia Power Studio")
    return st.sidebar.radio("Navegação", ["1) Editor (Simulink)", "2) Tabelas & Simulação", "3) Resultados", "4) Exportar CSV", "Sobre"], index=0)


# =========================================================
# UI: EDITOR
# =========================================================
def section_editor():
    st.header("Editor visual (estilo Simulink)")
    st.caption("Adicione nós (BUS/GEN/LOAD) e conecte. LINE/TRAFO são arestas entre BUS. GEN/LOAD conecte ao BUS com CONNECT.")

    if streamlit_flow is None:
        st.error(
            "Não consegui importar streamlit-flow-component. Confira o requirements.txt.\n\n"
            f"Erro: {_FLOW_IMPORT_ERROR}"
        )
        return

    left, right = st.columns([2.2, 1.0], gap="large")

    with right:
        st.subheader("Adicionar elementos")

        # Add BUS
        with st.expander("➕ BUS", expanded=True):
            name = st.text_input("Nome do BUS", value=f"Bus {len(_nodes_by_kind(st.session_state.flow_state, 'BUS'))+1}", key="add_bus_name")
            vn = st.number_input("vn_kv", min_value=0.1, value=115.0, step=1.0, key="add_bus_vn")
            btype = st.selectbox("Tipo", ["PQ", "PV", "Slack"], index=0, key="add_bus_type")
            if st.button("Adicionar BUS"):
                nid = _uid("BUS")
                n = StreamlitFlowNode(
                    id=nid,
                    pos=(0, 0),
                    data={"kind": "BUS", "name": name, "vn_kv": float(vn), "bus_type": btype, "label": f"{name} ({btype})"},
                    node_type="default" if btype != "Slack" else "input",
                )
                st.session_state.flow_state.nodes.append(n)
                st.rerun()

        # Add GEN
        with st.expander("➕ GEN", expanded=False):
            gname = st.text_input("Nome do GEN", value=f"G{len(_nodes_by_kind(st.session_state.flow_state, 'GEN'))+1}", key="add_gen_name")
            p = st.number_input("P (MW)", value=10.0, step=1.0, key="add_gen_p")
            vm = st.number_input("Vset (pu)", value=1.02, step=0.01, key="add_gen_vm")
            if st.button("Adicionar GEN"):
                nid = _uid("GEN")
                n = StreamlitFlowNode(
                    id=nid,
                    pos=(0, 140),
                    data={"kind": "GEN", "name": gname, "p_mw": float(p), "vm_pu": float(vm), "min_q_mvar": -50.0, "max_q_mvar": 50.0, "slack": False, "label": f"{gname} (GEN)"},
                    node_type="default",
                )
                st.session_state.flow_state.nodes.append(n)
                st.rerun()

        # Add LOAD
        with st.expander("➕ LOAD", expanded=False):
            lname = st.text_input("Nome do LOAD", value=f"Load {len(_nodes_by_kind(st.session_state.flow_state, 'LOAD'))+1}", key="add_load_name")
            pl = st.number_input("P (MW)", value=10.0, step=1.0, key="add_load_p")
            ql = st.number_input("Q (MVAr)", value=3.0, step=0.5, key="add_load_q")
            if st.button("Adicionar LOAD"):
                nid = _uid("LOAD")
                n = StreamlitFlowNode(
                    id=nid,
                    pos=(0, 280),
                    data={"kind": "LOAD", "name": lname, "p_mw": float(pl), "q_mvar": float(ql), "label": f"{lname} (LOAD)"},
                    node_type="default",
                )
                st.session_state.flow_state.nodes.append(n)
                st.rerun()

        st.divider()
        st.subheader("Adicionar conexões (arestas)")

        bus_nodes = _nodes_by_kind(st.session_state.flow_state, "BUS")
        bus_opts = {f"{(n.data or {}).get('name', n.id)}  [{n.id}]": n.id for n in bus_nodes}

        with st.expander("➕ LINE (BUS→BUS)", expanded=True):
            if len(bus_opts) < 2:
                st.info("Crie pelo menos 2 BUS.")
            else:
                s = st.selectbox("From BUS", list(bus_opts.keys()), key="add_line_from")
                t = st.selectbox("To BUS", list(bus_opts.keys()), key="add_line_to")
                nm = st.text_input("Nome", value="L", key="add_line_name")
                length = st.number_input("length_km", value=10.0, step=1.0, key="add_line_len")
                r = st.number_input("r_ohm_per_km", value=0.08, step=0.01, key="add_line_r")
                x = st.number_input("x_ohm_per_km", value=0.32, step=0.01, key="add_line_x")
                c = st.number_input("c_nf_per_km", value=10.0, step=1.0, key="add_line_c")
                max_i = st.number_input("max_i_ka", value=0.4, step=0.1, key="add_line_maxi")
                std = st.text_input("std_type (opcional)", value="", key="add_line_std")
                if st.button("Criar LINE"):
                    sid = bus_opts[s]; tid = bus_opts[t]
                    eid = _uid("LINE")
                    e = StreamlitFlowEdge(
                        id=eid,
                        source=sid,
                        target=tid,
                        data={"kind": "LINE", "name": nm, "length_km": float(length), "r_ohm_per_km": float(r), "x_ohm_per_km": float(x), "c_nf_per_km": float(c), "max_i_ka": float(max_i), "std_type": str(std).strip()},
                        animated=False,
                    )
                    st.session_state.flow_state.edges.append(e)
                    st.rerun()

        with st.expander("➕ TRAFO (BUS→BUS)", expanded=False):
            if len(bus_opts) < 2:
                st.info("Crie pelo menos 2 BUS.")
            else:
                s = st.selectbox("HV BUS", list(bus_opts.keys()), key="add_tr_from")
                t = st.selectbox("LV BUS", list(bus_opts.keys()), key="add_tr_to")
                nm = st.text_input("Nome", value="T", key="add_tr_name")
                sn = st.number_input("sn_mva", value=100.0, step=10.0, key="add_tr_sn")
                vnh = st.number_input("vn_hv_kv", value=230.0, step=1.0, key="add_tr_vnh")
                vnl = st.number_input("vn_lv_kv", value=115.0, step=1.0, key="add_tr_vnl")
                vk = st.number_input("vk_percent", value=10.0, step=0.5, key="add_tr_vk")
                vkr = st.number_input("vkr_percent", value=0.4, step=0.1, key="add_tr_vkr")
                if st.button("Criar TRAFO"):
                    sid = bus_opts[s]; tid = bus_opts[t]
                    eid = _uid("TRAFO")
                    e = StreamlitFlowEdge(
                        id=eid,
                        source=sid,
                        target=tid,
                        data={"kind": "TRAFO", "name": nm, "sn_mva": float(sn), "vn_hv_kv": float(vnh), "vn_lv_kv": float(vnl), "vk_percent": float(vk), "vkr_percent": float(vkr)},
                        animated=False,
                    )
                    st.session_state.flow_state.edges.append(e)
                    st.rerun()

        with st.expander("➕ CONNECT (GEN/LOAD ↔ BUS)", expanded=False):
            dev_nodes = _nodes_by_kind(st.session_state.flow_state, "GEN") + _nodes_by_kind(st.session_state.flow_state, "LOAD")
            dev_opts = {f"{(n.data or {}).get('name', n.id)} ({(n.data or {}).get('kind')}) [{n.id}]": n.id for n in dev_nodes}
            if not dev_opts or not bus_opts:
                st.info("Crie BUS e GEN/LOAD.")
            else:
                dsel = st.selectbox("Dispositivo (GEN/LOAD)", list(dev_opts.keys()), key="add_conn_dev")
                bsel = st.selectbox("BUS", list(bus_opts.keys()), key="add_conn_bus")
                if st.button("Criar CONNECT"):
                    eid = _uid("CONNECT")
                    e = StreamlitFlowEdge(
                        id=eid,
                        source=dev_opts[dsel],
                        target=bus_opts[bsel],
                        data={"kind": "CONNECT"},
                        animated=True,
                    )
                    st.session_state.flow_state.edges.append(e)
                    st.rerun()

    with left:
        st.subheader("Canvas")
        st.info("Dica: você pode mover nós livremente. Edite nomes/parâmetros no painel à direita e crie conexões pelos botões acima.")
        st.session_state.flow_state = streamlit_flow(
            "flow",
            st.session_state.flow_state,
            height=650,
            fit_view=True,
        )

        st.divider()
        if st.button("📌 Gerar tabelas a partir do diagrama", type="primary"):
            st.session_state.tables = graph_to_tables(st.session_state.flow_state)
            ok, w, e = validate_tables(st.session_state.tables)
            if ok:
                st.success("Tabelas geradas e validadas.")
            if w:
                st.warning("Avisos:\n- " + "\n- ".join(w))
            if e:
                st.error("Erros:\n- " + "\n- ".join(e))


# =========================================================
# UI: TABLES + SIM
# =========================================================
def section_tables_and_sim():
    st.header("Tabelas & Simulação")
    st.caption("As tabelas abaixo vêm do seu diagrama. Você pode ajustar aqui também (opcional).")

    if st.session_state.tables is None:
        st.info("Ainda não gerei tabelas. Vá no Editor e clique em **Gerar tabelas**.")
        return

    tables = st.session_state.tables

    tabs = st.tabs(["buses", "gens", "loads", "lines", "trafos"])
    for tab, key in zip(tabs, ["buses", "gens", "loads", "lines", "trafos"]):
        with tab:
            tables[key] = st.data_editor(tables[key], num_rows="dynamic", use_container_width=True)

    st.session_state.tables = tables

    st.divider()

    method = st.radio(
        "Método",
        ["Newton-Raphson (pandapower)", "Simplificado (planilha)", "Comparar (ambos)"],
        index=2,
        horizontal=True,
    )

    with st.expander("Parâmetros NR", expanded=True):
        c1, c2, c3 = st.columns(3)
        tol = c1.number_input("Tolerância (MVA)", min_value=1e-12, max_value=1e-2, value=1e-8, format="%.1e")
        max_iter = c2.number_input("Máx. iterações", min_value=1, max_value=100, value=20, step=1)
        init = c3.selectbox("Inicialização", ["auto", "flat", "dc"], index=0)

    if st.button("▶ Executar simulação", type="primary"):
        ok, w, e = validate_tables(tables)
        if e:
            st.error("Corrija antes:\n- " + "\n- ".join(e))
            return
        if w:
            st.warning("Avisos:\n- " + "\n- ".join(w))

        results: Dict[str, Any] = {}

        if method in ("Newton-Raphson (pandapower)", "Comparar (ambos)"):
            try:
                net, bus_map = build_network_pp(tables)
                nr = run_newton_raphson(net, tol=float(tol), max_iter=int(max_iter), init=str(init))
                nr["bus_map"] = bus_map
                results["newton"] = nr
            except Exception as ex:
                results["newton"] = {"converged": False, "error": str(ex)}

        if method in ("Simplificado (planilha)", "Comparar (ambos)"):
            slack_candidates = tables["buses"][tables["buses"]["type"].astype(str).str.upper().eq("SLACK")]["name"].astype(str).tolist()
            slack = slack_candidates[0] if slack_candidates else None
            results["simplified"] = {
                "dc": dc_power_flow(tables["buses"], tables["gens"], tables["loads"], tables["lines"], slack_bus_name=slack),
                "voltage_drop": voltage_drop_and_losses(tables["buses"], tables["loads"], tables["lines"]),
            }

        st.session_state.results = results
        st.success("Simulação concluída. Vá em Resultados.")


# =========================================================
# UI: RESULTS
# =========================================================
def section_results():
    st.header("Resultados")
    results = st.session_state.get("results", {})
    if not results:
        st.info("Sem resultados ainda.")
        return

    tabs = st.tabs(["Newton-Raphson", "Simplificado", "Comparação"])

    with tabs[0]:
        nr = results.get("newton")
        if not nr:
            st.info("Não executado.")
        else:
            if nr.get("error"):
                st.error(nr["error"])
            else:
                st.write("Convergiu ✅" if nr.get("converged") else "Não convergiu ❌")
                bus_res = nr.get("bus_results", pd.DataFrame())
                line_res = nr.get("line_results", pd.DataFrame())
                trafo_res = nr.get("trafo_results", pd.DataFrame())

                st.subheader("Barras")
                st.dataframe(bus_res, use_container_width=True)
                st.plotly_chart(fig_voltage_profile(bus_res), use_container_width=True)
                st.plotly_chart(fig_angle_profile(bus_res), use_container_width=True)

                st.subheader("Linhas")
                st.dataframe(line_res, use_container_width=True)

                st.subheader("Transformadores")
                st.dataframe(trafo_res, use_container_width=True)

                c1, c2 = st.columns(2)
                c1.metric("Perdas P (MW)", f"{nr.get('total_losses_mw', 0.0):.4f}")
                c2.metric("Perdas Q (MVAr)", f"{nr.get('total_losses_mvar', 0.0):.4f}")

    with tabs[1]:
        simp = results.get("simplified")
        if not simp:
            st.info("Não executado.")
        else:
            dc = simp.get("dc", {})
            vd = simp.get("voltage_drop", {})

            st.subheader("DC (simplificado)")
            if dc.get("error"):
                st.error(dc["error"])
            else:
                st.caption(f"Slack: {dc.get('slack_bus')}")
                st.dataframe(dc.get("bus_results", pd.DataFrame()), use_container_width=True)
                st.dataframe(dc.get("line_results", pd.DataFrame()), use_container_width=True)

            st.subheader("Queda/Perdas (aprox.)")
            if vd.get("error"):
                st.error(vd["error"])
            else:
                st.dataframe(vd.get("line_results", pd.DataFrame()), use_container_width=True)
                c1, c2 = st.columns(2)
                c1.metric("Perdas P est. (MW)", f"{vd.get('total_losses_mw', 0.0):.4f}")
                c2.metric("Perdas Q est. (MVAr)", f"{vd.get('total_losses_mvar', 0.0):.4f}")

    with tabs[2]:
        nr = results.get("newton", {})
        simp = results.get("simplified", {})
        if not nr or not simp:
            st.info("Execute 'Comparar (ambos)' na simulação.")
        else:
            df = pd.DataFrame(
                [
                    {"método": "Newton-Raphson", "P_loss_MW": nr.get("total_losses_mw", 0.0), "Q_loss_MVAr": nr.get("total_losses_mvar", 0.0)},
                    {"método": "Simplificado", "P_loss_MW": simp.get("voltage_drop", {}).get("total_losses_mw", 0.0),
                     "Q_loss_MVAr": simp.get("voltage_drop", {}).get("total_losses_mvar", 0.0)},
                ]
            )
            st.dataframe(df, use_container_width=True)


# =========================================================
# UI: EXPORT CSV
# =========================================================
def section_export_csv():
    st.header("Exportar CSV")
    if st.session_state.tables is None:
        st.info("Gere as tabelas no Editor primeiro.")
        return
    for name, df in st.session_state.tables.items():
        st.download_button(
            f"Baixar CSV: {name}",
            data=export_table_csv(df),
            file_name=f"{name}.csv",
            mime="text/csv",
            key=f"csv_{name}",
        )


def section_about():
    st.header("Sobre")
    st.write(
        "App single-file para Streamlit Cloud com editor visual (ReactFlow via streamlit-flow-component) "
        "e simulação (pandapower + métodos simplificados)."
    )
    st.write("Fluxo de trabalho: **Editor → Gerar tabelas → Simulação → Resultados**.")


# =========================================================
# MAIN
# =========================================================
def main():
    init_state()

    page = sidebar_nav()
    if page.startswith("1"):
        section_editor()
    elif page.startswith("2"):
        section_tables_and_sim()
    elif page.startswith("3"):
        section_results()
    elif page.startswith("4"):
        section_export_csv()
    else:
        section_about()


if __name__ == "__main__":
    main()
