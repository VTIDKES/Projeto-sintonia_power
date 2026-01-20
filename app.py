"""
Power System Studio (Streamlit)

Run:
  streamlit run app.py

- Newton-Raphson AC (pandapower)
- Simplified methods (DC flow + voltage drop/loss estimates)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="Sintonia Power System Studio (Single File)", layout="wide")


# =========================================================
# DEFAULT TABLES
# =========================================================
def default_tables() -> Dict[str, pd.DataFrame]:
    buses = pd.DataFrame(
        [
            {"name": "Bus 1", "vn_kv": 230.0, "type": "Slack", "x": 0.0, "y": 0.0},
            {"name": "Bus 2", "vn_kv": 115.0, "type": "PQ", "x": 1.0, "y": 0.0},
            {"name": "Bus 3", "vn_kv": 115.0, "type": "PQ", "x": 2.0, "y": -0.4},
            {"name": "Bus 4", "vn_kv": 115.0, "type": "PV", "x": 2.0, "y": 0.5},
        ]
    )

    gens = pd.DataFrame(
        [
            {
                "name": "G1",
                "bus": "Bus 4",
                "p_mw": 50.0,
                "vm_pu": 1.02,
                "min_q_mvar": -50.0,
                "max_q_mvar": 50.0,
                "slack": False,
            }
        ]
    )

    loads = pd.DataFrame(
        [
            {"name": "Load 1", "bus": "Bus 2", "p_mw": 40.0, "q_mvar": 15.0},
            {"name": "Load 2", "bus": "Bus 3", "p_mw": 30.0, "q_mvar": 10.0},
        ]
    )

    # canonical: from_bus/to_bus
    lines = pd.DataFrame(
        [
            {
                "name": "L23",
                "from_bus": "Bus 2",
                "to_bus": "Bus 3",
                "length_km": 10.0,
                "r_ohm_per_km": 0.08,
                "x_ohm_per_km": 0.32,
                "c_nf_per_km": 10.0,
                "max_i_ka": 0.40,
                "std_type": "",
            },
            {
                "name": "L24",
                "from_bus": "Bus 2",
                "to_bus": "Bus 4",
                "length_km": 8.0,
                "r_ohm_per_km": 0.08,
                "x_ohm_per_km": 0.30,
                "c_nf_per_km": 10.0,
                "max_i_ka": 0.40,
                "std_type": "",
            },
        ]
    )

    trafos = pd.DataFrame(
        [
            {
                "name": "T12",
                "hv_bus": "Bus 1",
                "lv_bus": "Bus 2",
                "sn_mva": 100.0,
                "vn_hv_kv": 230.0,
                "vn_lv_kv": 115.0,
                "vk_percent": 10.0,
                "vkr_percent": 0.40,
                "pfe_kw": 0.0,
                "i0_percent": 0.0,
                "tap_side": "hv",
                "tap_neutral": 0,
                "tap_min": -2,
                "tap_max": 2,
                "tap_step_percent": 1.25,
            }
        ]
    )

    buses = buses.reindex(columns=["name", "vn_kv", "type", "x", "y"])
    gens = gens.reindex(columns=["name", "bus", "p_mw", "vm_pu", "min_q_mvar", "max_q_mvar", "slack"])
    loads = loads.reindex(columns=["name", "bus", "p_mw", "q_mvar"])
    lines = lines.reindex(
        columns=[
            "name",
            "from_bus",
            "to_bus",
            "length_km",
            "r_ohm_per_km",
            "x_ohm_per_km",
            "c_nf_per_km",
            "max_i_ka",
            "std_type",
        ]
    )
    trafos = trafos.reindex(
        columns=[
            "name",
            "hv_bus",
            "lv_bus",
            "sn_mva",
            "vn_hv_kv",
            "vn_lv_kv",
            "vk_percent",
            "vkr_percent",
            "pfe_kw",
            "i0_percent",
            "tap_side",
            "tap_neutral",
            "tap_min",
            "tap_max",
            "tap_step_percent",
        ]
    )

    return {"buses": buses, "gens": gens, "loads": loads, "lines": lines, "trafos": trafos}


def export_table_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# =========================================================
# VALIDATION
# =========================================================
def validate_tables(tables: Dict[str, pd.DataFrame]) -> Tuple[bool, List[str], List[str]]:
    warnings: List[str] = []
    errors: List[str] = []

    for key in ["buses", "gens", "loads", "lines", "trafos"]:
        if key not in tables:
            errors.append(f"Tabela ausente: {key}")
            return False, warnings, errors

    buses = tables["buses"]
    if buses.empty:
        errors.append("buses: tabela vazia.")
        return False, warnings, errors

    for c in ["name", "vn_kv", "type"]:
        if c not in buses.columns:
            errors.append(f"buses: faltando coluna '{c}'")

    if errors:
        return False, warnings, errors

    bus_names = buses["name"].astype(str)
    if bus_names.duplicated().any():
        errors.append("buses: nomes duplicados em 'name'.")

    slack_count = buses["type"].astype(str).str.upper().eq("SLACK").sum()
    if slack_count == 0:
        warnings.append("Nenhuma barra Slack definida (buses.type). Recomendado 1 Slack.")
    if slack_count > 1:
        warnings.append("Mais de uma barra Slack definida; será usada a primeira.")

    busset = set(bus_names.tolist())

    def check_ref(dfkey: str, col: str):
        df = tables[dfkey]
        if df.empty or col not in df.columns:
            return
        bad = df[~df[col].astype(str).isin(busset)]
        if len(bad) > 0:
            errors.append(f"{dfkey}: referência inválida de barra na coluna '{col}'.")

    check_ref("gens", "bus")
    check_ref("loads", "bus")
    if not tables["lines"].empty:
        check_ref("lines", "from_bus")
        check_ref("lines", "to_bus")
    if not tables["trafos"].empty:
        check_ref("trafos", "hv_bus")
        check_ref("trafos", "lv_bus")

    return (len(errors) == 0), warnings, errors


# =========================================================
# BUILD PANDAPOWER NETWORK + NEWTON-RAPHSON
# =========================================================
def build_network(tables: Dict[str, pd.DataFrame]):
    import pandapower as pp

    net = pp.create_empty_network()

    buses = tables["buses"].copy()
    buses["name"] = buses["name"].astype(str)

    bus_map: Dict[str, int] = {}
    for _, r in buses.iterrows():
        idx = pp.create_bus(net, vn_kv=float(r["vn_kv"]), name=str(r["name"]))
        bus_map[str(r["name"])] = int(idx)

    slack_buses = buses[buses["type"].astype(str).str.upper() == "SLACK"]["name"].astype(str).tolist()
    if slack_buses:
        pp.create_ext_grid(net, bus=bus_map[slack_buses[0]], vm_pu=1.0, name="Slack")

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

    # Try to capture convergence trace if available; if not, we still return iterations
    convergence_trace = None

    pp.runpp(net, algorithm="nr", tolerance_mva=tol, max_iteration=max_iter, init=init)

    bus_res = net.res_bus.copy()
    line_res = net.res_line.copy() if "res_line" in net and len(net.line) > 0 else pd.DataFrame()
    trafo_res = net.res_trafo.copy() if "res_trafo" in net and len(net.trafo) > 0 else pd.DataFrame()

    total_losses_mw = float(net.res_line.pl_mw.sum()) if not line_res.empty else 0.0
    total_losses_mvar = float(net.res_line.ql_mvar.sum()) if not line_res.empty else 0.0

    # pandapower provides net["_ppc"] internal; not stable for trace. Keep minimal.
    return {
        "converged": True,
        "iterations": int(net.get("iterations", max_iter)),
        "bus_results": bus_res,
        "line_results": line_res,
        "trafo_results": trafo_res,
        "total_losses_mw": total_losses_mw,
        "total_losses_mvar": total_losses_mvar,
        "convergence_trace": convergence_trace,
    }


# =========================================================
# SIMPLIFIED METHODS
# =========================================================
def dc_power_flow(buses: pd.DataFrame, gens: pd.DataFrame, loads: pd.DataFrame, lines: pd.DataFrame, slack_bus_name: str | None):
    """
    Lossless DC power flow:
      Bbus * theta = P (in pu-ish, but we compute in MW and use X as weights)
    This is a spreadsheet-like quick check (not a full AC solution).
    """
    try:
        if buses.empty:
            return {"error": "buses vazio."}

        bus_names = buses["name"].astype(str).tolist()
        n = len(bus_names)
        idx = {name: i for i, name in enumerate(bus_names)}

        if slack_bus_name is None:
            # pick first bus if none
            slack_bus_name = bus_names[0]

        if slack_bus_name not in idx:
            return {"error": f"Slack '{slack_bus_name}' não existe em buses."}

        # Net injections P = Pg - Pl
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

        # Build susceptance matrix from X (approx)
        B = np.zeros((n, n), dtype=float)

        if lines.empty:
            return {"error": "lines vazio (DC precisa de linhas)."}

        for _, ln in lines.iterrows():
            fb = str(ln["from_bus"])
            tb = str(ln["to_bus"])
            if fb not in idx or tb not in idx:
                continue
            x = float(ln.get("x_ohm_per_km", 0.0001)) * float(ln.get("length_km", 1.0))
            x = max(x, 1e-6)
            b = 1.0 / x
            i = idx[fb]
            j = idx[tb]
            B[i, i] += b
            B[j, j] += b
            B[i, j] -= b
            B[j, i] -= b

        # Solve for theta excluding slack
        s = idx[slack_bus_name]
        mask = [i for i in range(n) if i != s]
        Bred = B[np.ix_(mask, mask)]
        Pred = P[mask]

        theta = np.zeros(n, dtype=float)
        theta_red = np.linalg.solve(Bred, Pred)
        theta[mask] = theta_red
        theta[s] = 0.0

        # Line flows P_ij = (theta_i - theta_j)/X
        line_rows = []
        for _, ln in lines.iterrows():
            fb = str(ln["from_bus"])
            tb = str(ln["to_bus"])
            if fb not in idx or tb not in idx:
                continue
            x = float(ln.get("x_ohm_per_km", 0.0001)) * float(ln.get("length_km", 1.0))
            x = max(x, 1e-6)
            pij = (theta[idx[fb]] - theta[idx[tb]]) / x
            line_rows.append({"name": ln.get("name", ""), "from_bus": fb, "to_bus": tb, "p_flow_dc": pij})

        bus_df = pd.DataFrame({"bus": bus_names, "p_inj_mw": P, "theta_rad": theta, "theta_deg": theta * 180 / math.pi})

        return {"slack_bus": slack_bus_name, "bus_results": bus_df, "line_results": pd.DataFrame(line_rows)}
    except Exception as e:
        return {"error": str(e)}


def voltage_drop_and_losses(buses: pd.DataFrame, loads: pd.DataFrame, lines: pd.DataFrame):
    """
    Very rough per-line losses estimate (spreadsheet-like):
      I ~= S / (sqrt(3)*V)
      P_loss ~= 3 * I^2 * R
    Assumes nominal V for each line based on from_bus vn_kv.
    This is NOT a full AC solution.
    """
    try:
        if buses.empty or lines.empty:
            return {"error": "buses/lines vazio."}

        vn = {str(r["name"]): float(r.get("vn_kv", 0.0)) for _, r in buses.iterrows()}
        p_load = {}
        q_load = {}
        if not loads.empty:
            for _, l in loads.iterrows():
                b = str(l["bus"])
                p_load[b] = p_load.get(b, 0.0) + float(l.get("p_mw", 0.0))
                q_load[b] = q_load.get(b, 0.0) + float(l.get("q_mvar", 0.0))

        rows = []
        total_p = 0.0
        total_q = 0.0

        for _, ln in lines.iterrows():
            fb = str(ln["from_bus"])
            tb = str(ln["to_bus"])
            V_kv = vn.get(fb, 0.0)
            if V_kv <= 0:
                V_kv = vn.get(tb, 0.0)
            if V_kv <= 0:
                V_kv = 1.0

            # crude: assume line carries load at "to_bus"
            Pmw = p_load.get(tb, 0.0)
            Qmvar = q_load.get(tb, 0.0)
            Smva = math.sqrt(Pmw**2 + Qmvar**2)

            V_ll = V_kv * 1e3
            I = (Smva * 1e6) / (math.sqrt(3) * V_ll) if V_ll > 0 else 0.0

            R = float(ln.get("r_ohm_per_km", 0.0)) * float(ln.get("length_km", 1.0))
            X = float(ln.get("x_ohm_per_km", 0.0)) * float(ln.get("length_km", 1.0))

            P_loss_w = 3 * (I**2) * R
            Q_loss_var = 3 * (I**2) * X  # rough

            P_loss_mw = P_loss_w / 1e6
            Q_loss_mvar = Q_loss_var / 1e6

            total_p += P_loss_mw
            total_q += Q_loss_mvar

            # rough voltage drop: dV ≈ (R*P + X*Q)/V
            dV = 0.0
            if V_ll > 0:
                dV = (R * (Pmw * 1e6) + X * (Qmvar * 1e6)) / V_ll / 1e3  # kV-ish

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
                    "dV_kv_est": dV,
                }
            )

        return {"line_results": pd.DataFrame(rows), "total_losses_mw": total_p, "total_losses_mvar": total_q}
    except Exception as e:
        return {"error": str(e)}


# =========================================================
# VISUALIZATION
# =========================================================
def fig_network(buses: pd.DataFrame, lines: pd.DataFrame, trafos: pd.DataFrame):
    fig = go.Figure()

    if buses.empty:
        fig.update_layout(title="Rede (sem barras)")
        return fig

    b = buses.copy()
    b["name"] = b["name"].astype(str)
    if "x" not in b.columns or "y" not in b.columns:
        b["x"] = np.arange(len(b))
        b["y"] = 0.0

    pos = {row["name"]: (float(row.get("x", 0.0)), float(row.get("y", 0.0))) for _, row in b.iterrows()}

    # draw lines
    if not lines.empty:
        for _, ln in lines.iterrows():
            fb = str(ln["from_bus"])
            tb = str(ln["to_bus"])
            if fb not in pos or tb not in pos:
                continue
            x0, y0 = pos[fb]
            x1, y1 = pos[tb]
            fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines", name=str(ln.get("name", "line"))))

    # draw trafos (dashed)
    if not trafos.empty:
        for _, t in trafos.iterrows():
            hb = str(t["hv_bus"])
            lb = str(t["lv_bus"])
            if hb not in pos or lb not in pos:
                continue
            x0, y0 = pos[hb]
            x1, y1 = pos[lb]
            fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines", line=dict(dash="dash"), name=str(t.get("name", "trafo"))))

    # draw buses
    fig.add_trace(go.Scatter(x=b["x"], y=b["y"], mode="markers+text", text=b["name"], textposition="top center", name="Buses"))

    fig.update_layout(title="Diagrama unifilar (simples)", showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def fig_voltage_profile(bus_res: pd.DataFrame, bus_map: Dict[str, int] | None):
    fig = go.Figure()
    if bus_res is None or bus_res.empty:
        fig.update_layout(title="Tensões (vazio)")
        return fig

    x = list(range(len(bus_res)))
    fig.add_trace(go.Scatter(x=x, y=bus_res["vm_pu"], mode="lines+markers", name="vm_pu"))
    fig.update_layout(title="Perfil de tensão (pu)", xaxis_title="Índice da barra", yaxis_title="vm_pu")
    return fig


def fig_angle_profile(bus_res: pd.DataFrame, bus_map: Dict[str, int] | None):
    fig = go.Figure()
    if bus_res is None or bus_res.empty or "va_degree" not in bus_res.columns:
        fig.update_layout(title="Ângulos (vazio)")
        return fig
    x = list(range(len(bus_res)))
    fig.add_trace(go.Scatter(x=x, y=bus_res["va_degree"], mode="lines+markers", name="va_degree"))
    fig.update_layout(title="Ângulo de tensão (graus)", xaxis_title="Índice da barra", yaxis_title="graus")
    return fig


def fig_convergence(trace, iterations: int, converged: bool):
    fig = go.Figure()
    fig.update_layout(title=f"Convergência (iters={iterations}, converged={converged})")
    return fig


# =========================================================
# APP UI
# =========================================================
def init_state():
    if "tables" not in st.session_state:
        st.session_state.tables = default_tables()
    if "results" not in st.session_state:
        st.session_state.results = {}


def sidebar_nav() -> str:
    st.sidebar.title("Sintonia Power Studio (single file)")
    pages = ["1) Modelagem", "2) Simulação", "3) Resultados", "4) Exportar CSV", "Sobre"]
    return st.sidebar.radio("Navegação", pages, index=0)


def section_modelagem():
    st.header("Modelagem da Rede")
    tables = st.session_state.tables

    with st.expander("Barras", expanded=True):
        tables["buses"] = st.data_editor(tables["buses"], num_rows="dynamic", use_container_width=True)

    with st.expander("Geradores", expanded=False):
        tables["gens"] = st.data_editor(tables["gens"], num_rows="dynamic", use_container_width=True)

    with st.expander("Cargas", expanded=False):
        tables["loads"] = st.data_editor(tables["loads"], num_rows="dynamic", use_container_width=True)

    with st.expander("Linhas", expanded=False):
        tables["lines"] = st.data_editor(tables["lines"], num_rows="dynamic", use_container_width=True)

    with st.expander("Transformadores", expanded=False):
        tables["trafos"] = st.data_editor(tables["trafos"], num_rows="dynamic", use_container_width=True)

    st.session_state.tables = tables

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        if st.button("Validar", type="primary"):
            ok, w, e = validate_tables(tables)
            if ok:
                st.success("OK")
            if w:
                st.warning("Avisos:\n- " + "\n- ".join(w))
            if e:
                st.error("Erros:\n- " + "\n- ".join(e))
    with c2:
        if st.button("Reset exemplo"):
            st.session_state.tables = default_tables()
            st.session_state.results = {}
            st.success("Exemplo carregado.")
    with c3:
        st.info("Defina x/y nas barras para desenhar o diagrama.")

    st.plotly_chart(fig_network(tables["buses"], tables["lines"], tables["trafos"]), use_container_width=True)


def section_simulacao():
    st.header("Simulação")
    tables = st.session_state.tables

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

    if st.button("Executar", type="primary"):
        ok, w, e = validate_tables(tables)
        if e:
            st.error("Corrija antes:\n- " + "\n- ".join(e))
            return
        if w:
            st.warning("Avisos:\n- " + "\n- ".join(w))

        results: Dict[str, Any] = {}

        if method in ("Newton-Raphson (pandapower)", "Comparar (ambos)"):
            try:
                net, bus_map = build_network(tables)
                nr = run_newton_raphson(net, tol=float(tol), max_iter=int(max_iter), init=str(init))
                nr["bus_map"] = bus_map
                results["newton"] = nr
            except Exception as ex:
                results["newton"] = {"converged": False, "error": str(ex)}

        if method in ("Simplificado (planilha)", "Comparar (ambos)"):
            slack_candidates = (
                tables["buses"][tables["buses"]["type"].astype(str).str.upper() == "SLACK"]["name"].astype(str).tolist()
            )
            slack = slack_candidates[0] if slack_candidates else None
            results["simplified"] = {
                "dc": dc_power_flow(tables["buses"], tables["gens"], tables["loads"], tables["lines"], slack_bus_name=slack),
                "voltage_drop": voltage_drop_and_losses(tables["buses"], tables["loads"], tables["lines"]),
            }

        st.session_state.results = results
        st.success("Pronto! Vá em Resultados.")


def section_resultados():
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
                st.success("Convergiu")
                bus_res = nr.get("bus_results", pd.DataFrame())
                line_res = nr.get("line_results", pd.DataFrame())
                trafo_res = nr.get("trafo_results", pd.DataFrame())

                st.subheader("Barras")
                st.dataframe(bus_res, use_container_width=True)
                st.plotly_chart(fig_voltage_profile(bus_res, None), use_container_width=True)
                st.plotly_chart(fig_angle_profile(bus_res, None), use_container_width=True)

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

            st.subheader("DC")
            if dc.get("error"):
                st.error(dc["error"])
            else:
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
            st.info("Use 'Comparar (ambos)' na simulação.")
        else:
            df = pd.DataFrame(
                [
                    {"método": "Newton-Raphson", "P_loss_MW": nr.get("total_losses_mw", 0.0), "Q_loss_MVAr": nr.get("total_losses_mvar", 0.0)},
                    {"método": "Simplificado", "P_loss_MW": simp.get("voltage_drop", {}).get("total_losses_mw", 0.0), "Q_loss_MVAr": simp.get("voltage_drop", {}).get("total_losses_mvar", 0.0)},
                ]
            )
            st.dataframe(df, use_container_width=True)


def section_exportar_csv():
    st.header("Exportar CSV")
    tables = st.session_state.tables
    for name, df in tables.items():
        st.download_button(
            f"Baixar CSV: {name}",
            data=export_table_csv(df),
            file_name=f"{name}.csv",
            mime="text/csv",
            key=f"csv_{name}",
        )


def section_sobre():
    st.header("Sobre")
    st.write("App single-file para Streamlit Cloud (sem pasta modules).")


def main():
    init_state()
    page = sidebar_nav()

    if page.startswith("1"):
        section_modelagem()
    elif page.startswith("2"):
        section_simulacao()
    elif page.startswith("3"):
        section_resultados()
    elif page.startswith("4"):
        section_exportar_csv()
    else:
        section_sobre()


if __name__ == "__main__":
    main()
