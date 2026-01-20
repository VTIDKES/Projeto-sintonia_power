"""
Power System Studio (Streamlit)

Run:
  streamlit run app.py

- Newton-Raphson AC (pandapower)
- Simplified methods (DC flow + voltage drop/loss estimates)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Power System Wizard (pandapower)", layout="wide")


# =========================================================
# Utils
# =========================================================
def _f(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, str) and x.strip() == "":
            return default
        return float(x)
    except Exception:
        return default


def _i(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        if isinstance(x, str) and x.strip() == "":
            return default
        return int(float(x))
    except Exception:
        return default


def _s(x: Any, default: str = "") -> str:
    try:
        if x is None:
            return default
        return str(x)
    except Exception:
        return default


def _bool(x: Any, default: bool = False) -> bool:
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


def _df_download(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# =========================================================
# State Init
# =========================================================
def init_state() -> None:
    if "tables" not in st.session_state:
        st.session_state.tables = default_tables()
    if "results" not in st.session_state:
        st.session_state.results = {}
    if "net_built" not in st.session_state:
        st.session_state.net_built = None


# =========================================================
# Default Tables (start minimal, user can add rows)
# =========================================================
def default_tables() -> Dict[str, pd.DataFrame]:
    buses = pd.DataFrame(
        [
            {"name": "Bus 1", "vn_kv": 220.0, "type": "Slack"},
            {"name": "Bus 2", "vn_kv": 110.0, "type": "PQ"},
        ]
    )

    ext_grids = pd.DataFrame(
        [
            {
                "name": "Grid",
                "bus": "Bus 1",
                "vm_pu": 1.0,
                "s_sc_max_mva": 100.0,
                "s_sc_min_mva": 50.0,
                "rx_min": 0.2,
                "rx_max": 0.35,
            }
        ]
    )

    gens = pd.DataFrame(
        columns=["name", "bus", "p_mw", "vm_pu", "min_q_mvar", "max_q_mvar", "slack"]
    )

    loads = pd.DataFrame(columns=["name", "bus", "p_mw", "q_mvar"])

    lines = pd.DataFrame(
        [
            {
                "name": "L12",
                "from_bus": "Bus 1",
                "to_bus": "Bus 2",
                "length_km": 10.0,
                "std_type": "N2XS(FL)2Y 1x120 RM/35 64/110 kV",
                "r_ohm_per_km": "",
                "x_ohm_per_km": "",
                "c_nf_per_km": "",
                "max_i_ka": "",
            }
        ]
    )

    trafos = pd.DataFrame(
        [
            {
                "name": "T12",
                "hv_bus": "Bus 1",
                "lv_bus": "Bus 2",
                "std_type": "100 MVA 220/110 kV",
                "sn_mva": "",
                "vn_hv_kv": "",
                "vn_lv_kv": "",
                "vk_percent": "",
                "vkr_percent": "",
                "pfe_kw": "",
                "i0_percent": "",
            }
        ]
    )

    switches = pd.DataFrame(
        columns=[
            "name",
            "bus",
            "element_type",  # "l" line | "t" trafo | "b" bus-bus
            "element_name",  # name of line/trafo/bus2
            "closed",
        ]
    )

    # normalize columns order
    buses = buses.reindex(columns=["name", "vn_kv", "type"])
    ext_grids = ext_grids.reindex(columns=["name", "bus", "vm_pu", "s_sc_max_mva", "s_sc_min_mva", "rx_min", "rx_max"])
    lines = lines.reindex(columns=["name", "from_bus", "to_bus", "length_km", "std_type", "r_ohm_per_km", "x_ohm_per_km", "c_nf_per_km", "max_i_ka"])
    trafos = trafos.reindex(columns=["name", "hv_bus", "lv_bus", "std_type", "sn_mva", "vn_hv_kv", "vn_lv_kv", "vk_percent", "vkr_percent", "pfe_kw", "i0_percent"])

    return {
        "buses": buses,
        "ext_grids": ext_grids,
        "gens": gens,
        "loads": loads,
        "lines": lines,
        "trafos": trafos,
        "switches": switches,
    }


# =========================================================
# Validation
# =========================================================
def validate_tables(t: Dict[str, pd.DataFrame]) -> Tuple[bool, List[str], List[str]]:
    warn: List[str] = []
    err: List[str] = []

    buses = t["buses"]
    if buses.empty:
        err.append("Crie pelo menos 1 bus.")
        return False, warn, err
    if buses["name"].astype(str).duplicated().any():
        err.append("buses.name tem duplicados.")
    if (buses["vn_kv"].astype(float) <= 0).any():
        err.append("buses.vn_kv deve ser > 0.")

    busset = set(buses["name"].astype(str).tolist())

    def check_ref(df: pd.DataFrame, col: str, label: str):
        if df.empty or col not in df.columns:
            return
        bad = df[~df[col].astype(str).isin(busset)]
        if len(bad) > 0:
            err.append(f"{label}: referência inválida na coluna '{col}'.")

    check_ref(t["ext_grids"], "bus", "ext_grids")
    check_ref(t["gens"], "bus", "gens")
    check_ref(t["loads"], "bus", "loads")
    check_ref(t["lines"], "from_bus", "lines")
    check_ref(t["lines"], "to_bus", "lines")
    check_ref(t["trafos"], "hv_bus", "trafos")
    check_ref(t["trafos"], "lv_bus", "trafos")

    # Slack check
    slack_count = buses["type"].astype(str).str.upper().eq("SLACK").sum()
    if slack_count == 0 and t["ext_grids"].empty:
        warn.append("Nenhuma Slack em buses.type e ext_grids vazio → fluxo pode não convergir.")
    if slack_count > 1:
        warn.append("Mais de uma Slack em buses.type; ok, mas usualmente só 1.")

    return (len(err) == 0), warn, err


# =========================================================
# Build pandapower net from tables
# =========================================================
def build_net_from_tables(t: Dict[str, pd.DataFrame]):
    import pandapower as pp

    net = pp.create_empty_network()

    # buses
    bus_map: Dict[str, int] = {}
    for _, r in t["buses"].iterrows():
        name = _s(r["name"])
        vn = _f(r["vn_kv"])
        idx = pp.create_bus(net, vn_kv=vn, name=name)
        bus_map[name] = int(idx)

    # ext_grid
    for _, r in t["ext_grids"].iterrows():
        if _s(r.get("bus", "")) not in bus_map:
            continue
        pp.create_ext_grid(
            net,
            bus=bus_map[_s(r["bus"])],
            vm_pu=_f(r.get("vm_pu", 1.0), 1.0),
            name=_s(r.get("name", "Grid"), "Grid"),
            s_sc_max_mva=_f(r.get("s_sc_max_mva", np.nan), np.nan),
            s_sc_min_mva=_f(r.get("s_sc_min_mva", np.nan), np.nan),
            rx_min=_f(r.get("rx_min", np.nan), np.nan),
            rx_max=_f(r.get("rx_max", np.nan), np.nan),
        )

    # gens
    for _, r in t["gens"].iterrows():
        b = _s(r.get("bus", ""))
        if b not in bus_map:
            continue
        pp.create_gen(
            net,
            bus=bus_map[b],
            p_mw=_f(r.get("p_mw", 0.0)),
            vm_pu=_f(r.get("vm_pu", 1.0), 1.0),
            min_q_mvar=_f(r.get("min_q_mvar", -999.0), -999.0),
            max_q_mvar=_f(r.get("max_q_mvar", 999.0), 999.0),
            slack=_bool(r.get("slack", False), False),
            name=_s(r.get("name", "gen"), "gen"),
        )

    # loads
    for _, r in t["loads"].iterrows():
        b = _s(r.get("bus", ""))
        if b not in bus_map:
            continue
        pp.create_load(
            net,
            bus=bus_map[b],
            p_mw=_f(r.get("p_mw", 0.0)),
            q_mvar=_f(r.get("q_mvar", 0.0)),
            name=_s(r.get("name", "load"), "load"),
        )

    # lines
    line_name_to_idx: Dict[str, int] = {}
    for _, r in t["lines"].iterrows():
        fb = _s(r.get("from_bus", ""))
        tb = _s(r.get("to_bus", ""))
        if fb not in bus_map or tb not in bus_map:
            continue

        std = _s(r.get("std_type", "")).strip()
        length = _f(r.get("length_km", 1.0), 1.0)

        if std:
            idx = pp.create_line(
                net,
                from_bus=bus_map[fb],
                to_bus=bus_map[tb],
                length_km=length,
                std_type=std,
                name=_s(r.get("name", "line"), "line"),
            )
        else:
            idx = pp.create_line_from_parameters(
                net,
                from_bus=bus_map[fb],
                to_bus=bus_map[tb],
                length_km=length,
                r_ohm_per_km=_f(r.get("r_ohm_per_km", 0.0)),
                x_ohm_per_km=_f(r.get("x_ohm_per_km", 0.0001), 0.0001),
                c_nf_per_km=_f(r.get("c_nf_per_km", 0.0)),
                max_i_ka=_f(r.get("max_i_ka", 1.0), 1.0),
                name=_s(r.get("name", "line"), "line"),
            )
        line_name_to_idx[_s(r.get("name", f"line_{idx}"), f"line_{idx}")] = int(idx)

    # trafos
    trafo_name_to_idx: Dict[str, int] = {}
    for _, r in t["trafos"].iterrows():
        hv = _s(r.get("hv_bus", ""))
        lv = _s(r.get("lv_bus", ""))
        if hv not in bus_map or lv not in bus_map:
            continue

        std = _s(r.get("std_type", "")).strip()
        name = _s(r.get("name", "trafo"), "trafo")

        if std:
            idx = pp.create_transformer(net, hv_bus=bus_map[hv], lv_bus=bus_map[lv], std_type=std, name=name)
        else:
            idx = pp.create_transformer_from_parameters(
                net,
                hv_bus=bus_map[hv],
                lv_bus=bus_map[lv],
                sn_mva=_f(r.get("sn_mva", 100.0), 100.0),
                vn_hv_kv=_f(r.get("vn_hv_kv", 220.0), 220.0),
                vn_lv_kv=_f(r.get("vn_lv_kv", 110.0), 110.0),
                vk_percent=_f(r.get("vk_percent", 10.0), 10.0),
                vkr_percent=_f(r.get("vkr_percent", 0.4), 0.4),
                pfe_kw=_f(r.get("pfe_kw", 0.0)),
                i0_percent=_f(r.get("i0_percent", 0.0)),
                name=name,
            )
        trafo_name_to_idx[name] = int(idx)

    # switches
    # element_type: "l" (line), "t" (trafo), "b" (bus-bus)
    for _, r in t["switches"].iterrows():
        bus = _s(r.get("bus", ""))
        et = _s(r.get("element_type", "")).strip().lower()
        ename = _s(r.get("element_name", ""))
        closed = _bool(r.get("closed", True), True)

        if bus not in bus_map:
            continue

        if et == "l":
            if ename not in line_name_to_idx:
                continue
            pp.create_switch(net, bus=bus_map[bus], element=line_name_to_idx[ename], et="l", closed=closed, name=_s(r.get("name", "sw"), "sw"))
        elif et == "t":
            if ename not in trafo_name_to_idx:
                continue
            pp.create_switch(net, bus=bus_map[bus], element=trafo_name_to_idx[ename], et="t", closed=closed, name=_s(r.get("name", "sw"), "sw"))
        elif et == "b":
            # bus-bus switch: element is another bus index
            if ename not in bus_map:
                continue
            pp.create_switch(net, bus=bus_map[bus], element=bus_map[ename], et="b", closed=closed, name=_s(r.get("name", "sw"), "sw"))

    return net


# =========================================================
# Runs
# =========================================================
def run_power_flow(net, tol: float, max_iter: int, init: str) -> Dict[str, Any]:
    import pandapower as pp

    try:
        pp.runpp(net, algorithm="nr", tolerance_mva=tol, max_iteration=max_iter, init=init)
    except Exception as e:
        return {"ok": False, "error": str(e)}

    bus = net.res_bus.copy()
    line = net.res_line.copy() if len(net.line) else pd.DataFrame()
    trafo = net.res_trafo.copy() if len(net.trafo) else pd.DataFrame()

    ploss = float(net.res_line.pl_mw.sum()) if len(net.line) and "pl_mw" in net.res_line.columns else 0.0
    qloss = float(net.res_line.ql_mvar.sum()) if len(net.line) and "ql_mvar" in net.res_line.columns else 0.0

    return {"ok": True, "bus": bus, "line": line, "trafo": trafo, "ploss_mw": ploss, "qloss_mvar": qloss}


def run_short_circuit(net, tk_s: float, ith: bool, ip: bool) -> Dict[str, Any]:
    import pandapower.shortcircuit as sc

    try:
        sc.calc_sc(net, tk_s=tk_s, ith=ith, ip=ip)
    except Exception as e:
        return {"ok": False, "error": str(e)}

    res_bus = getattr(net, "res_bus_sc", None)
    res_line = getattr(net, "res_line_sc", None)

    return {"ok": True, "bus_sc": res_bus.copy() if res_bus is not None else pd.DataFrame(),
            "line_sc": res_line.copy() if res_line is not None else pd.DataFrame()}


# =========================================================
# Simple plots
# =========================================================
def plot_voltage_profile(bus_res: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if bus_res is None or bus_res.empty or "vm_pu" not in bus_res.columns:
        fig.update_layout(title="Perfil de tensão (sem dados)")
        return fig
    fig.add_trace(go.Scatter(y=bus_res["vm_pu"], mode="lines+markers"))
    fig.update_layout(title="Perfil de tensão (pu)", xaxis_title="Índice da barra", yaxis_title="vm_pu")
    return fig


# =========================================================
# UI
# =========================================================
def sidebar_nav() -> str:
    st.sidebar.title("Power System Wizard")
    return st.sidebar.radio("Navegação", ["1) Montar Rede", "2) Simulações", "3) Resultados", "4) Exportar CSV", "Sobre"], index=0)


def page_build():
    st.header("1) Montar Rede (formulário estilo Multisim)")
    st.caption("Edite as tabelas. Você escolhe BUS/linha/trafo/elementos. Depois clique em **Construir net**.")

    t = st.session_state.tables

    tabs = st.tabs(["buses", "ext_grids", "gens", "loads", "lines", "trafos", "switches"])

    with tabs[0]:
        st.write("Barras: type = Slack/PV/PQ (Slack preferível 1).")
        t["buses"] = st.data_editor(t["buses"], num_rows="dynamic", use_container_width=True)

    with tabs[1]:
        st.write("Ext Grid: necessário para curto-circuito (parâmetros s_sc/rx).")
        t["ext_grids"] = st.data_editor(t["ext_grids"], num_rows="dynamic", use_container_width=True)

    with tabs[2]:
        t["gens"] = st.data_editor(t["gens"], num_rows="dynamic", use_container_width=True)

    with tabs[3]:
        t["loads"] = st.data_editor(t["loads"], num_rows="dynamic", use_container_width=True)

    with tabs[4]:
        st.write("Lines: use std_type OU preencha parâmetros r/x/c/max_i.")
        t["lines"] = st.data_editor(t["lines"], num_rows="dynamic", use_container_width=True)

    with tabs[5]:
        st.write("Trafos: use std_type OU preencha parâmetros.")
        t["trafos"] = st.data_editor(t["trafos"], num_rows="dynamic", use_container_width=True)

    with tabs[6]:
        st.write("Switches: element_type = l (linha), t (trafo), b (bus-bus). element_name = nome do elemento.")
        t["switches"] = st.data_editor(t["switches"], num_rows="dynamic", use_container_width=True)

    st.session_state.tables = t

    st.divider()
    c1, c2, c3 = st.columns([1, 1, 2])

    with c1:
        if st.button("Validar", type="primary"):
            ok, w, e = validate_tables(st.session_state.tables)
            if ok:
                st.success("Validação OK")
            if w:
                st.warning("Avisos:\n- " + "\n- ".join(w))
            if e:
                st.error("Erros:\n- " + "\n- ".join(e))

    with c2:
        if st.button("Reset exemplo"):
            st.session_state.tables = default_tables()
            st.session_state.results = {}
            st.session_state.net_built = None
            st.success("Exemplo resetado.")

    with c3:
        if st.button("🔧 Construir net (pandapower)", type="primary"):
            ok, w, e = validate_tables(st.session_state.tables)
            if e:
                st.error("Corrija antes:\n- " + "\n- ".join(e))
                return
            if w:
                st.warning("Avisos:\n- " + "\n- ".join(w))
            try:
                net = build_net_from_tables(st.session_state.tables)
                st.session_state.net_built = net
                st.success("net construído com sucesso.")
            except Exception as ex:
                st.error(f"Falha ao construir net: {ex}")


def page_sim():
    st.header("2) Simulações")
    if st.session_state.net_built is None:
        st.info("Primeiro vá em **Montar Rede** e clique em **Construir net**.")
        return

    net = st.session_state.net_built

    st.subheader("Fluxo de Potência (Newton-Raphson)")
    c1, c2, c3 = st.columns(3)
    tol = c1.number_input("tolerance_mva", min_value=1e-12, max_value=1e-2, value=1e-8, format="%.1e")
    max_iter = c2.number_input("max_iteration", min_value=1, max_value=100, value=20, step=1)
    init = c3.selectbox("init", ["auto", "flat", "dc"], index=0)

    if st.button("▶ Rodar runpp"):
        res = run_power_flow(net, tol=float(tol), max_iter=int(max_iter), init=str(init))
        st.session_state.results["pf"] = res
        if res.get("ok"):
            st.success("Fluxo executado.")
        else:
            st.error(res.get("error", "erro"))

    st.divider()

    st.subheader("Curto-circuito (pandapower.shortcircuit)")
    c1, c2, c3 = st.columns(3)
    tk_s = c1.number_input("tk_s", min_value=0.01, max_value=10.0, value=2.0, step=0.1)
    ith = c2.checkbox("ith", value=True)
    ip = c3.checkbox("ip", value=True)

    if st.button("⚡ Rodar calc_sc"):
        res = run_short_circuit(net, tk_s=float(tk_s), ith=bool(ith), ip=bool(ip))
        st.session_state.results["sc"] = res
        if res.get("ok"):
            st.success("Curto-circuito executado.")
        else:
            st.error(res.get("error", "erro"))


def page_results():
    st.header("3) Resultados")

    if not st.session_state.results:
        st.info("Sem resultados ainda. Vá em **Simulações**.")
        return

    tabs = st.tabs(["Fluxo de Potência", "Curto-circuito"])

    with tabs[0]:
        pf = st.session_state.results.get("pf")
        if not pf:
            st.info("Sem resultado de runpp.")
        elif not pf.get("ok"):
            st.error(pf.get("error", "erro"))
        else:
            st.subheader("res_bus")
            st.dataframe(pf["bus"], use_container_width=True)
            st.plotly_chart(plot_voltage_profile(pf["bus"]), use_container_width=True)

            st.subheader("res_line")
            st.dataframe(pf["line"], use_container_width=True)

            st.subheader("res_trafo")
            st.dataframe(pf["trafo"], use_container_width=True)

            c1, c2 = st.columns(2)
            c1.metric("Perdas P (MW)", f"{pf.get('ploss_mw', 0.0):.4f}")
            c2.metric("Perdas Q (MVAr)", f"{pf.get('qloss_mvar', 0.0):.4f}")

    with tabs[1]:
        sc = st.session_state.results.get("sc")
        if not sc:
            st.info("Sem resultado de curto-circuito.")
        elif not sc.get("ok"):
            st.error(sc.get("error", "erro"))
        else:
            st.subheader("res_bus_sc")
            st.dataframe(sc.get("bus_sc", pd.DataFrame()), use_container_width=True)

            st.subheader("res_line_sc")
            st.dataframe(sc.get("line_sc", pd.DataFrame()), use_container_width=True)


def page_export():
    st.header("4) Exportar CSV")
    t = st.session_state.tables
    for name, df in t.items():
        st.download_button(
            f"Baixar CSV: {name}",
            data=_df_download(df),
            file_name=f"{name}.csv",
            mime="text/csv",
            key=f"dl_{name}",
        )


def page_about():
    st.header("Sobre")
    st.write(
        "Este app é um **wizard** (formulário) para montar redes no pandapower: "
        "o usuário preenche tabelas (barras, linhas, trafos, etc.) e o app monta o `net` "
        "e roda fluxo (NR) e curto-circuito."
    )
    st.write("Fluxo: **Montar Rede → Construir net → Simulações → Resultados**.")


def main():
    init_state()
    page = sidebar_nav()

    if page.startswith("1"):
        page_build()
    elif page.startswith("2"):
        page_sim()
    elif page.startswith("3"):
        page_results()
    elif page.startswith("4"):
        page_export()
    else:
        page_about()


if __name__ == "__main__":
    main()

