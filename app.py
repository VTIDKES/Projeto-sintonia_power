"""
Power System Studio (Streamlit)

Run:
  streamlit run app.py

- Newton-Raphson AC (pandapower)
- Simplified methods (DC flow + voltage drop/loss estimates)
"""

from __future__ import annotations

from typing import Any, Dict
import traceback
import importlib

import pandas as pd
import streamlit as st

from modules.data_handler import default_tables, export_table_csv

# =========================================================
# IMPORT BLINDADO (para Streamlit Cloud não esconder o erro)
# =========================================================
try:
    nb = importlib.import_module("modules.network_builder")
    # Mostra o que existe (pra confirmar nomes)
    _available = [a for a in dir(nb) if "build" in a.lower() or "validate" in a.lower()]
    build_network = getattr(nb, "build_network")
    validate_tables = getattr(nb, "validate_tables")
except Exception:
    st.set_page_config(page_title="Power System Studio", layout="wide")
    st.error("Falha ao importar: modules.network_builder")
    st.code(traceback.format_exc())
    st.write("Dica: abra o arquivo `modules/network_builder.py` e confirme que existem as funções:")
    st.write("- def build_network(...):")
    st.write("- def validate_tables(...):")
    st.stop()

try:
    from modules.power_flow import run_newton_raphson
    from modules.simplified_calc import dc_power_flow, voltage_drop_and_losses
    from modules.visualization import (
        fig_voltage_profile,
        fig_angle_profile,
        fig_convergence,
        fig_network,
    )
except Exception:
    st.set_page_config(page_title="Power System Studio", layout="wide")
    st.error("Falha ao importar algum módulo em modules/ (power_flow, simplified_calc, visualization)")
    st.code(traceback.format_exc())
    st.stop()

st.set_page_config(page_title="Power System Studio", layout="wide")


# =========================================================
# State init
# =========================================================
def _init_state() -> None:
    if "tables" not in st.session_state:
        st.session_state.tables = default_tables()
    if "results" not in st.session_state:
        st.session_state.results = {}
    if "meta" not in st.session_state:
        st.session_state.meta = {}


# =========================================================
# Sidebar navigation
# =========================================================
def _sidebar_nav() -> str:
    st.sidebar.title("Power System Studio")
    pages = [
        "1) Modelagem da Rede",
        "2) Simulação",
        "3) Resultados",
        "4) Exportar CSV",
        "Sobre",
    ]
    return st.sidebar.radio("Navegação", pages, index=0)


# =========================================================
# Section 1 - Network modeling
# =========================================================
def _section_data_tables() -> None:
    st.header("Modelagem da Rede")
    st.caption("Adicione/edite/remova elementos. Use os nomes das barras como chave.")

    tables: Dict[str, pd.DataFrame] = st.session_state.tables

    with st.expander("Barras (Buses)", expanded=True):
        st.write("Colunas: name, vn_kv, type (Slack/PV/PQ), x, y")
        tables["buses"] = st.data_editor(tables["buses"], num_rows="dynamic", use_container_width=True)

    with st.expander("Geradores", expanded=False):
        st.write("Colunas: name, bus, p_mw, vm_pu, min_q_mvar, max_q_mvar, slack")
        tables["gens"] = st.data_editor(tables["gens"], num_rows="dynamic", use_container_width=True)

    with st.expander("Cargas", expanded=False):
        st.write("Colunas: name, bus, p_mw, q_mvar")
        tables["loads"] = st.data_editor(tables["loads"], num_rows="dynamic", use_container_width=True)

    with st.expander("Linhas", expanded=False):
        st.write("Use r/x/c/max_i ou std_type. Preferível: from_bus/to_bus.")
        tables["lines"] = st.data_editor(tables["lines"], num_rows="dynamic", use_container_width=True)

    with st.expander("Transformadores", expanded=False):
        st.write("Parâmetros aproximados (vk%, vkr%).")
        tables["trafos"] = st.data_editor(tables["trafos"], num_rows="dynamic", use_container_width=True)

    st.session_state.tables = tables

    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button("Validar Rede", type="primary"):
            ok, warnings, errors = validate_tables(tables)
            if ok:
                st.success("Validação OK")
            if warnings:
                st.warning("Avisos:\n- " + "\n- ".join(warnings))
            if errors:
                st.error("Erros:\n- " + "\n- ".join(errors))

    with col2:
        if st.button("Carregar Exemplo (reset)"):
            st.session_state.tables = default_tables()
            st.session_state.results = {}
            st.success("Exemplo carregado.")

    with col3:
        st.info("Dica: use coordenadas x/y nas barras para desenhar o diagrama unifilar.")


# =========================================================
# Section 2 - Simulation
# =========================================================
def _section_simulation() -> None:
    st.header("Simulação")

    tables: Dict[str, pd.DataFrame] = st.session_state.tables

    method = st.radio(
        "Método de análise",
        ["Newton-Raphson (pandapower)", "Simplificado (planilha)", "Comparar (ambos)"],
        index=2,
        horizontal=True,
    )

    with st.expander("Parâmetros", expanded=True):
        col1, col2, col3 = st.columns(3)
        tol = col1.number_input("Tolerância (MVA)", min_value=1e-12, max_value=1e-2, value=1e-8, format="%.1e")
        max_iter = col2.number_input("Máx. iterações", min_value=1, max_value=100, value=20, step=1)
        init = col3.selectbox("Inicialização", ["auto", "flat", "dc"], index=0)

    if st.button("Executar fluxo de potência", type="primary"):
        ok, warnings, errors = validate_tables(tables)

        if errors:
            st.error("Corrija os erros antes de simular:\n- " + "\n- ".join(errors))
            return

        if warnings:
            st.warning("Avisos:\n- " + "\n- ".join(warnings))

        results: Dict[str, Any] = {}

        if method in ("Newton-Raphson (pandapower)", "Comparar (ambos)"):
            try:
                net, bus_name_map = build_network(tables)
                nr = run_newton_raphson(net, tol=float(tol), max_iter=int(max_iter), init=str(init))
                nr["bus_name_map"] = bus_name_map
                results["newton"] = nr
            except Exception as e:
                results["newton"] = {"converged": False, "error": str(e)}

        if method in ("Simplificado (planilha)", "Comparar (ambos)"):
            slack_candidates = (
                tables["buses"][tables["buses"]["type"].astype(str).str.upper() == "SLACK"]["name"]
                .astype(str)
                .tolist()
            )
            slack_name = slack_candidates[0] if slack_candidates else None

            dc = dc_power_flow(tables["buses"], tables["gens"], tables["loads"], tables["lines"], slack_bus_name=slack_name)
            vd = voltage_drop_and_losses(tables["buses"], tables["loads"], tables["lines"])
            results["simplified"] = {"dc": dc, "voltage_drop": vd}

        st.session_state.results = results
        st.success("Simulação concluída. Vá para a aba Resultados.")

    st.subheader("Diagrama unifilar (opcional)")
    st.plotly_chart(fig_network(tables["buses"], tables["lines"], tables["trafos"]), use_container_width=True)


# =========================================================
# Section 3 - Results
# =========================================================
def _section_results() -> None:
    st.header("Resultados")

    results: Dict[str, Any] = st.session_state.get("results", {})
    if not results:
        st.info("Nenhuma simulação executada ainda.")
        return

    tabs = st.tabs(["Newton-Raphson", "Simplificado", "Comparação"])

    with tabs[0]:
        nr = results.get("newton")
        if not nr:
            st.info("Não executado.")
        else:
            st.success("Convergiu" if nr.get("converged") else "Não convergiu")
            if nr.get("error"):
                st.code(nr.get("error"))

            bus_res = nr.get("bus_results", pd.DataFrame())
            line_res = nr.get("line_results", pd.DataFrame())
            trafo_res = nr.get("trafo_results", pd.DataFrame())

            st.subheader("Tensões e ângulos nas barras")
            st.dataframe(bus_res, use_container_width=True)
            st.plotly_chart(fig_voltage_profile(bus_res, nr.get("bus_name_map")), use_container_width=True)
            st.plotly_chart(fig_angle_profile(bus_res, nr.get("bus_name_map")), use_container_width=True)

            st.subheader("Fluxos em linhas")
            st.dataframe(line_res, use_container_width=True)

            st.subheader("Transformadores")
            st.dataframe(trafo_res, use_container_width=True)

            st.subheader("Perdas totais")
            c1, c2 = st.columns(2)
            c1.metric("Perdas ativas (MW)", f"{nr.get('total_losses_mw', 0.0):.4f}")
            c2.metric("Perdas reativas (MVAr)", f"{nr.get('total_losses_mvar', 0.0):.4f}")

            st.subheader("Convergência")
            st.plotly_chart(
                fig_convergence(nr.get("convergence_trace"), nr.get("iterations"), bool(nr.get("converged"))),
                use_container_width=True,
            )

    with tabs[1]:
        simp = results.get("simplified")
        if not simp:
            st.info("Não executado.")
        else:
            dc = simp.get("dc", {})
            vd = simp.get("voltage_drop", {})

            st.subheader("DC power flow (simplificado)")
            if dc.get("error"):
                st.error(dc.get("error"))
            else:
                st.caption(f"Slack: {dc.get('slack_bus')}")
                st.dataframe(dc.get("bus_results", pd.DataFrame()), use_container_width=True)
                st.dataframe(dc.get("line_results", pd.DataFrame()), use_container_width=True)

            st.subheader("Quedas de tensão e perdas (estimativa)")
            if vd.get("error"):
                st.error(vd.get("error"))
            else:
                st.dataframe(vd.get("line_results", pd.DataFrame()), use_container_width=True)
                c1, c2 = st.columns(2)
                c1.metric("Perdas ativas estimadas (MW)", f"{vd.get('total_losses_mw', 0.0):.4f}")
                # ✅ corrigido: aspas
                c2.metric("Perdas reativas estimadas (MVAr)", f"{vd.get('total_losses_mvar', 0.0):.4f}")

    with tabs[2]:
        nr = results.get("newton", {})
        simp = results.get("simplified", {})
        if not nr or not simp:
            st.info("Execute 'Comparar (ambos)' na aba Simulação.")
        else:
            df = pd.DataFrame(
                [
                    {"método": "Newton-Raphson", "P_loss_MW": nr.get("total_losses_mw", 0.0), "Q_loss_MVAr": nr.get("total_losses_mvar", 0.0)},
                    {"método": "Simplificado", "P_loss_MW": simp.get("voltage_drop", {}).get("total_losses_mw", 0.0), "Q_loss_MVAr": simp.get("voltage_drop", {}).get("total_losses_mvar", 0.0)},
                ]
            )
            st.dataframe(df, use_container_width=True)


def _section_export_csv() -> None:
    st.header("Exportar CSV")
    tables: Dict[str, pd.DataFrame] = st.session_state.tables
    for name, df in tables.items():
        st.download_button(
            f"Baixar CSV: {name}",
            data=export_table_csv(df),
            file_name=f"{name}.csv",
            mime="text/csv",
            key=f"csv_{name}",
        )


def _section_about() -> None:
    st.header("Sobre")
    st.write("Template de fluxo de potência com pandapower + métodos simplificados.")


def main() -> None:
    _init_state()
    page = _sidebar_nav()

    if page.startswith("1"):
        _section_data_tables()
    elif page.startswith("2"):
        _section_simulation()
    elif page.startswith("3"):
        _section_results()
    elif page.startswith("4"):
        _section_export_csv()
    else:
        _section_about()


if __name__ == "__main__":
    main()
