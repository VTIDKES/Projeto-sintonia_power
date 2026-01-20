"""Power System Studio (Streamlit)

Main Streamlit app.

Run locally:
  streamlit run app.py

This app lets you build a small power system network and run power flow using:
- Newton-Raphson AC (pandapower)
- Simplified spreadsheet-like methods (DC flow + voltage drop/loss estimates)

Project structure:
- app.py
- modules/

Author: generated template
"""

from __future__ import annotations

import io
import json
from typing import Any, Dict

import pandas as pd
import streamlit as st

from modules.data_handler import default_tables, export_project_json, import_project_json, export_table_csv
from modules.network_builder import build_network, validate_tables
from modules.power_flow import run_newton_raphson
from modules.simplified_calc import dc_power_flow, voltage_drop_and_losses
from modules.visualization import fig_voltage_profile, fig_angle_profile, fig_convergence, fig_network


st.set_page_config(page_title="Power System Studio", layout="wide")


def _init_state() -> None:
    if "tables" not in st.session_state:
        st.session_state.tables = default_tables()
    if "results" not in st.session_state:
        st.session_state.results = {}
    if "meta" not in st.session_state:
        st.session_state.meta = {}


def _sidebar_nav() -> str:
    st.sidebar.title("Power System Studio")
    pages = [
        "1) Modelagem da Rede",
        "2) Simulação",
        "3) Resultados",
        "4) Relatórios",
        "Sobre",
    ]
    return st.sidebar.radio("Navegação", pages, index=0)


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
        st.write("Use r/x/c/max_i ou std_type. Se std_type estiver vazio, usa parâmetros explícitos.")
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
                st.error("Erros\n- " + "\n- ".join(errors))

    with col2:
        if st.button("Carregar Exemplo (reset)"):
            st.session_state.tables = default_tables()
            st.session_state.results = {}
            st.success("Exemplo carregado.")

    with col3:
        st.info("Dica: coloque coordenadas x/y nas barras para desenhar o diagrama unifilar.")


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
            st.error("Corrija os erros antes de simular\n- " + "\n- ".join(errors))
            return
        if warnings:
            st.warning("Avisos:\n- " + "\n- ".join(warnings))

        results: Dict[str, Any] = {}

        # Build pandapower net only if needed
        net = None
        if method in ("Newton-Raphson (pandapower)", "Comparar (ambos)"):
            try:
                net, bus_name_map = build_network(tables)
                nr = run_newton_raphson(net, tol=float(tol), max_iter=int(max_iter), init=str(init))
                nr["bus_name_map"] = bus_name_map
                results["newton"] = nr
            except Exception as e:
                results["newton"] = {"converged": False, "error": str(e)}

        if method in ("Simplificado (planilha)", "Comparar (ambos)"):
            # DC flow uses only buses/gens/loads/lines
            slack_candidates = tables["buses"][tables["buses"]["type"].astype(str).str.upper() == "SLACK"]["name"].astype(str).tolist()
            slack_name = slack_candidates[0] if slack_candidates else None
            dc = dc_power_flow(tables["buses"], tables["gens"], tables["loads"], tables["lines"], slack_bus_name=slack_name)
            vd = voltage_drop_and_losses(tables["buses"], tables["loads"], tables["lines"])
            results["simplified"] = {"dc": dc, "voltage_drop": vd}

        st.session_state.results = results
        st.success("Simulação concluída. Vá para a aba Resultados.")

    st.subheader("Diagrama unifilar (opcional)")
    st.plotly_chart(fig_network(tables["buses"], tables["lines"], tables["trafos"]), use_container_width=True)


def _section_results() -> None:
    st.header("Resultados")

    results: Dict[str, Any] = st.session_state.get("results", {})
    if not results:
        st.info("Nenhuma simulação executada ainda. Vá em 'Simulação'.")
        return

    tabs = st.tabs(["Newton-Raphson", "Simplificado", "Comparação"])

    with tabs[0]:
        nr = results.get("newton")
        if not nr:
            st.info("Não executado.")
        else:
            converged = nr.get("converged", False)
            if converged:
                st.success("Convergiu")
            else:
                st.error("Não convergiu")
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
                fig_convergence(nr.get("convergence_trace"), nr.get("iterations"), bool(converged)),
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
                c2.metric("Perdas reativas estimadas (MVAr)", f"{vd.get('total_losses_mvar', 0.0):.4f}")

    with tabs[2]:
        nr = results.get("newton", {})
        simp = results.get("simplified", {})

        if not nr or not simp:
            st.info("Execute 'Comparar (ambos)' na aba Simulação.")
        else:
            st.subheader("Perdas: Newton-Raphson vs Estimativas")
            df = pd.DataFrame([
                {"método": "Newton-Raphson", "P_loss_MW": nr.get("total_losses_mw", 0.0), "Q_loss_MVAr": nr.get("total_losses_mvar", 0.0)},
                {"método": "Simplificado", "P_loss_MW": simp.get("voltage_drop", {}).get("total_losses_mw", 0.0), "Q_loss_MVAr": simp.get("voltage_drop", {}).get("total_losses_mvar", 0.0)},
            ])
            st.dataframe(df, use_container_width=True)

            st.subheader("Observações")
            st.write(
                "O método simplificado é útil para conferência rápida (estilo planilha), "
                "mas pode divergir bastante do AC completo, principalmente em redes com tap, "
                "reativos relevantes e níveis de tensão diferentes."
            )


def _section_reports() -> None:
    st.header("Relatórios e Exportação")

    tables: Dict[str, pd.DataFrame] = st.session_state.tables
    results: Dict[str, Any] = st.session_state.get("results", {})

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Exportar configuração")
        project_json = export_project_json(tables=tables, results=results)
        st.download_button(
            "Baixar JSON do projeto",
            data=project_json,
            file_name="power_system_studio_project.json",
            mime="application/json",
        )

        # Individual CSVs
        for name, df in tables.items():
            st.download_button(
                f"Baixar CSV: {name}",
                data=export_table_csv(df),
                file_name=f"{name}.csv",
                mime="text/csv",
                key=f"csv_{name}",
            )

    with col2:
        st.subheader("Importar configuração")
        uploaded = st.file_uploader("Escolha um JSON do projeto", type=["json"])
        if uploaded is not None:
            try:
                tables_new, results_new, meta = import_project_json(uploaded.read().decode("utf-8"))
                st.session_state.tables = tables_new
                st.session_state.results = results_new
                st.session_state.meta = meta
                st.success("Projeto importado.")
            except Exception as e:
                st.error(f"Falha ao importar: {e}")

    st.divider()
    st.subheader("Resumo executivo")

    if results.get("newton"):
        nr = results["newton"]
        st.write(
            {
                "converged": nr.get("converged"),
                "iterations": nr.get("iterations"),
                "total_losses_mw": nr.get("total_losses_mw"),
                "total_losses_mvar": nr.get("total_losses_mvar"),
            }
        )
    else:
        st.info("Execute uma simulação para gerar o resumo.")


def _section_about() -> None:
    st.header("Sobre")
    st.markdown(
        """
Este projeto é um **template completo** para um app Streamlit de fluxo de potência.

**O que ele entrega agora:**
- CRUD via `st.data_editor` para barras, geradores, cargas, linhas e transformadores.
- Export/Import do projeto em JSON e export de tabelas CSV.
- Fluxo de potência AC via **Newton-Raphson (pandapower)**.
- Métodos **simplificados** para conferência (DC e estimativas de queda/perdas).
- Visualizações (tensões, ângulos, diagrama e convergência best-effort).

**Próximos upgrades (diferenciais):**
- Histórico de simulações (salvar múltiplas execuções e comparar).
- Análise de sensibilidade (varrer carga/geração e plotar impacto).
- Temas claro/escuro (Streamlit themes).
"""
    )


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
        _section_reports()
    else:
        _section_about()


if __name__ == "__main__":
    main()
