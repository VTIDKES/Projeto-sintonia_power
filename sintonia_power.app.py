# STREAMLIT + PANDAPOWER
# Sistemas Elétricos de Potência
# Autor: Vitor

import streamlit as st
import pandapower as pp
import pandapower.networks as pn
import pandas as pd

st.set_page_config(page_title="Sistemas Elétricos de Potência", layout="wide")

st.title("⚡ Sistemas Elétricos de Potência – pandapower")
st.write("Aplicação interativa para fluxo de carga em sistemas elétricos.")

st.sidebar.header("Configuração da Rede")

network_type = st.sidebar.selectbox(
    "Escolha a rede",
    ["Rede simples 3 barras", "Rede IEEE 9 barras"]
)

def create_network(network_type):
    if network_type == "Rede simples 3 barras":
        net = pp.create_empty_network()

        b1 = pp.create_bus(net, vn_kv=13.8, name="Barra 1")
        b2 = pp.create_bus(net, vn_kv=13.8, name="Barra 2")
        b3 = pp.create_bus(net, vn_kv=13.8, name="Barra 3")

        pp.create_ext_grid(net, bus=b1, vm_pu=1.02)

        pp.create_line_from_parameters(
            net, b1, b2, length_km=2,
            r_ohm_per_km=0.642, x_ohm_per_km=0.083,
            c_nf_per_km=210, max_i_ka=0.4
        )

        pp.create_line_from_parameters(
            net, b2, b3, length_km=1.5,
            r_ohm_per_km=0.642, x_ohm_per_km=0.083,
            c_nf_per_km=210, max_i_ka=0.4
        )

        pp.create_load(net, b2, p_mw=2.0, q_mvar=0.8)
        pp.create_load(net, b3, p_mw=1.5, q_mvar=0.6)

        return net
    else:
        return pn.case9()

net = create_network(network_type)

if st.button("Executar Fluxo de Carga"):
    pp.runpp(net)

    st.success("Fluxo de carga executado!")

    st.subheader("Tensões nas Barras")
    st.dataframe(net.res_bus[["vm_pu", "va_degree"]])

    st.subheader("Carregamento das Linhas")
    st.dataframe(net.res_line[["p_from_mw", "q_from_mvar", "loading_percent"]])
