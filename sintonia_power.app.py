# STREAMLIT + PANDAPOWER - Análise Completa
# Sistemas Elétricos de Potência
# Autor: Vitor

import streamlit as st
import pandapower as pp
import pandapower.networks as pn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Sistemas Elétricos de Potência", layout="wide")

st.title("⚡ Sistemas Elétricos de Potência – Análise Completa")
st.write("Aplicação interativa para fluxo de carga e análise de sistemas elétricos.")

# ========== SIDEBAR - CONFIGURAÇÃO ==========
st.sidebar.header("⚙️ Configuração da Rede")

network_type = st.sidebar.selectbox(
    "Escolha a rede",
    ["Rede simples 3 barras", "Rede com 2 Transformadores", "Rede com Transformador Único", "Rede IEEE 9 barras"]
)

# ========== PARÂMETROS EDITÁVEIS ==========
if network_type != "Rede IEEE 9 barras":
    st.sidebar.header("🔧 Parâmetros do Sistema")
    
    with st.sidebar.expander("⚡ Gerador", expanded=False):
        tensao_gerador = st.number_input("Tensão (kV)", value=13.8, step=0.1, key="v_gen")
        vm_pu = st.number_input("Tensão (pu)", value=1.0, step=0.01, key="vm_gen")
    
    if network_type in ["Rede com 2 Transformadores", "Rede com Transformador Único"]:
        with st.sidebar.expander("🔄 Transformador 1", expanded=False):
            st.write("**Trafo 1**")
            trafo1_sn = st.number_input("Potência Nominal (MVA)", value=10.0, step=1.0, key="t1_sn")
            trafo1_vn_lv = st.number_input("Tensão Baixa (kV)", value=6.9, step=0.1, key="t1_lv")
            trafo1_vn_hv = st.number_input("Tensão Alta (kV)", value=34.5, step=0.1, key="t1_hv")
            trafo1_vk = st.number_input("Impedância (%)", value=8.0, step=0.1, key="t1_vk")
            trafo1_vkr = st.number_input("Resistência (%)", value=0.5, step=0.1, key="t1_vkr")
    
    if network_type == "Rede com 2 Transformadores":
        with st.sidebar.expander("🔄 Transformador 2", expanded=False):
            st.write("**Trafo 2**")
            trafo2_sn = st.number_input("Potência Nominal (MVA)", value=5.0, step=1.0, key="t2_sn")
            trafo2_vn_hv = st.number_input("Tensão Alta (kV)", value=34.5, step=0.1, key="t2_hv")
            trafo2_vn_lv = st.number_input("Tensão Baixa (kV)", value=13.8, step=0.1, key="t2_lv")
            trafo2_vk = st.number_input("Impedância (%)", value=7.0, step=0.1, key="t2_vk")
            trafo2_vkr = st.number_input("Resistência (%)", value=0.5, step=0.1, key="t2_vkr")
    
    if network_type != "Rede simples 3 barras":
        with st.sidebar.expander("🔌 Linha de Transmissão", expanded=False):
            linha_length = st.number_input("Comprimento (km)", value=100.0, step=10.0, key="l_len")
            linha_r = st.number_input("Resistência (Ohm/km)", value=0.5, step=0.1, key="l_r")
            linha_x = st.number_input("Reatância (Ohm/km)", value=0.05, step=0.01, key="l_x")
            linha_c = st.number_input("Capacitância (kOhm x km)", value=270.0, step=10.0, key="l_c")
            linha_imax = st.number_input("Corrente Máx (kA)", value=1.0, step=0.1, key="l_imax")
    
    with st.sidebar.expander("💡 Carga", expanded=False):
        if network_type == "Rede simples 3 barras":
            carga_p = st.number_input("Potência Ativa (MW)", value=2.0, step=1.0, key="c_p")
            carga_fp = st.number_input("Fator de Potência", value=0.9, min_value=0.1, max_value=1.0, step=0.05, key="c_fp")
        else:
            carga_p = st.number_input("Potência Ativa (MW)", value=40.0, step=1.0, key="c_p")
            carga_fp = st.number_input("Fator de Potência", value=0.85, min_value=0.1, max_value=1.0, step=0.05, key="c_fp")
        carga_q = carga_p * np.tan(np.arccos(carga_fp))
        st.write(f"Potência Reativa: {carga_q:.2f} Mvar")

# ========== CRIAÇÃO DA REDE ==========
def create_network(network_type):
    if network_type == "Rede simples 3 barras":
        net = pp.create_empty_network()
        
        b1 = pp.create_bus(net, vn_kv=13.8, name="Barra 1")
        b2 = pp.create_bus(net, vn_kv=13.8, name="Barra 2")
        b3 = pp.create_bus(net, vn_kv=13.8, name="Barra 3")
        
        pp.create_ext_grid(net, bus=b1, vm_pu=1.02, name="Gerador")
        
        pp.create_line_from_parameters(
            net, b1, b2, length_km=2,
            r_ohm_per_km=0.642, x_ohm_per_km=0.083,
            c_nf_per_km=210, max_i_ka=0.4, name="LT1"
        )
        
        pp.create_line_from_parameters(
            net, b2, b3, length_km=1.5,
            r_ohm_per_km=0.642, x_ohm_per_km=0.083,
            c_nf_per_km=210, max_i_ka=0.4, name="LT2"
        )
        
        pp.create_load(net, b2, p_mw=2.0, q_mvar=0.8, name="Carga 1")
        pp.create_load(net, b3, p_mw=carga_p, q_mvar=carga_q, name="Carga 2")
        
        return net
    
    elif network_type == "Rede com 2 Transformadores":
        net = pp.create_empty_network()
        
        b_gen = pp.create_bus(net, vn_kv=trafo1_vn_lv, name="Gerador")
        b1 = pp.create_bus(net, vn_kv=trafo1_vn_lv, name="Barra 1")
        b2 = pp.create_bus(net, vn_kv=trafo1_vn_hv, name="Barra 2")
        b3 = pp.create_bus(net, vn_kv=trafo2_vn_hv, name="Barra 3")
        b4 = pp.create_bus(net, vn_kv=trafo2_vn_lv, name="Barra 4")
        
        pp.create_ext_grid(net, bus=b_gen, vm_pu=vm_pu, name="Gerador")
        
        pp.create_transformer_from_parameters(
            net, b1, b2, sn_mva=trafo1_sn, 
            vn_hv_kv=trafo1_vn_hv, vn_lv_kv=trafo1_vn_lv,
            vk_percent=trafo1_vk, vkr_percent=trafo1_vkr, 
            pfe_kw=0, i0_percent=0, name="Trafo 1"
        )
        
        pp.create_line_from_parameters(
            net, b2, b3, length_km=linha_length,
            r_ohm_per_km=linha_r, x_ohm_per_km=linha_x,
            c_nf_per_km=linha_c*1e6, max_i_ka=linha_imax, name="LT"
        )
        
        pp.create_transformer_from_parameters(
            net, b3, b4, sn_mva=trafo2_sn,
            vn_hv_kv=trafo2_vn_hv, vn_lv_kv=trafo2_vn_lv,
            vk_percent=trafo2_vk, vkr_percent=trafo2_vkr,
            pfe_kw=0, i0_percent=0, name="Trafo 2"
        )
        
        pp.create_load(net, b4, p_mw=carga_p, q_mvar=carga_q, name="Carga")
        
        return net
    
    elif network_type == "Rede com Transformador Único":
        net = pp.create_empty_network()
        
        b_gen = pp.create_bus(net, vn_kv=tensao_gerador, name="Gerador")
        b1 = pp.create_bus(net, vn_kv=trafo1_vn_lv, name="Barra 1")
        b2 = pp.create_bus(net, vn_kv=trafo1_vn_hv, name="Barra 2")
        b3 = pp.create_bus(net, vn_kv=trafo1_vn_hv, name="Barra 3")
        
        pp.create_ext_grid(net, bus=b_gen, vm_pu=vm_pu, name="Gerador")
        
        pp.create_transformer_from_parameters(
            net, b1, b2, sn_mva=trafo1_sn,
            vn_hv_kv=trafo1_vn_hv, vn_lv_kv=trafo1_vn_lv,
            vk_percent=trafo1_vk, vkr_percent=trafo1_vkr,
            pfe_kw=0, i0_percent=0, name="Trafo"
        )
        
        pp.create_line_from_parameters(
            net, b2, b3, length_km=linha_length,
            r_ohm_per_km=linha_r, x_ohm_per_km=linha_x,
            c_nf_per_km=linha_c*1e6, max_i_ka=linha_imax, name="LT"
        )
        
        pp.create_load(net, b3, p_mw=carga_p, q_mvar=carga_q, name="Carga")
        
        return net
    
    else:
        return pn.case9()

net = create_network(network_type)

# ========== DIAGRAMA UNIFILAR ==========
st.header("📐 Diagrama Unifilar do Sistema")

def draw_system_diagram(network_type):
    if network_type == "Rede simples 3 barras":
        fig = go.Figure()
        
        fig.add_shape(type="circle", x0=0.5, y0=4.5, x1=1.5, y1=5.5,
                     line=dict(color="black", width=2))
        fig.add_annotation(x=1, y=5, text="~", showarrow=False, font=dict(size=20))
        
        for i, x in enumerate([2, 5, 8]):
            fig.add_shape(type="line", x0=x, y0=4, x1=x, y1=6,
                         line=dict(color="black", width=3))
            fig.add_annotation(x=x, y=6.3, text=f"{i+1}", showarrow=False, font=dict(size=14))
        
        fig.add_shape(type="line", x0=1.5, y0=5, x1=2, y1=5, line=dict(color="black", width=2))
        fig.add_shape(type="line", x0=2, y0=5, x1=5, y1=5, line=dict(color="black", width=2))
        fig.add_shape(type="line", x0=5, y0=5, x1=8, y1=5, line=dict(color="black", width=2))
        
        fig.add_annotation(x=5, y=3.5, text="P+jQ", showarrow=True, 
                          arrowhead=2, arrowsize=1, arrowwidth=2, ax=0, ay=30)
        fig.add_annotation(x=8, y=3.5, text="P+jQ", showarrow=True,
                          arrowhead=2, arrowsize=1, arrowwidth=2, ax=0, ay=30)
        
        fig.update_layout(
            showlegend=False, height=300,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 9]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[3, 7])
        )
        
        return fig
    
    elif network_type == "Rede com 2 Transformadores":
        fig = go.Figure()
        
        fig.add_shape(type="circle", x0=0.3, y0=4.5, x1=1.3, y1=5.5,
                     line=dict(color="black", width=2))
        fig.add_annotation(x=0.8, y=5, text="~", showarrow=False, font=dict(size=18))
        fig.add_annotation(x=0.8, y=3.8, text="gerador", showarrow=False, font=dict(size=10))
        fig.add_annotation(x=0.8, y=3.4, text=f"|V|={tensao_gerador}kV", showarrow=False, font=dict(size=9))
        
        fig.add_shape(type="line", x0=2, y0=4, x1=2, y1=6, line=dict(color="black", width=3))
        fig.add_annotation(x=2, y=6.3, text="1", showarrow=False, font=dict(size=14))
        
        fig.add_shape(type="circle", x0=2.7, y0=4.7, x1=3.3, y1=5.3, line=dict(color="black", width=2))
        fig.add_shape(type="circle", x0=3.3, y0=4.7, x1=3.9, y1=5.3, line=dict(color="black", width=2))
        fig.add_annotation(x=3.3, y=3.8, text="trafo 1", showarrow=False, font=dict(size=10))
        fig.add_annotation(x=3.3, y=3.4, text=f"{trafo1_vn_lv}kV/{trafo1_vn_hv}kV", 
                          showarrow=False, font=dict(size=8))
        fig.add_annotation(x=3.3, y=3.0, text=f"X={trafo1_vk}%", showarrow=False, font=dict(size=8))
        fig.add_annotation(x=3.3, y=2.6, text=f"{trafo1_sn}MVA", showarrow=False, font=dict(size=8))
        
        fig.add_shape(type="line", x0=4.5, y0=4, x1=4.5, y1=6, line=dict(color="black", width=3))
        fig.add_annotation(x=4.5, y=6.3, text="2", showarrow=False, font=dict(size=14))
        
        fig.add_shape(type="line", x0=4.5, y0=5, x1=7.5, y1=5, line=dict(color="black", width=2))
        fig.add_annotation(x=6, y=5.5, text=f"{linha_length} km", showarrow=False, font=dict(size=9))
        fig.add_annotation(x=6, y=4.7, text=f"X={linha_x} Ohm/km", showarrow=False, font=dict(size=8))
        fig.add_annotation(x=6, y=4.3, text=f"Xc={linha_c} kOhm x km", showarrow=False, font=dict(size=8))
        
        fig.add_shape(type="line", x0=7.5, y0=4, x1=7.5, y1=6, line=dict(color="black", width=3))
        fig.add_annotation(x=7.5, y=6.3, text="3", showarrow=False, font=dict(size=14))
        
        fig.add_shape(type="circle", x0=8.2, y0=4.7, x1=8.8, y1=5.3, line=dict(color="black", width=2))
        fig.add_shape(type="circle", x0=8.8, y0=4.7, x1=9.4, y1=5.3, line=dict(color="black", width=2))
        fig.add_annotation(x=8.8, y=3.8, text="trafo 2", showarrow=False, font=dict(size=10))
        fig.add_annotation(x=8.8, y=3.4, text=f"{trafo2_vn_hv}kV/{trafo2_vn_lv}kV",
                          showarrow=False, font=dict(size=8))
        fig.add_annotation(x=8.8, y=3.0, text=f"X={trafo2_vk}%", showarrow=False, font=dict(size=8))
        fig.add_annotation(x=8.8, y=2.6, text=f"{trafo2_sn}MVA", showarrow=False, font=dict(size=8))
        
        fig.add_shape(type="line", x0=10, y0=4, x1=10, y1=6, line=dict(color="black", width=3))
        fig.add_annotation(x=10, y=6.3, text="4", showarrow=False, font=dict(size=14))
        
        fig.add_annotation(x=10.5, y=5, text="carga", showarrow=False, font=dict(size=10))
        fig.add_annotation(x=11, y=4.7, text=f"{carga_p} MW", showarrow=True,
                          arrowhead=2, ax=-20, ay=0, font=dict(size=9))
        fig.add_annotation(x=11, y=4.3, text=f"cos={carga_fp}", showarrow=False, font=dict(size=8))
        fig.add_annotation(x=11, y=3.9, text="(atrasado)", showarrow=False, font=dict(size=8))
        
        fig.add_shape(type="line", x0=1.3, y0=5, x1=2, y1=5, line=dict(color="black", width=2))
        fig.add_shape(type="line", x0=2, y0=5, x1=2.7, y1=5, line=dict(color="black", width=2))
        fig.add_shape(type="line", x0=3.9, y0=5, x1=4.5, y1=5, line=dict(color="black", width=2))
        fig.add_shape(type="line", x0=7.5, y0=5, x1=8.2, y1=5, line=dict(color="black", width=2))
        fig.add_shape(type="line", x0=9.4, y0=5, x1=10, y1=5, line=dict(color="black", width=2))
        fig.add_shape(type="line", x0=10, y0=5, x1=10.5, y1=5, line=dict(color="black", width=2))
        
        fig.update_layout(
            showlegend=False, height=350,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 12]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[2, 7])
        )
        
        return fig
    
    elif network_type == "Rede com Transformador Único":
        fig = go.Figure()
        
        fig.add_shape(type="circle", x0=0.3, y0=4.5, x1=1.3, y1=5.5,
                     line=dict(color="black", width=2))
        fig.add_annotation(x=0.8, y=5, text="~", showarrow=False, font=dict(size=18))
        fig.add_annotation(x=0.8, y=3.8, text="gerador", showarrow=False, font=dict(size=10))
        fig.add_annotation(x=0.8, y=3.4, text=f"|V|={tensao_gerador}kV", showarrow=False, font=dict(size=9))
        
        fig.add_shape(type="line", x0=2, y0=4, x1=2, y1=6, line=dict(color="black", width=3))
        fig.add_annotation(x=2, y=6.3, text="1", showarrow=False, font=dict(size=14))
        
        fig.add_shape(type="circle", x0=2.7, y0=4.7, x1=3.3, y1=5.3, line=dict(color="black", width=2))
        fig.add_shape(type="circle", x0=3.3, y0=4.7, x1=3.9, y1=5.3, line=dict(color="black", width=2))
        fig.add_annotation(x=3.3, y=3.8, text=f"{trafo1_vn_lv}kV/{trafo1_vn_hv}kV",
                          showarrow=False, font=dict(size=8))
        fig.add_annotation(x=3.3, y=3.4, text=f"X={trafo1_vk}%", showarrow=False, font=dict(size=8))
        fig.add_annotation(x=3.3, y=3.0, text=f"{trafo1_sn}MVA", showarrow=False, font=dict(size=8))
        
        fig.add_shape(type="line", x0=4.5, y0=4, x1=4.5, y1=6, line=dict(color="black", width=3))
        fig.add_annotation(x=4.5, y=6.3, text="2", showarrow=False, font=dict(size=14))
        
        fig.add_shape(type="line", x0=4.5, y0=5, x1=8, y1=5, line=dict(color="black", width=2))
        fig.add_annotation(x=6.2, y=5.5, text=f"{linha_length} km", showarrow=False, font=dict(size=9))
        fig.add_annotation(x=6.2, y=4.7, text=f"X={linha_x} Ohm/km", showarrow=False, font=dict(size=8))
        fig.add_annotation(x=6.2, y=4.3, text=f"Xc={linha_c} kOhm x km", showarrow=False, font=dict(size=8))
        
        fig.add_shape(type="line", x0=8, y0=4, x1=8, y1=6, line=dict(color="black", width=3))
        fig.add_annotation(x=8, y=6.3, text="3", showarrow=False, font=dict(size=14))
        
        fig.add_annotation(x=8.5, y=5, text="carga", showarrow=False, font=dict(size=10))
        fig.add_annotation(x=9, y=4.7, text=f"{carga_p} MW", showarrow=True,
                          arrowhead=2, ax=-20, ay=0, font=dict(size=9))
        fig.add_annotation(x=9, y=4.3, text=f"fp={carga_fp} atrasado", showarrow=False, font=dict(size=8))
        
        fig.add_shape(type="line", x0=1.3, y0=5, x1=2, y1=5, line=dict(color="black", width=2))
        fig.add_shape(type="line", x0=2, y0=5, x1=2.7, y1=5, line=dict(color="black", width=2))
        fig.add_shape(type="line", x0=3.9, y0=5, x1=4.5, y1=5, line=dict(color="black", width=2))
        fig.add_shape(type="line", x0=8, y0=5, x1=8.5, y1=5, line=dict(color="black", width=2))
        
        fig.update_layout(
            showlegend=False, height=350,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 10]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[2, 7])
        )
        
        return fig
    
    return None

diagram = draw_system_diagram(network_type)
if diagram:
    st.plotly_chart(diagram, use_container_width=True)
else:
    st.info("Diagrama não disponível para esta rede")

# ========== SELEÇÃO DE ELEMENTOS ==========
st.sidebar.header("🔍 Seleção de Elementos")

elementos_selecionados = st.sidebar.multiselect(
    "Selecione os elementos para análise",
    ["Barras", "Linhas", "Transformadores", "Geradores", "Cargas"],
    default=["Barras", "Linhas"]
)

# ========== PARÂMETROS CALCULADOS ==========
st.sidebar.header("📊 Parâmetros Calculados")

parametros_barras = st.sidebar.multiselect(
    "Parâmetros das Barras",
    ["Tensão (pu)", "Ângulo (graus)", "Potência Ativa", "Potência Reativa"],
    default=["Tensão (pu)", "Ângulo (graus)"]
)

parametros_linhas = st.sidebar.multiselect(
    "Parâmetros das Linhas",
    ["Potência Ativa", "Potência Reativa", "Carregamento (%)", "Perdas"],
    default=["Carregamento (%)"]
)

# ========== ANÁLISE GRÁFICA ==========
st.sidebar.header("📈 Análise Gráfica")

graficos_selecionados = st.sidebar.multiselect(
    "Selecione os gráficos",
    ["Perfil de Tensão", "Carregamento de Linhas", "Curvas de Pertinência", 
     "Análise TR2E sob Carga", "Fluxo de Potência"],
    default=["Perfil de Tensão"]
)

# ========== EXECUÇÃO DO FLUXO DE CARGA ==========
col1, col2 = st.columns([1, 3])

with col1:
    executar = st.button("▶️ Executar Fluxo de Carga", use_container_width=True)

with col2:
    if executar:
        try:
            pp.runpp(net)
            st.success("✅ Fluxo de carga executado com sucesso!")
        except Exception as e:
            st.error(f"❌ Erro na execução: {str(e)}")
            executar = False

# ========== EXIBIÇÃO DOS RESULTADOS ==========
if executar:
    
    if "Barras" in elementos_selecionados:
        st.subheader("🔌 Tensões nas Barras")
        
        colunas_barras = []
        if "Tensão (pu)" in parametros_barras:
            colunas_barras.append("vm_pu")
        if "Ângulo (graus)" in parametros_barras:
            colunas_barras.append("va_degree")
        if "Potência Ativa" in parametros_barras:
            colunas_barras.append("p_mw")
        if "Potência Reativa" in parametros_barras:
            colunas_barras.append("q_mvar")
        
        if colunas_barras:
            df_barras = net.res_bus[colunas_barras].copy()
            df_barras.index = net.bus['name'].values
            st.dataframe(df_barras, use_container_width=True)
    
    if "Linhas" in elementos_selecionados and len(net.line) > 0:
        st.subheader("⚡ Carregamento das Linhas")
        
        colunas_linhas = []
        if "Potência Ativa" in parametros_linhas:
            colunas_linhas.extend(["p_from_mw", "p_to_mw"])
        if "Potência Reativa" in parametros_linhas:
            colunas_linhas.extend(["q_from_mvar", "q_to_mvar"])
        if "Carregamento (%)" in parametros_linhas:
            colunas_linhas.append("loading_percent")
        if "Perdas" in parametros_linhas:
            colunas_linhas.append("pl_mw")
        
        if colunas_linhas:
            df_linhas = net.res_line[colunas_linhas].copy()
            if 'name' in net.line.columns:
                df_linhas.index = net.line['name'].values
            st.dataframe(df_linhas, use_container_width=True)
    
    if "Transformadores" in elementos_selecionados and len(net.trafo) > 0:
        st.subheader("🔄 Transformadores")
        
        df_trafo = net.res_trafo[["p_hv_mw", "q_hv_mvar", "loading_percent"]].copy()
        if 'name' in net.trafo.columns:
            df_trafo.index = net.trafo['name'].values
        st.dataframe(df_trafo, use_container_width=True)
    
    if "Geradores" in elementos_selecionados:
        st.subheader("⚙️ Geradores")
        df_gen = net.res_ext_grid[["p_mw", "q_mvar"]].copy()
        st.dataframe(df_gen, use_container_width=True)
    
    if "Cargas" in elementos_selecionados and len(net.load) > 0:
        st.subheader("💡 Cargas")
        df_cargas = pd.DataFrame({
            'P (MW)': net.load['p_mw'].values,
            'Q (Mvar)': net.load['q_mvar'].values
        })
        if 'name' in net.load.columns:
            df_cargas.index = net.load['name'].values
        st.dataframe(df_cargas, use_container_width=True)
    
    st.header("📊 Análises Gráficas")
    
    if "Perfil de Tensão" in graficos_selecionados:
        st.subheader("📈 Perfil de Tensão nas Barras")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(net.res_bus))),
            y=net.res_bus['vm_pu'].values,
            mode='lines+markers',
            name='Tensão (pu)',
            line=dict(color='blue', width=2),
            marker=dict(size=8)
        ))
        
        fig.add_hline(y=1.05, line_dash="dash", line_color="red", 
                     annotation_text="Limite Superior")
        fig.add_hline(y=0.95, line_dash="dash", line_color="red", 
                     annotation_text="Limite Inferior")
        
        fig.update_layout(
            xaxis_title="Barra",
            yaxis_title="Tensão (pu)",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    if "Carregamento de Linhas" in graficos_selecionados and len(net.line) > 0:
        st.subheader("⚡ Carregamento das Linhas")
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[f"Linha {i}" for i in range(len(net.res_line))],
            y=net.res_line['loading_percent'].values,
            marker_color='orange'
        ))
        
        fig.add_hline(y=100, line_dash="dash", line_color="red", 
                     annotation_text="Limite 100%")
        
        fig.update_layout(
            xaxis_title="Linha",
            yaxis_title="Carregamento (%)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    if "Curvas de Pertinência" in graficos_selecionados:
        st.subheader("📉 Funções de Pertinência - Lógica Fuzzy")
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('I_d', 'I_d*+I_f', 'I_d2', 'I_dE', 'Tipo', ''),
            specs=[[{}, {}, {}], [{}, {}, None]]
        )
        
        x = np.linspace(0, 200, 100)
        baixa = np.maximum(0, np.minimum(1, (22 - x) / 4))
        alta = np.maximum(0, np.minimum(1, (x - 18) / 4))
        
        fig.add_trace(go.Scatter(x=x, y=baixa, name='Baixa', 
                                line=dict(color='blue')), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=alta, name='Alta', 
                                line=dict(color='green')), row=1, col=1)
        
        x2 = np.linspace(0, 2, 100)
        baixa2 = np.maximum(0, np.minimum(1, (0.35 - x2) / 0.1))
        alta2 = np.maximum(0, np.minimum(1, (x2 - 0.25) / 0.1))
        
        fig.add_trace(go.Scatter(x=x2, y=baixa2, name='Baixa', 
                                line=dict(color='blue'), showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=x2, y=alta2, name='Alta', 
                                line=dict(color='green'), showlegend=False), row=1, col=2)
        
        x3 = np.linspace(0, 100, 100)
        baixa3 = np.maximum(0, np.minimum(1, (20 - x3) / 10))
        alta3 = np.maximum(0, np.minimum(1, (x3 - 10) / 10))
        
        fig.add_trace(go.Scatter(x=x3, y=baixa3, name='Baixa', 
                                line=dict(color='blue'), showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=x3, y=alta3, name='Alta', 
                                line=dict(color='green'), showlegend=False), row=2, col=1)
        
        x4 = np.linspace(0, 10, 100)
        baixa4 = np.maximum(0, np.minimum(1, (20 - x4) / 10))
        alta4 = np.maximum(0, np.minimum(1, (x4 - 10) / 10))
        
        fig.add_trace(go.Scatter(x=x4, y=baixa4, name='Baixa', 
                                line=dict(color='blue'), showlegend=False), row=2, col=2)
        fig.add_trace(go.Scatter(x=x4, y=alta4, name='Alta', 
                                line=dict(color='green'), showlegend=False), row=2, col=2)
        
        tipos = ['EV', 'EC', 'EP', 'ED']
        valores = [0, 1, 2, 3]
        cores = ['green', 'blue', 'cyan', 'yellow']
        
        for i, (tipo, valor, cor) in enumerate(zip(tipos, valores, cores)):
            fig.add_trace(go.Scatter(
                x=[valor, valor], y=[0, 1], 
                mode='lines', name=tipo,
                line=dict(color=cor, width=3)
            ), row=1, col=3)
        
        fig.update_layout(height=600, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    if "Análise TR2E sob Carga" in graficos_selecionados:
        st.subheader("🔄 Análise de TR2E - Tempo sob Diferentes Ângulos")
        
        angulos = [270, 285, 300, 315, 330, 345, 0, 15, 30, 45, 60, 75, 90]
        tempos = [3.6, 3.5, 3.5, 3.3, 3.4, 3.3, 3.3, 3.3, 3.5, 3.5, 3.5, 3.6, 3.6]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=angulos, y=tempos,
            mode='lines+markers',
            line=dict(color='black', width=2, dash='dot'),
            marker=dict(color='darkblue', size=10, symbol='diamond')
        ))
        
        fig.update_layout(
            title="Análise de TR2E para Situações de Energização sob Carga",
            xaxis_title="Ângulo (°)",
            yaxis_title="Tempo (ms)",
            height=400,
            yaxis=dict(range=[3.1, 3.7])
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    if "Fluxo de Potência" in graficos_selecionados:
        st.subheader("🔀 Fluxo de Potência no Sistema")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Potência Total Gerada", 
                     f"{net.res_ext_grid['p_mw'].sum():.2f} MW")
            st.metric("Potência Total Consumida", 
                     f"{net.res_load['p_mw'].sum():.2f} MW")
        
        with col2:
            st.metric("Perdas Totais", 
                     f"{net.res_line['pl_mw'].sum():.3f} MW" if len(net.line) > 0 else "N/A")
            st.metric("Eficiência", 
                     f"{(net.res_load['p_mw'].sum() / net.res_ext_grid['p_mw'].sum() * 100):.2f} %")

st.sidebar.markdown("---")
st.sidebar.info("💡 Desenvolvido com Streamlit e pandapower")
