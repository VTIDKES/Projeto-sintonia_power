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
    ["Rede simples 3 barras", "Rede IEEE 9 barras", "Rede Personalizada"]
)

# ========== CRIAÇÃO DA REDE ==========
def create_network(network_type):
    if network_type == "Rede simples 3 barras":
        net = pp.create_empty_network()
        
        b1 = pp.create_bus(net, vn_kv=13.8, name="Barra 1")
        b2 = pp.create_bus(net, vn_kv=13.8, name="Barra 2")
        b3 = pp.create_bus(net, vn_kv=13.8, name="Barra 3")
        
        pp.create_ext_grid(net, bus=b1, vm_pu=1.02, name="Gerador Síncrono")
        
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
        pp.create_load(net, b3, p_mw=1.5, q_mvar=0.6, name="Carga 2")
        
        return net
    
    elif network_type == "Rede Personalizada":
        net = pp.create_empty_network()
        
        # Criação das barras do sistema da imagem 2
        bger = pp.create_bus(net, vn_kv=13.8, name="GER")
        bger_barra = pp.create_bus(net, vn_kv=13.8, name="BGER")
        
        blt1 = pp.create_bus(net, vn_kv=138, name="BLT1")
        blt1o = pp.create_bus(net, vn_kv=138, name="BLT1O")
        
        blt2 = pp.create_bus(net, vn_kv=138, name="BLT2")
        blt2o = pp.create_bus(net, vn_kv=138, name="BLT2O")
        
        blt3 = pp.create_bus(net, vn_kv=138, name="BLT3")
        blt3o = pp.create_bus(net, vn_kv=138, name="BLT3O")
        
        bgch1 = pp.create_bus(net, vn_kv=13.8, name="BGCH1")
        bgch2 = pp.create_bus(net, vn_kv=13.8, name="BGCH2")
        bgch3 = pp.create_bus(net, vn_kv=13.8, name="BGCH3")
        bgchm = pp.create_bus(net, vn_kv=13.8, name="BGCHM")
        
        # Gerador síncrono
        pp.create_ext_grid(net, bus=bger, vm_pu=1.0, name="GER 90MVA")
        
        # Transformadores elevadores
        pp.create_transformer_from_parameters(
            net, bger_barra, blt1, sn_mva=25, vn_hv_kv=138, vn_lv_kv=13.8,
            vk_percent=6, vkr_percent=0.5, pfe_kw=0, i0_percent=0, name="TR1E"
        )
        pp.create_transformer_from_parameters(
            net, bger_barra, blt2, sn_mva=25, vn_hv_kv=138, vn_lv_kv=13.8,
            vk_percent=6, vkr_percent=0.5, pfe_kw=0, i0_percent=0, name="TR2E"
        )
        pp.create_transformer_from_parameters(
            net, bger_barra, blt3, sn_mva=25, vn_hv_kv=138, vn_lv_kv=13.8,
            vk_percent=6, vkr_percent=0.5, pfe_kw=0, i0_percent=0, name="TR3E"
        )
        
        # Linhas de transmissão
        pp.create_line_from_parameters(
            net, blt1, blt1o, length_km=100,
            r_ohm_per_km=0.05, x_ohm_per_km=0.4,
            c_nf_per_km=10, max_i_ka=1.0, name="LT1"
        )
        pp.create_line_from_parameters(
            net, blt2, blt2o, length_km=50,
            r_ohm_per_km=0.05, x_ohm_per_km=0.4,
            c_nf_per_km=10, max_i_ka=1.0, name="LT2"
        )
        pp.create_line_from_parameters(
            net, blt3, blt3o, length_km=50,
            r_ohm_per_km=0.05, x_ohm_per_km=0.4,
            c_nf_per_km=10, max_i_ka=1.0, name="LT3"
        )
        
        # Transformadores abaixadores
        pp.create_transformer_from_parameters(
            net, blt1o, bgch1, sn_mva=25, vn_hv_kv=138, vn_lv_kv=13.8,
            vk_percent=6, vkr_percent=0.5, pfe_kw=0, i0_percent=0, name="TR1A"
        )
        pp.create_transformer_from_parameters(
            net, blt2o, bgch2, sn_mva=25, vn_hv_kv=138, vn_lv_kv=13.8,
            vk_percent=6, vkr_percent=0.5, pfe_kw=0, i0_percent=0, name="TR2A"
        )
        pp.create_transformer_from_parameters(
            net, blt3o, bgch3, sn_mva=25, vn_hv_kv=138, vn_lv_kv=13.8,
            vk_percent=6, vkr_percent=0.5, pfe_kw=0, i0_percent=0, name="TR3A"
        )
        
        # Cargas
        pp.create_load(net, bgch1, p_mw=5.0, q_mvar=2.0, name="Alimentador 1")
        pp.create_load(net, bgch2, p_mw=4.0, q_mvar=1.5, name="Alimentador 2")
        pp.create_load(net, bgch3, p_mw=3.0, q_mvar=1.0, name="Alimentador 3")
        
        # Motor
        pp.create_load(net, bgchm, p_mw=2.0, q_mvar=1.5, name="Motor")
        
        return net
    
    else:
        return pn.case9()

net = create_network(network_type)

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
    
    # BARRAS
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
    
    # LINHAS
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
    
    # TRANSFORMADORES
    if "Transformadores" in elementos_selecionados and len(net.trafo) > 0:
        st.subheader("🔄 Transformadores")
        
        df_trafo = net.res_trafo[["p_hv_mw", "q_hv_mvar", "loading_percent"]].copy()
        if 'name' in net.trafo.columns:
            df_trafo.index = net.trafo['name'].values
        st.dataframe(df_trafo, use_container_width=True)
    
    # GERADORES
    if "Geradores" in elementos_selecionados:
        st.subheader("⚙️ Geradores")
        df_gen = net.res_ext_grid[["p_mw", "q_mvar"]].copy()
        st.dataframe(df_gen, use_container_width=True)
    
    # CARGAS
    if "Cargas" in elementos_selecionados and len(net.load) > 0:
        st.subheader("💡 Cargas")
        df_cargas = pd.DataFrame({
            'P (MW)': net.load['p_mw'].values,
            'Q (Mvar)': net.load['q_mvar'].values
        })
        if 'name' in net.load.columns:
            df_cargas.index = net.load['name'].values
        st.dataframe(df_cargas, use_container_width=True)
    
    # ========== GRÁFICOS ==========
    st.header("📊 Análises Gráficas")
    
    # PERFIL DE TENSÃO
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
    
    # CARREGAMENTO DE LINHAS
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
    
    # CURVAS DE PERTINÊNCIA (Exemplo baseado na imagem 3)
    if "Curvas de Pertinência" in graficos_selecionados:
        st.subheader("📉 Funções de Pertinência - Lógica Fuzzy")
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('I_d', 'I_d*+I_f', 'I_d2', 'I_dE', 'Tipo', ''),
            specs=[[{}, {}, {}], [{}, {}, None]]
        )
        
        # Curva Id
        x = np.linspace(0, 200, 100)
        baixa = np.maximum(0, np.minimum(1, (22 - x) / 4))
        alta = np.maximum(0, np.minimum(1, (x - 18) / 4))
        
        fig.add_trace(go.Scatter(x=x, y=baixa, name='Baixa', 
                                line=dict(color='blue')), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=alta, name='Alta', 
                                line=dict(color='green')), row=1, col=1)
        
        # Curva Id*+If
        x2 = np.linspace(0, 2, 100)
        baixa2 = np.maximum(0, np.minimum(1, (0.35 - x2) / 0.1))
        alta2 = np.maximum(0, np.minimum(1, (x2 - 0.25) / 0.1))
        
        fig.add_trace(go.Scatter(x=x2, y=baixa2, name='Baixa', 
                                line=dict(color='blue'), showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=x2, y=alta2, name='Alta', 
                                line=dict(color='green'), showlegend=False), row=1, col=2)
        
        # Curva Id2
        x3 = np.linspace(0, 100, 100)
        baixa3 = np.maximum(0, np.minimum(1, (20 - x3) / 10))
        alta3 = np.maximum(0, np.minimum(1, (x3 - 10) / 10))
        
        fig.add_trace(go.Scatter(x=x3, y=baixa3, name='Baixa', 
                                line=dict(color='blue'), showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=x3, y=alta3, name='Alta', 
                                line=dict(color='green'), showlegend=False), row=2, col=1)
        
        # Curva IdE
        x4 = np.linspace(0, 10, 100)
        baixa4 = np.maximum(0, np.minimum(1, (20 - x4) / 10))
        alta4 = np.maximum(0, np.minimum(1, (x4 - 10) / 10))
        
        fig.add_trace(go.Scatter(x=x4, y=baixa4, name='Baixa', 
                                line=dict(color='blue'), showlegend=False), row=2, col=2)
        fig.add_trace(go.Scatter(x=x4, y=alta4, name='Alta', 
                                line=dict(color='green'), showlegend=False), row=2, col=2)
        
        # Tipo
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
    
    # ANÁLISE TR2E (baseado na imagem 1)
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
    
    # FLUXO DE POTÊNCIA
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
