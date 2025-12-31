# STREAMLIT + PANDAPOWER - Análise Completa
# Sistemas Elétricos de Potência
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import pandapower as pp

st.set_page_config(page_title="Power System Studio", layout="wide")

# =========================
# Estado inicial
# =========================
if "barras" not in st.session_state:
    st.session_state.barras = pd.DataFrame(columns=["name", "x", "y", "tipo", "vn_kv"])
if "linhas" not in st.session_state:
    st.session_state.linhas = pd.DataFrame(columns=["from", "to", "length_km"])
if "cargas" not in st.session_state:
    st.session_state.cargas = pd.DataFrame(columns=["barra", "p_mw", "q_mvar"])
if "geradores" not in st.session_state:
    st.session_state.geradores = pd.DataFrame(columns=["barra", "p_mw", "vm_pu"])

# =========================
# Cabeçalho
# =========================
st.title("⚡ Power System Studio v2.0")
st.markdown("**Desenho Interativo de Redes Elétricas com Pandapower**")

# =========================
# Layout em colunas
# =========================
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🔧 Ferramentas")
    
    # Tabs para organizar elementos
    tab1, tab2, tab3, tab4 = st.tabs(["🔵 Barras", "➖ Linhas", "📊 Cargas", "⚙️ Geradores"])
    
    # TAB 1: Adicionar Barras
    with tab1:
        st.markdown("#### Adicionar Barra")
        barra_name = st.text_input("Nome da Barra", key="barra_name")
        tipo_barra = st.selectbox("Tipo", ["PQ", "PV", "Swing"], key="tipo_barra")
        vn_kv = st.number_input("Tensão Nominal (kV)", value=110.0, min_value=0.1, key="vn_kv")
        
        col_x, col_y = st.columns(2)
        with col_x:
            x_coord = st.number_input("Posição X", value=0.0, key="x_coord")
        with col_y:
            y_coord = st.number_input("Posição Y", value=0.0, key="y_coord")
        
        if st.button("➕ Adicionar Barra", use_container_width=True):
            if barra_name.strip() != "":
                new_barra = pd.DataFrame({
                    "name": [barra_name],
                    "x": [x_coord],
                    "y": [y_coord],
                    "tipo": [tipo_barra],
                    "vn_kv": [vn_kv]
                })
                st.session_state.barras = pd.concat([st.session_state.barras, new_barra], ignore_index=True)
                st.success(f"✅ Barra '{barra_name}' adicionada!")
                st.rerun()
            else:
                st.error("❌ Nome da barra não pode ser vazio")
    
    # TAB 2: Conectar Linhas
    with tab2:
        st.markdown("#### Conectar Barras")
        if len(st.session_state.barras) >= 2:
            from_barra = st.selectbox("De", st.session_state.barras["name"], key="from_barra")
            to_barra = st.selectbox("Para", st.session_state.barras["name"], key="to_barra")
            length_km = st.number_input("Comprimento (km)", value=5.0, min_value=0.1, key="length_km")
            
            if st.button("➕ Adicionar Linha", use_container_width=True):
                if from_barra != to_barra:
                    new_linha = pd.DataFrame({
                        "from": [from_barra],
                        "to": [to_barra],
                        "length_km": [length_km]
                    })
                    st.session_state.linhas = pd.concat([st.session_state.linhas, new_linha], ignore_index=True)
                    st.success(f"✅ Linha {from_barra} → {to_barra} adicionada!")
                    st.rerun()
                else:
                    st.error("❌ Barras devem ser diferentes")
        else:
            st.info("ℹ️ Adicione pelo menos 2 barras primeiro")
    
    # TAB 3: Adicionar Cargas
    with tab3:
        st.markdown("#### Adicionar Carga")
        if len(st.session_state.barras) > 0:
            barra_carga = st.selectbox("Barra", st.session_state.barras["name"], key="barra_carga")
            p_mw = st.number_input("Potência Ativa (MW)", value=10.0, key="p_mw")
            q_mvar = st.number_input("Potência Reativa (MVAr)", value=5.0, key="q_mvar")
            
            if st.button("➕ Adicionar Carga", use_container_width=True):
                new_carga = pd.DataFrame({
                    "barra": [barra_carga],
                    "p_mw": [p_mw],
                    "q_mvar": [q_mvar]
                })
                st.session_state.cargas = pd.concat([st.session_state.cargas, new_carga], ignore_index=True)
                st.success(f"✅ Carga adicionada na barra '{barra_carga}'")
                st.rerun()
        else:
            st.info("ℹ️ Adicione barras primeiro")
    
    # TAB 4: Adicionar Geradores
    with tab4:
        st.markdown("#### Adicionar Gerador")
        if len(st.session_state.barras) > 0:
            barra_gen = st.selectbox("Barra", st.session_state.barras["name"], key="barra_gen")
            p_mw_gen = st.number_input("Potência (MW)", value=50.0, key="p_mw_gen")
            vm_pu = st.number_input("Tensão (pu)", value=1.02, min_value=0.9, max_value=1.1, key="vm_pu")
            
            if st.button("➕ Adicionar Gerador", use_container_width=True):
                new_gen = pd.DataFrame({
                    "barra": [barra_gen],
                    "p_mw": [p_mw_gen],
                    "vm_pu": [vm_pu]
                })
                st.session_state.geradores = pd.concat([st.session_state.geradores, new_gen], ignore_index=True)
                st.success(f"✅ Gerador adicionado na barra '{barra_gen}'")
                st.rerun()
        else:
            st.info("ℹ️ Adicione barras primeiro")
    
    # Botões de ação
    st.markdown("---")
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🗑️ Limpar Tudo", use_container_width=True):
            st.session_state.barras = pd.DataFrame(columns=["name", "x", "y", "tipo", "vn_kv"])
            st.session_state.linhas = pd.DataFrame(columns=["from", "to", "length_km"])
            st.session_state.cargas = pd.DataFrame(columns=["barra", "p_mw", "q_mvar"])
            st.session_state.geradores = pd.DataFrame(columns=["barra", "p_mw", "vm_pu"])
            st.rerun()
    
    with col_btn2:
        if st.button("⚡ Calcular Fluxo", use_container_width=True, type="primary"):
            st.session_state.run_power_flow = True

# =========================
# Coluna 2: Visualização
# =========================
with col2:
    st.subheader("📊 Diagrama Unifilar")
    
    # Criar figura
    fig = go.Figure()
    
    # Desenhar linhas
    for idx, linha in st.session_state.linhas.iterrows():
        b_from = st.session_state.barras[st.session_state.barras["name"] == linha["from"]]
        b_to = st.session_state.barras[st.session_state.barras["name"] == linha["to"]]
        
        if not b_from.empty and not b_to.empty:
            fig.add_trace(go.Scatter(
                x=[b_from.iloc[0]["x"], b_to.iloc[0]["x"]],
                y=[b_from.iloc[0]["y"], b_to.iloc[0]["y"]],
                mode="lines",
                line=dict(color="gray", width=3),
                hoverinfo="text",
                hovertext=f"Linha: {linha['from']} → {linha['to']}<br>Comprimento: {linha['length_km']} km",
                showlegend=False
            ))
    
    # Desenhar barras
    if not st.session_state.barras.empty:
        cores = {"PQ": "blue", "PV": "green", "Swing": "red"}
        
        fig.add_trace(go.Scatter(
            x=st.session_state.barras["x"],
            y=st.session_state.barras["y"],
            mode="markers+text",
            marker=dict(
                size=25, 
                color=[cores.get(t, "blue") for t in st.session_state.barras["tipo"]],
                line=dict(width=2, color="white")
            ),
            text=st.session_state.barras["name"],
            textposition="top center",
            textfont=dict(size=12, color="black"),
            hoverinfo="text",
            hovertext=[f"Barra: {row['name']}<br>Tipo: {row['tipo']}<br>Tensão: {row['vn_kv']} kV" 
                      for _, row in st.session_state.barras.iterrows()],
            showlegend=False
        ))
    
    # Atualizar layout
    fig.update_layout(
        height=600,
        xaxis=dict(title="X (m)", showgrid=True, zeroline=True),
        yaxis=dict(title="Y (m)", showgrid=True, zeroline=True),
        hovermode='closest',
        plot_bgcolor='#f0f2f6',
        dragmode='pan'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Tabelas de elementos
    with st.expander("📋 Elementos da Rede", expanded=False):
        tab_b, tab_l, tab_c, tab_g = st.tabs(["Barras", "Linhas", "Cargas", "Geradores"])
        
        with tab_b:
            st.dataframe(st.session_state.barras, use_container_width=True)
        with tab_l:
            st.dataframe(st.session_state.linhas, use_container_width=True)
        with tab_c:
            st.dataframe(st.session_state.cargas, use_container_width=True)
        with tab_g:
            st.dataframe(st.session_state.geradores, use_container_width=True)

# =========================
# Cálculo de Fluxo de Potência
# =========================
if st.session_state.get("run_power_flow", False):
    st.session_state.run_power_flow = False
    
    if len(st.session_state.barras) == 0:
        st.error("❌ Adicione barras antes de calcular o fluxo!")
    else:
        try:
            with st.spinner("⚡ Calculando fluxo de potência..."):
                # Criar rede Pandapower
                net = pp.create_empty_network()
                barra_map = {}
                
                # Adicionar barras
                for idx, row in st.session_state.barras.iterrows():
                    barra_map[row["name"]] = pp.create_bus(
                        net, 
                        vn_kv=row["vn_kv"], 
                        name=row["name"]
                    )
                
                # Adicionar ext_grid na primeira barra Swing
                swing_barras = st.session_state.barras[st.session_state.barras["tipo"] == "Swing"]
                if len(swing_barras) > 0:
                    swing_bus = barra_map[swing_barras.iloc[0]["name"]]
                    pp.create_ext_grid(net, bus=swing_bus, vm_pu=1.02, name="Subestação")
                elif len(barra_map) > 0:
                    # Se não há Swing, usar primeira barra
                    first_bus = list(barra_map.values())[0]
                    pp.create_ext_grid(net, bus=first_bus, vm_pu=1.02, name="Subestação")
                
                # Adicionar linhas
                for idx, row in st.session_state.linhas.iterrows():
                    pp.create_line(
                        net, 
                        barra_map[row["from"]], 
                        barra_map[row["to"]], 
                        length_km=row["length_km"], 
                        std_type="NAYY 4x50 SE"
                    )
                
                # Adicionar cargas
                for idx, row in st.session_state.cargas.iterrows():
                    pp.create_load(
                        net,
                        bus=barra_map[row["barra"]],
                        p_mw=row["p_mw"],
                        q_mvar=row["q_mvar"],
                        name=f"Carga_{row['barra']}"
                    )
                
                # Adicionar geradores
                for idx, row in st.session_state.geradores.iterrows():
                    pp.create_gen(
                        net,
                        bus=barra_map[row["barra"]],
                        p_mw=row["p_mw"],
                        vm_pu=row["vm_pu"],
                        name=f"Gerador_{row['barra']}"
                    )
                
                # Executar fluxo de carga
                pp.runpp(net)
                
                st.success("✅ Fluxo de carga calculado com sucesso!")
                
                # Mostrar resultados
                col_res1, col_res2 = st.columns(2)
                
                with col_res1:
                    st.subheader("📊 Tensões nas Barras")
                    res_bus = net.res_bus[["vm_pu", "va_degree"]].copy()
                    res_bus["nome"] = [net.bus.at[i, "name"] for i in res_bus.index]
                    st.dataframe(res_bus[["nome", "vm_pu", "va_degree"]], use_container_width=True)
                
                with col_res2:
                    st.subheader("⚡ Carregamento das Linhas")
                    st.dataframe(net.res_line[["loading_percent", "p_from_mw", "q_from_mvar"]], use_container_width=True)
                
        except Exception as e:
            st.error(f"❌ Erro no cálculo: {str(e)}")

# Rodapé
st.markdown("---")
st.caption("Power System Studio v2.0 | Desenvolvido com Streamlit, Plotly e Pandapower | ⚡ Análise de Sistemas Elétricos de Potência")
