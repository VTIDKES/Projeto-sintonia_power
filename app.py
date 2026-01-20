"""
╔═══════════════════════════════════════════════════════════════════════════╗
║          POWER SYSTEM PROFESSIONAL STUDIO v2.0                            ║
║          Sistema Completo de Análise de Redes Elétricas                   ║
║          COM SELEÇÃO EDITÁVEL DE BARRAS                                   ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

# ═════════════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO DA APLICAÇÃO
# ═════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Power System Professional Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS Customizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1e3c72;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════
# CLASSES E ESTRUTURAS DE DADOS
# ═════════════════════════════════════════════════════════════════════════

class BusType(Enum):
    SLACK = "Slack"
    PV = "PV" 
    PQ = "PQ"

@dataclass
class SystemLimits:
    v_min_pu: float = 0.95
    v_max_pu: float = 1.05
    loading_max_percent: float = 100.0

@dataclass
class ProjectInfo:
    name: str = "Novo Projeto"
    company: str = ""
    engineer: str = ""
    date: str = ""
    description: str = ""

# ═════════════════════════════════════════════════════════════════════════
# BANCO DE DADOS DE EQUIPAMENTOS
# ═════════════════════════════════════════════════════════════════════════

class EquipmentDatabase:
    @staticmethod
    def get_conductors() -> pd.DataFrame:
        return pd.DataFrame([
            {"name": "1/0 AWG (53.5mm²)", "area_mm2": 53.5, "r_ohm_km_ac": 0.323, "ampacity_75C": 230},
            {"name": "2/0 AWG (67.4mm²)", "area_mm2": 67.4, "r_ohm_km_ac": 0.256, "ampacity_75C": 265},
            {"name": "4/0 AWG (107mm²)", "area_mm2": 107.2, "r_ohm_km_ac": 0.161, "ampacity_75C": 350},
            {"name": "336.4 MCM (170mm²)", "area_mm2": 170.0, "r_ohm_km_ac": 0.102, "ampacity_75C": 530},
            {"name": "795 MCM (403mm²)", "area_mm2": 403.0, "r_ohm_km_ac": 0.043, "ampacity_75C": 860},
        ])

# ═════════════════════════════════════════════════════════════════════════
# INICIALIZAÇÃO
# ═════════════════════════════════════════════════════════════════════════

def init_state():
    defaults = {
        "project_info": ProjectInfo(),
        "tables": default_tables(),
        "results": {},
        "net_built": None,
        "system_limits": SystemLimits(),
        "conductor_db": EquipmentDatabase.get_conductors(),
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def default_tables() -> Dict[str, pd.DataFrame]:
    buses = pd.DataFrame([
        {"name": "SE-PRINCIPAL", "vn_kv": 230.0, "type": "Slack", "zone": "Norte", "x_km": 0.0, "y_km": 0.0},
        {"name": "SE-INDUSTRIAL", "vn_kv": 138.0, "type": "PQ", "zone": "Norte", "x_km": 45.0, "y_km": 10.0},
        {"name": "SE-COMERCIAL", "vn_kv": 13.8, "type": "PQ", "zone": "Sul", "x_km": 30.0, "y_km": -15.0},
    ])
    
    ext_grids = pd.DataFrame([{
        "name": "REDE-BASICA", "bus": "SE-PRINCIPAL", "vm_pu": 1.00, "va_degree": 0.0,
        "s_sc_max_mva": 5000, "s_sc_min_mva": 3000
    }])
    
    gens = pd.DataFrame([{
        "name": "GER-INDUSTRIAL", "bus": "SE-INDUSTRIAL", "p_mw": 50.0, "vm_pu": 1.02,
        "min_q_mvar": -30, "max_q_mvar": 40, "type": "Síncrono"
    }])
    
    loads = pd.DataFrame([
        {"name": "CARGA-IND-01", "bus": "SE-INDUSTRIAL", "p_mw": 80.0, "q_mvar": 30.0, "type": "Industrial"},
        {"name": "CARGA-COM-01", "bus": "SE-COMERCIAL", "p_mw": 25.0, "q_mvar": 10.0, "type": "Commercial"},
    ])
    
    lines = pd.DataFrame([{
        "name": "LT-230-01", "from_bus": "SE-PRINCIPAL", "to_bus": "SE-INDUSTRIAL",
        "length_km": 45.0, "r_ohm_per_km": 0.035, "x_ohm_per_km": 0.38,
        "c_nf_per_km": 11.0, "max_i_ka": 1.2
    }])
    
    trafos = pd.DataFrame([{
        "name": "TR-230/138-01", "hv_bus": "SE-PRINCIPAL", "lv_bus": "SE-INDUSTRIAL",
        "sn_mva": 100, "vn_hv_kv": 230, "vn_lv_kv": 138, "vk_percent": 12.0
    }])
    
    switches = pd.DataFrame([{
        "name": "DJ-01", "bus": "SE-PRINCIPAL", "element_type": "l",
        "element_name": "LT-230-01", "closed": True, "type": "CB"
    }])
    
    shunts = pd.DataFrame([{
        "name": "BC-IND-01", "bus": "SE-INDUSTRIAL", "q_mvar": 30.0, "in_service": True
    }])
    
    return {
        "buses": buses, "ext_grids": ext_grids, "gens": gens,
        "loads": loads, "lines": lines, "trafos": trafos,
        "switches": switches, "shunts": shunts
    }

# ═════════════════════════════════════════════════════════════════════════
# VALIDAÇÃO
# ═════════════════════════════════════════════════════════════════════════

def validate_system(tables: Dict) -> Tuple[bool, List[str], List[str]]:
    warnings, errors = [], []
    
    buses = tables["buses"]
    if buses.empty:
        errors.append("❌ Sistema precisa de pelo menos uma barra")
        return False, warnings, errors
    
    if buses["name"].duplicated().any():
        errors.append("❌ Nomes de barras duplicados")
    
    slack_count = (buses["type"] == "Slack").sum()
    if slack_count == 0:
        errors.append("❌ Sistema precisa de pelo menos uma barra Slack")
    
    bus_names = set(buses["name"])
    for table_name, ref_cols in [
        ("ext_grids", ["bus"]),
        ("gens", ["bus"]),
        ("loads", ["bus"]),
        ("lines", ["from_bus", "to_bus"]),
        ("trafos", ["hv_bus", "lv_bus"]),
    ]:
        df = tables.get(table_name, pd.DataFrame())
        for col in ref_cols:
            if col in df.columns:
                invalid = df[~df[col].isin(bus_names)]
                if not invalid.empty:
                    errors.append(f"❌ {table_name}: barras inválidas em '{col}'")
    
    return len(errors) == 0, warnings, errors

# ═════════════════════════════════════════════════════════════════════════
# FUNÇÃO AUXILIAR PARA OBTER LISTA DE BARRAS
# ═════════════════════════════════════════════════════════════════════════

def get_bus_names() -> List[str]:
    """Retorna lista de nomes de barras disponíveis"""
    if "tables" in st.session_state and not st.session_state.tables["buses"].empty:
        return st.session_state.tables["buses"]["name"].tolist()
    return []

# ═════════════════════════════════════════════════════════════════════════
# INTERFACE - SIDEBAR
# ═════════════════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.markdown('<div class="main-header">⚡ PSPS v2.0</div>', unsafe_allow_html=True)
        
        page = st.radio(
            "**Navegação**",
            [
                "🏠 Dashboard",
                "📊 Dados do Sistema",
                "⚙️ Simulações",
                "ℹ️ Sobre"
            ],
            index=0
        )
        
        st.divider()
        
        with st.expander("📋 Informações do Projeto"):
            project = st.session_state.project_info
            st.text_input("Nome", value=project.name, key="proj_name")
            st.text_input("Empresa", value=project.company, key="proj_company")
        
        with st.expander("⚠️ Limites Operacionais"):
            limits = st.session_state.system_limits
            limits.v_min_pu = st.number_input("V mín (pu)", 0.8, 1.0, limits.v_min_pu, 0.01)
            limits.v_max_pu = st.number_input("V máx (pu)", 1.0, 1.2, limits.v_max_pu, 0.01)
        
        return page.split()[1]

# ═════════════════════════════════════════════════════════════════════════
# PÁGINAS
# ═════════════════════════════════════════════════════════════════════════

def page_dashboard():
    st.title("🏠 Dashboard do Sistema")
    
    tables = st.session_state.tables
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Barras", len(tables["buses"]))
    col2.metric("Linhas", len(tables["lines"]))
    col3.metric("Transformadores", len(tables["trafos"]))
    col4.metric("Cargas", len(tables["loads"]))
    
    st.info("Configure o sistema em **Dados do Sistema** para começar")

def page_system_data():
    st.title("📊 Dados do Sistema Elétrico")
    
    tabs = st.tabs([
        "🔌 Barras",
        "⚡ Geradores",
        "🏭 Cargas",
        "📏 Linhas",
        "🔄 Transformadores",
        "🔀 Chaves",
        "🌐 Grid Externo",
        "⚙️ Compensação"
    ])
    
    t = st.session_state.tables
    bus_names = get_bus_names()
    
    with tabs[0]:
        st.markdown("### Barras do Sistema")
        st.info("💡 Defina primeiro as barras antes de configurar outros equipamentos")
        
        t["buses"] = st.data_editor(
            t["buses"],
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "type": st.column_config.SelectboxColumn(
                    "Tipo",
                    options=["Slack", "PV", "PQ"],
                    required=True,
                    help="Slack: barra de referência | PV: geração | PQ: carga"
                ),
                "vn_kv": st.column_config.NumberColumn(
                    "Tensão (kV)", 
                    min_value=0.1, 
                    format="%.2f",
                    help="Tensão nominal da barra"
                ),
                "zone": st.column_config.TextColumn("Zona", help="Zona elétrica"),
                "x_km": st.column_config.NumberColumn("X (km)", format="%.2f"),
                "y_km": st.column_config.NumberColumn("Y (km)", format="%.2f"),
            }
        )
    
    with tabs[1]:
        st.markdown("### Geradores")
        
        # Adicionar novo gerador com seleção de barra
        with st.expander("➕ Adicionar Novo Gerador"):
            col1, col2, col3 = st.columns(3)
            with col1:
                new_gen_name = st.text_input("Nome do Gerador", key="new_gen_name")
                new_gen_bus = st.selectbox("Barra de Conexão", bus_names, key="new_gen_bus")
            with col2:
                new_gen_p = st.number_input("P (MW)", 0.0, 1000.0, 50.0, key="new_gen_p")
                new_gen_vm = st.number_input("V (pu)", 0.9, 1.1, 1.02, 0.01, key="new_gen_vm")
            with col3:
                new_gen_qmin = st.number_input("Q mín (MVAr)", -500.0, 0.0, -30.0, key="new_gen_qmin")
                new_gen_qmax = st.number_input("Q máx (MVAr)", 0.0, 500.0, 40.0, key="new_gen_qmax")
            
            if st.button("Adicionar Gerador", type="primary"):
                if new_gen_name and new_gen_bus:
                    new_row = pd.DataFrame([{
                        "name": new_gen_name,
                        "bus": new_gen_bus,
                        "p_mw": new_gen_p,
                        "vm_pu": new_gen_vm,
                        "min_q_mvar": new_gen_qmin,
                        "max_q_mvar": new_gen_qmax,
                        "type": "Síncrono"
                    }])
                    t["gens"] = pd.concat([t["gens"], new_row], ignore_index=True)
                    st.success(f"✅ Gerador '{new_gen_name}' adicionado")
                    st.rerun()
        
        # Editar geradores existentes
        if not t["gens"].empty:
            t["gens"] = st.data_editor(
                t["gens"],
                num_rows="dynamic",
                use_container_width=True,
                column_config={
                    "bus": st.column_config.SelectboxColumn(
                        "Barra",
                        options=bus_names,
                        required=True,
                        help="Selecione a barra onde o gerador está conectado"
                    ),
                    "p_mw": st.column_config.NumberColumn("P (MW)", format="%.2f"),
                    "vm_pu": st.column_config.NumberColumn("V (pu)", format="%.3f"),
                }
            )
        else:
            st.info("Nenhum gerador cadastrado. Use o formulário acima para adicionar.")
    
    with tabs[2]:
        st.markdown("### Cargas")
        
        # Adicionar nova carga
        with st.expander("➕ Adicionar Nova Carga"):
            col1, col2 = st.columns(2)
            with col1:
                new_load_name = st.text_input("Nome da Carga", key="new_load_name")
                new_load_bus = st.selectbox("Barra de Conexão", bus_names, key="new_load_bus")
                new_load_type = st.selectbox("Tipo", ["Industrial", "Commercial", "Residential"], key="new_load_type")
            with col2:
                new_load_p = st.number_input("P (MW)", 0.0, 1000.0, 25.0, key="new_load_p")
                new_load_q = st.number_input("Q (MVAr)", 0.0, 500.0, 10.0, key="new_load_q")
            
            if st.button("Adicionar Carga", type="primary"):
                if new_load_name and new_load_bus:
                    new_row = pd.DataFrame([{
                        "name": new_load_name,
                        "bus": new_load_bus,
                        "p_mw": new_load_p,
                        "q_mvar": new_load_q,
                        "type": new_load_type
                    }])
                    t["loads"] = pd.concat([t["loads"], new_row], ignore_index=True)
                    st.success(f"✅ Carga '{new_load_name}' adicionada")
                    st.rerun()
        
        if not t["loads"].empty:
            t["loads"] = st.data_editor(
                t["loads"],
                num_rows="dynamic",
                use_container_width=True,
                column_config={
                    "bus": st.column_config.SelectboxColumn(
                        "Barra",
                        options=bus_names,
                        required=True,
                        help="Selecione a barra onde a carga está conectada"
                    ),
                    "type": st.column_config.SelectboxColumn(
                        "Tipo",
                        options=["Industrial", "Commercial", "Residential"]
                    ),
                }
            )
    
    with tabs[3]:
        st.markdown("### Linhas de Transmissão")
        
        # Adicionar nova linha
        with st.expander("➕ Adicionar Nova Linha"):
            col1, col2 = st.columns(2)
            with col1:
                new_line_name = st.text_input("Nome da Linha", key="new_line_name")
                new_line_from = st.selectbox("Barra Origem", bus_names, key="new_line_from")
                new_line_to = st.selectbox("Barra Destino", bus_names, key="new_line_to")
            with col2:
                new_line_length = st.number_input("Comprimento (km)", 0.1, 1000.0, 45.0, key="new_line_length")
                new_line_r = st.number_input("R (Ω/km)", 0.001, 1.0, 0.035, format="%.4f", key="new_line_r")
                new_line_x = st.number_input("X (Ω/km)", 0.001, 1.0, 0.38, format="%.4f", key="new_line_x")
            
            if st.button("Adicionar Linha", type="primary"):
                if new_line_name and new_line_from and new_line_to:
                    if new_line_from == new_line_to:
                        st.error("❌ Barras de origem e destino devem ser diferentes")
                    else:
                        new_row = pd.DataFrame([{
                            "name": new_line_name,
                            "from_bus": new_line_from,
                            "to_bus": new_line_to,
                            "length_km": new_line_length,
                            "r_ohm_per_km": new_line_r,
                            "x_ohm_per_km": new_line_x,
                            "c_nf_per_km": 11.0,
                            "max_i_ka": 1.2
                        }])
                        t["lines"] = pd.concat([t["lines"], new_row], ignore_index=True)
                        st.success(f"✅ Linha '{new_line_name}' adicionada")
                        st.rerun()
        
        if not t["lines"].empty:
            t["lines"] = st.data_editor(
                t["lines"],
                num_rows="dynamic",
                use_container_width=True,
                column_config={
                    "from_bus": st.column_config.SelectboxColumn(
                        "Barra Origem",
                        options=bus_names,
                        required=True
                    ),
                    "to_bus": st.column_config.SelectboxColumn(
                        "Barra Destino",
                        options=bus_names,
                        required=True
                    ),
                }
            )
    
    with tabs[4]:
        st.markdown("### Transformadores")
        
        # Adicionar novo transformador
        with st.expander("➕ Adicionar Novo Transformador"):
            col1, col2 = st.columns(2)
            with col1:
                new_trafo_name = st.text_input("Nome", key="new_trafo_name")
                new_trafo_hv = st.selectbox("Barra AT", bus_names, key="new_trafo_hv")
                new_trafo_lv = st.selectbox("Barra BT", bus_names, key="new_trafo_lv")
            with col2:
                new_trafo_sn = st.number_input("S (MVA)", 1.0, 500.0, 100.0, key="new_trafo_sn")
                new_trafo_vk = st.number_input("Vk (%)", 1.0, 20.0, 12.0, key="new_trafo_vk")
            
            if st.button("Adicionar Transformador", type="primary"):
                if new_trafo_name and new_trafo_hv and new_trafo_lv:
                    if new_trafo_hv == new_trafo_lv:
                        st.error("❌ Barras AT e BT devem ser diferentes")
                    else:
                        new_row = pd.DataFrame([{
                            "name": new_trafo_name,
                            "hv_bus": new_trafo_hv,
                            "lv_bus": new_trafo_lv,
                            "sn_mva": new_trafo_sn,
                            "vn_hv_kv": 230,
                            "vn_lv_kv": 138,
                            "vk_percent": new_trafo_vk
                        }])
                        t["trafos"] = pd.concat([t["trafos"], new_row], ignore_index=True)
                        st.success(f"✅ Transformador '{new_trafo_name}' adicionado")
                        st.rerun()
        
        if not t["trafos"].empty:
            t["trafos"] = st.data_editor(
                t["trafos"],
                num_rows="dynamic",
                use_container_width=True,
                column_config={
                    "hv_bus": st.column_config.SelectboxColumn(
                        "Barra AT",
                        options=bus_names,
                        required=True
                    ),
                    "lv_bus": st.column_config.SelectboxColumn(
                        "Barra BT",
                        options=bus_names,
                        required=True
                    ),
                }
            )
    
    with tabs[5]:
        st.markdown("### Chaves e Disjuntores")
        if not t["switches"].empty:
            t["switches"] = st.data_editor(
                t["switches"],
                num_rows="dynamic",
                use_container_width=True,
                column_config={
                    "bus": st.column_config.SelectboxColumn(
                        "Barra",
                        options=bus_names,
                        required=True
                    ),
                }
            )
    
    with tabs[6]:
        st.markdown("### Conexão com Rede Externa")
        if not t["ext_grids"].empty:
            t["ext_grids"] = st.data_editor(
                t["ext_grids"],
                num_rows="dynamic",
                use_container_width=True,
                column_config={
                    "bus": st.column_config.SelectboxColumn(
                        "Barra",
                        options=bus_names,
                        required=True,
                        help="Barra de conexão com a rede externa (normalmente Slack)"
                    ),
                }
            )
    
    with tabs[7]:
        st.markdown("### Bancos de Capacitores")
        if not t["shunts"].empty:
            t["shunts"] = st.data_editor(
                t["shunts"],
                num_rows="dynamic",
                use_container_width=True,
                column_config={
                    "bus": st.column_config.SelectboxColumn(
                        "Barra",
                        options=bus_names,
                        required=True
                    ),
                }
            )
    
    st.session_state.tables = t
    
    # Ações
    st.divider()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("✅ Validar Sistema", type="primary", use_container_width=True):
            ok, warns, errs = validate_system(t)
            if ok:
                st.success("✅ Sistema validado com sucesso!")
            if warns:
                for w in warns:
                    st.warning(w)
            if errs:
                for e in errs:
                    st.error(e)
    
    with col2:
        if st.button("🔨 Construir Rede", type="primary", use_container_width=True):
            ok, warns, errs = validate_system(t)
            if errs:
                st.error("❌ Corrija os erros antes de construir")
                for e in errs:
                    st.error(e)
            else:
                st.success("✅ Rede validada! (Construção simulada)")
    
    with col3:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.tables = default_tables()
            st.rerun()

def page_simulations():
    st.title("⚙️ Simulações")
    st.info("Funcionalidade em desenvolvimento")

def page_about():
    st.title("ℹ️ Sobre o Sistema")
    st.markdown("""
    ## Power System Professional Studio v2.0
    
    ### Sistema com Seleção Editável de Barras
    
    #### 🎯 Melhorias Implementadas:
    - ✅ **Seleção por Dropdown**: Todas as conexões de equipamentos agora usam seleção dropdown
    - ✅ **Validação Automática**: Verifica se as barras selecionadas existem
    - ✅ **Formulários de Adição**: Interface intuitiva para adicionar novos equipamentos
    - ✅ **Prevenção de Erros**: Valida barras origem/destino diferentes em linhas e transformadores
    
    #### 📝 Como Usar:
    1. Defina primeiro as **Barras** do sistema
    2. Use os formulários "➕ Adicionar" em cada aba para adicionar equipamentos
    3. Selecione as barras de conexão usando os dropdowns
    4. Valide o sistema antes de construir a rede
    """)

# ═════════════════════════════════════════════════════════════════════════
# APLICAÇÃO PRINCIPAL
# ═════════════════════════════════════════════════════════════════════════

def main():
    init_state()
    page = render_sidebar()
    
    if page == "Dashboard":
        page_dashboard()
    elif page == "Dados":
        page_system_data()
    elif page == "Simulações":
        page_simulations()
    else:
        page_about()

if __name__ == "__main__":
    main()
