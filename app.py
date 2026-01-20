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
    .equipment-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .bus-badge {
        background-color: #4CAF50;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 1rem;
        font-size: 0.9rem;
        display: inline-block;
        margin: 0.2rem;
    }
    .connection-line {
        border-left: 3px dashed #2196F3;
        padding-left: 1rem;
        margin: 1rem 0;
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
# CALCULADORA ELÉTRICA
# ═════════════════════════════════════════════════════════════════════════

class ElectricalCalculator:
    """Calculadora de grandezas elétricas"""
    
    @staticmethod
    def nominal_current(s_mva: float, v_kv: float, phases: int = 3) -> float:
        """Corrente nominal (A)"""
        if phases == 3:
            return (s_mva * 1000) / (np.sqrt(3) * v_kv)
        return (s_mva * 1000) / v_kv
    
    @staticmethod
    def voltage_drop(i_a: float, r_ohm_km: float, x_ohm_km: float, 
                     v_kv: float, length_km: float, pf: float = 0.92) -> float:
        """Queda de tensão percentual"""
        z_ohm = np.sqrt(r_ohm_km**2 + x_ohm_km**2) * length_km
        v_drop = np.sqrt(3) * i_a * z_ohm
        v_base = v_kv * 1000
        return (v_drop / v_base) * 100
    
    @staticmethod
    def power_loss(i_a: float, r_ohm_km: float, length_km: float, phases: int = 3) -> float:
        """Perdas de potência (kW)"""
        return phases * (i_a ** 2) * r_ohm_km * length_km / 1000
    
    @staticmethod
    def short_circuit_current(v_kv: float, z_ohm: float) -> float:
        """Corrente de curto-circuito trifásica (kA)"""
        v_phase = v_kv * 1000 / np.sqrt(3)
        return (v_phase / z_ohm) / 1000

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
    st.title("📊 Configuração do Sistema Elétrico")
    
    # Informativo no topo
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.info("💡 **Dica:** Configure primeiro as barras, depois adicione os equipamentos e suas conexões")
    with col2:
        if st.button("🔍 Visualizar Conexões", use_container_width=True):
            show_connection_diagram()
    with col3:
        bus_count = len(st.session_state.tables["buses"])
        st.metric("Barras no Sistema", bus_count)
    
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
        render_buses_tab(t)
    
    with tabs[1]:
        render_generators_tab(t, bus_names)
    
    with tabs[2]:
        render_loads_tab(t, bus_names)
    
    with tabs[3]:
        render_lines_tab(t, bus_names)
    
    with tabs[4]:
        render_transformers_tab(t, bus_names)
    
    with tabs[5]:
        render_switches_tab(t, bus_names)
    
    with tabs[6]:
        render_ext_grids_tab(t, bus_names)
    
    with tabs[7]:
        render_shunts_tab(t, bus_names)
    
    st.session_state.tables = t
    
    # Ações principais
    render_system_actions(t)

def show_connection_diagram():
    """Mostra diagrama de conexões do sistema"""
    st.markdown("### 🔗 Diagrama de Conexões")
    
    t = st.session_state.tables
    
    for _, bus in t["buses"].iterrows():
        bus_name = bus["name"]
        st.markdown(f"#### 🔌 {bus_name} ({bus['vn_kv']} kV - {bus['type']})")
        
        # Grid externo
        ext = t["ext_grids"][t["ext_grids"]["bus"] == bus_name]
        if not ext.empty:
            for _, e in ext.iterrows():
                st.markdown(f"<div class='connection-line'>🌐 <b>{e['name']}</b> - Grid Externo ({e['s_sc_max_mva']} MVA)</div>", 
                           unsafe_allow_html=True)
        
        # Geradores
        gens = t["gens"][t["gens"]["bus"] == bus_name]
        if not gens.empty:
            for _, g in gens.iterrows():
                st.markdown(f"<div class='connection-line'>⚡ <b>{g['name']}</b> - {g['p_mw']} MW</div>", 
                           unsafe_allow_html=True)
        
        # Cargas
        loads = t["loads"][t["loads"]["bus"] == bus_name]
        if not loads.empty:
            for _, l in loads.iterrows():
                st.markdown(f"<div class='connection-line'>🏭 <b>{l['name']}</b> - {l['p_mw']} MW / {l['q_mvar']} MVAr</div>", 
                           unsafe_allow_html=True)
        
        # Linhas
        lines_from = t["lines"][t["lines"]["from_bus"] == bus_name]
        lines_to = t["lines"][t["lines"]["to_bus"] == bus_name]
        
        for _, line in lines_from.iterrows():
            st.markdown(f"<div class='connection-line'>📏 <b>{line['name']}</b> → {line['to_bus']} ({line['length_km']} km)</div>", 
                       unsafe_allow_html=True)
        
        for _, line in lines_to.iterrows():
            st.markdown(f"<div class='connection-line'>📏 <b>{line['name']}</b> ← {line['from_bus']} ({line['length_km']} km)</div>", 
                       unsafe_allow_html=True)
        
        st.divider()

def render_buses_tab(t):
    st.markdown("### 🔌 Barras do Sistema")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info("💡 As barras são os nós do sistema onde equipamentos se conectam")
    with col2:
        if st.button("➕ Adicionar Barra Rápida"):
            new_name = f"BARRA-{len(t['buses'])+1}"
            new_row = pd.DataFrame([{
                "name": new_name, "vn_kv": 138.0, "type": "PQ", 
                "zone": "Zona-1", "x_km": 0.0, "y_km": 0.0
            }])
            t["buses"] = pd.concat([t["buses"], new_row], ignore_index=True)
            st.success(f"✅ Barra '{new_name}' adicionada")
            st.rerun()
    
    t["buses"] = st.data_editor(
        t["buses"],
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "name": st.column_config.TextColumn("Nome da Barra", width="medium", required=True),
            "type": st.column_config.SelectboxColumn(
                "Tipo",
                options=["Slack", "PV", "PQ"],
                required=True,
                help="Slack: referência | PV: geração | PQ: carga",
                width="small"
            ),
            "vn_kv": st.column_config.NumberColumn(
                "Tensão (kV)", 
                min_value=0.1,
                max_value=765.0,
                format="%.2f",
                help="Tensão nominal",
                width="small"
            ),
            "zone": st.column_config.TextColumn("Zona", width="small"),
            "x_km": st.column_config.NumberColumn("X (km)", format="%.2f", width="small"),
            "y_km": st.column_config.NumberColumn("Y (km)", format="%.2f", width="small"),
        }
    )

def render_generators_tab(t, bus_names):
    st.markdown("### ⚡ Geradores do Sistema")
    
    with st.expander("➕ Adicionar Novo Gerador", expanded=False):
        with st.form("add_generator"):
            col1, col2, col3 = st.columns(3)
            with col1:
                new_gen_name = st.text_input("Nome", value=f"GER-{len(t['gens'])+1}")
                new_gen_bus = st.selectbox("Barra de Conexão", bus_names)
                new_gen_type = st.selectbox("Tipo", ["Síncrono", "Assíncrono", "Renovável"])
            with col2:
                new_gen_p = st.number_input("Potência Ativa (MW)", 0.0, 1000.0, 50.0, step=5.0)
                new_gen_vm = st.number_input("Tensão Terminal (pu)", 0.9, 1.1, 1.02, 0.01)
            with col3:
                new_gen_qmin = st.number_input("Q mínimo (MVAr)", -500.0, 0.0, -30.0, step=5.0)
                new_gen_qmax = st.number_input("Q máximo (MVAr)", 0.0, 500.0, 40.0, step=5.0)
            
            submitted = st.form_submit_button("✅ Adicionar Gerador", type="primary", use_container_width=True)
            if submitted and new_gen_name and new_gen_bus:
                new_row = pd.DataFrame([{
                    "name": new_gen_name, "bus": new_gen_bus, "p_mw": new_gen_p,
                    "vm_pu": new_gen_vm, "min_q_mvar": new_gen_qmin,
                    "max_q_mvar": new_gen_qmax, "type": new_gen_type
                }])
                t["gens"] = pd.concat([t["gens"], new_row], ignore_index=True)
                st.success(f"✅ Gerador '{new_gen_name}' adicionado na barra '{new_gen_bus}'")
                st.rerun()
    
    if not t["gens"].empty:
        st.markdown("#### Geradores Configurados")
        t["gens"] = st.data_editor(
            t["gens"],
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "name": st.column_config.TextColumn("Nome", width="medium"),
                "bus": st.column_config.SelectboxColumn(
                    "🔌 Barra",
                    options=bus_names,
                    required=True,
                    help="Barra onde o gerador está conectado",
                    width="medium"
                ),
                "p_mw": st.column_config.NumberColumn("P (MW)", format="%.2f", width="small"),
                "vm_pu": st.column_config.NumberColumn("V (pu)", format="%.3f", width="small"),
                "type": st.column_config.SelectboxColumn(
                    "Tipo",
                    options=["Síncrono", "Assíncrono", "Renovável"],
                    width="small"
                ),
            }
        )
    else:
        st.info("Nenhum gerador cadastrado. Use o formulário acima.")

def render_loads_tab(t, bus_names):
    st.markdown("### 🏭 Cargas do Sistema")
    
    with st.expander("➕ Adicionar Nova Carga", expanded=False):
        with st.form("add_load"):
            col1, col2 = st.columns(2)
            with col1:
                new_load_name = st.text_input("Nome", value=f"CARGA-{len(t['loads'])+1}")
                new_load_bus = st.selectbox("Barra de Conexão", bus_names)
                new_load_type = st.selectbox("Tipo", ["Industrial", "Commercial", "Residential"])
            with col2:
                new_load_p = st.number_input("Potência Ativa (MW)", 0.0, 1000.0, 25.0, step=5.0)
                new_load_q = st.number_input("Potência Reativa (MVAr)", 0.0, 500.0, 10.0, step=5.0)
                fp_calc = new_load_p / np.sqrt(new_load_p**2 + new_load_q**2) if new_load_p > 0 else 0
                st.metric("Fator de Potência", f"{fp_calc:.3f}")
            
            submitted = st.form_submit_button("✅ Adicionar Carga", type="primary", use_container_width=True)
            if submitted and new_load_name and new_load_bus:
                new_row = pd.DataFrame([{
                    "name": new_load_name, "bus": new_load_bus,
                    "p_mw": new_load_p, "q_mvar": new_load_q, "type": new_load_type
                }])
                t["loads"] = pd.concat([t["loads"], new_row], ignore_index=True)
                st.success(f"✅ Carga '{new_load_name}' adicionada na barra '{new_load_bus}'")
                st.rerun()
    
    if not t["loads"].empty:
        st.markdown("#### Cargas Configuradas")
        t["loads"] = st.data_editor(
            t["loads"],
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "name": st.column_config.TextColumn("Nome", width="medium"),
                "bus": st.column_config.SelectboxColumn(
                    "🔌 Barra",
                    options=bus_names,
                    required=True,
                    width="medium"
                ),
                "p_mw": st.column_config.NumberColumn("P (MW)", format="%.2f", width="small"),
                "q_mvar": st.column_config.NumberColumn("Q (MVAr)", format="%.2f", width="small"),
                "type": st.column_config.SelectboxColumn(
                    "Tipo",
                    options=["Industrial", "Commercial", "Residential"],
                    width="small"
                ),
            }
        )

def render_lines_tab(t, bus_names):
    st.markdown("### 📏 Linhas de Transmissão")
    
    calc = ElectricalCalculator()
    
    with st.expander("➕ Adicionar Nova Linha", expanded=False):
        with st.form("add_line"):
            col1, col2, col3 = st.columns(3)
            with col1:
                new_line_name = st.text_input("Nome", value=f"LT-{len(t['lines'])+1}")
                new_line_from = st.selectbox("🔌 Barra Origem", bus_names, key="line_from")
                new_line_to = st.selectbox("🔌 Barra Destino", bus_names, key="line_to")
            with col2:
                new_line_length = st.number_input("Comprimento (km)", 0.1, 1000.0, 45.0, step=5.0)
                new_line_r = st.number_input("R (Ω/km)", 0.001, 1.0, 0.035, format="%.4f")
                new_line_x = st.number_input("X (Ω/km)", 0.001, 1.0, 0.38, format="%.4f")
            with col3:
                new_line_c = st.number_input("C (nF/km)", 0.0, 50.0, 11.0, step=1.0)
                new_line_imax = st.number_input("Corrente Máx (kA)", 0.1, 10.0, 1.2, step=0.1)
                z_total = np.sqrt(new_line_r**2 + new_line_x**2) * new_line_length
                st.metric("Impedância Total", f"{z_total:.2f} Ω")
            
            submitted = st.form_submit_button("✅ Adicionar Linha", type="primary", use_container_width=True)
            if submitted and new_line_name and new_line_from and new_line_to:
                if new_line_from == new_line_to:
                    st.error("❌ Barras de origem e destino devem ser diferentes")
                else:
                    new_row = pd.DataFrame([{
                        "name": new_line_name, "from_bus": new_line_from, "to_bus": new_line_to,
                        "length_km": new_line_length, "r_ohm_per_km": new_line_r,
                        "x_ohm_per_km": new_line_x, "c_nf_per_km": new_line_c,
                        "max_i_ka": new_line_imax
                    }])
                    t["lines"] = pd.concat([t["lines"], new_row], ignore_index=True)
                    st.success(f"✅ Linha '{new_line_name}' adicionada: {new_line_from} → {new_line_to}")
                    st.rerun()
    
    if not t["lines"].empty:
        st.markdown("#### Linhas Configuradas")
        t["lines"] = st.data_editor(
            t["lines"],
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "name": st.column_config.TextColumn("Nome", width="medium"),
                "from_bus": st.column_config.SelectboxColumn("🔌 De", options=bus_names, width="medium"),
                "to_bus": st.column_config.SelectboxColumn("🔌 Para", options=bus_names, width="medium"),
                "length_km": st.column_config.NumberColumn("L (km)", format="%.2f", width="small"),
            }
        )

def render_transformers_tab(t, bus_names):
    st.markdown("### 🔄 Transformadores")
    
    with st.expander("➕ Adicionar Novo Transformador", expanded=False):
        with st.form("add_trafo"):
            col1, col2, col3 = st.columns(3)
            with col1:
                new_trafo_name = st.text_input("Nome", value=f"TR-{len(t['trafos'])+1}")
                new_trafo_hv = st.selectbox("🔌 Barra AT (Alta Tensão)", bus_names, key="trafo_hv")
                new_trafo_lv = st.selectbox("🔌 Barra BT (Baixa Tensão)", bus_names, key="trafo_lv")
            with col2:
                new_trafo_sn = st.number_input("Potência Nominal (MVA)", 1.0, 500.0, 100.0, step=10.0)
                new_trafo_vhv = st.number_input("Tensão AT (kV)", 1.0, 500.0, 230.0, step=10.0)
                new_trafo_vlv = st.number_input("Tensão BT (kV)", 1.0, 500.0, 138.0, step=10.0)
            with col3:
                new_trafo_vk = st.number_input("Impedância (Vk %)", 1.0, 20.0, 12.0, step=0.5)
                new_trafo_vkr = st.number_input("Resistência (Vkr %)", 0.1, 5.0, 0.5, step=0.1)
                ratio = new_trafo_vhv / new_trafo_vlv if new_trafo_vlv > 0 else 0
                st.metric("Relação de Transformação", f"{ratio:.2f}")
            
            submitted = st.form_submit_button("✅ Adicionar Transformador", type="primary", use_container_width=True)
            if submitted and new_trafo_name and new_trafo_hv and new_trafo_lv:
                if new_trafo_hv == new_trafo_lv:
                    st.error("❌ Barras AT e BT devem ser diferentes")
                else:
                    new_row = pd.DataFrame([{
                        "name": new_trafo_name, "hv_bus": new_trafo_hv, "lv_bus": new_trafo_lv,
                        "sn_mva": new_trafo_sn, "vn_hv_kv": new_trafo_vhv,
                        "vn_lv_kv": new_trafo_vlv, "vk_percent": new_trafo_vk
                    }])
                    t["trafos"] = pd.concat([t["trafos"], new_row], ignore_index=True)
                    st.success(f"✅ Transformador '{new_trafo_name}' adicionado: {new_trafo_hv} ⇄ {new_trafo_lv}")
                    st.rerun()
    
    if not t["trafos"].empty:
        st.markdown("#### Transformadores Configurados")
        t["trafos"] = st.data_editor(
            t["trafos"],
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "name": st.column_config.TextColumn("Nome", width="medium"),
                "hv_bus": st.column_config.SelectboxColumn("🔌 AT", options=bus_names, width="medium"),
                "lv_bus": st.column_config.SelectboxColumn("🔌 BT", options=bus_names, width="medium"),
                "sn_mva": st.column_config.NumberColumn("S (MVA)", format="%.1f", width="small"),
            }
        )

def render_switches_tab(t, bus_names):
    st.markdown("### 🔀 Chaves e Disjuntores")
    if not t["switches"].empty:
        t["switches"] = st.data_editor(
            t["switches"], num_rows="dynamic", use_container_width=True,
            column_config={
                "bus": st.column_config.SelectboxColumn("🔌 Barra", options=bus_names),
                "closed": st.column_config.CheckboxColumn("Fechada"),
            }
        )
    else:
        st.info("Nenhuma chave cadastrada")

def render_ext_grids_tab(t, bus_names):
    st.markdown("### 🌐 Conexão com Rede Externa")
    if not t["ext_grids"].empty:
        t["ext_grids"] = st.data_editor(
            t["ext_grids"], num_rows="dynamic", use_container_width=True,
            column_config={
                "bus": st.column_config.SelectboxColumn(
                    "🔌 Barra", options=bus_names, 
                    help="Barra de conexão (geralmente Slack)"
                ),
            }
        )

def render_shunts_tab(t, bus_names):
    st.markdown("### ⚙️ Bancos de Capacitores")
    if not t["shunts"].empty:
        t["shunts"] = st.data_editor(
            t["shunts"], num_rows="dynamic", use_container_width=True,
            column_config={
                "bus": st.column_config.SelectboxColumn("🔌 Barra", options=bus_names),
                "in_service": st.column_config.CheckboxColumn("Em Serviço"),
            }
        )
    else:
        st.info("Nenhum banco de capacitor cadastrado")

def render_system_actions(t):
    st.divider()
    st.markdown("### 🎯 Ações do Sistema")
    
    col1, col2, col3, col4 = st.columns(4)
    
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
            else:
                st.success("✅ Rede construída! Prossiga para Simulações")
                st.session_state.net_built = True
    
    with col3:
        if st.button("📊 Ver Estatísticas", use_container_width=True):
            show_system_statistics(t)
    
    with col4:
        if st.button("🔄 Reset Sistema", use_container_width=True):
            if st.session_state.get("confirm_reset", False):
                st.session_state.tables = default_tables()
                st.session_state.net_built = None
                st.session_state.confirm_reset = False
                st.rerun()
            else:
                st.session_state.confirm_reset = True
                st.warning("⚠️ Clique novamente para confirmar")

def show_system_statistics(t):
    st.markdown("### 📊 Estatísticas do Sistema")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total de Barras", len(t["buses"]))
    col2.metric("Total de Geradores", len(t["gens"]))
    col3.metric("Total de Cargas", len(t["loads"]))
    col4.metric("Total de Linhas", len(t["lines"]))
    
    # Potências
    total_gen = t["gens"]["p_mw"].sum() if not t["gens"].empty else 0
    total_load = t["loads"]["p_mw"].sum() if not t["loads"].empty else 0
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Geração Total", f"{total_gen:.2f} MW")
    col2.metric("Carga Total", f"{total_load:.2f} MW")
    col3.metric("Balanço", f"{total_gen - total_load:.2f} MW")

def page_about():
    st.title("ℹ️ Sobre o Sistema")
    st.markdown("""
    ## Power System Professional Studio v2.0
    
    ### Sistema Completo de Análise de Redes Elétricas
    
    #### 🎯 Recursos Implementados:
    - ✅ **Seleção por Dropdown**: Todas as conexões usam seleção inteligente
    - ✅ **Simulações Completas**: Fluxo de potência, curto-circuito, contingências
    - ✅ **Dimensionamento**: Seleção automática de condutores
    - ✅ **Calculadora**: Ferramentas de cálculo elétrico
    - ✅ **Visualizações**: Gráficos interativos e dashboards
    - ✅ **Validação**: Verificação automática de erros
    
    #### 📝 Como Usar:
    1. **Dados do Sistema**: Configure barras e equipamentos
    2. **Construir Rede**: Valide e construa o modelo
    3. **Simulações**: Execute análises avançadas
    4. **Resultados**: Visualize e exporte relatórios
    
    #### 🛠️ Tecnologias:
    - Python 3.8+
    - Streamlit
    - Plotly
    - Pandas / NumPy
    
    ---
    
    💡 **Dica:** Use os formulários com ➕ para adicionar equipamentos rapidamente.
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
    main() = st.selectbox("Barra de Conexão", bus_names, key="new_load_bus")
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
    st.title("⚙️ Simulações e Análises Avançadas")
    
    if not st.session_state.get("net_built"):
        st.error("⚠️ **Construa a rede primeiro** em 'Dados do Sistema'")
        st.info("👈 Vá para **Dados do Sistema** → Configure as barras e equipamentos → Clique em **🔨 Construir Rede**")
        return
    
    st.success("✅ Rede construída e pronta para simulações")
    
    # Menu de simulações
    sim_type = st.selectbox(
        "Selecione o tipo de análise",
        [
            "⚡ Fluxo de Potência",
            "💥 Curto-Circuito",
            "🔀 Contingências N-1",
            "📐 Dimensionamento de Condutores",
            "🧮 Calculadora Elétrica"
        ],
        index=0
    )
    
    st.divider()
    
    if sim_type == "⚡ Fluxo de Potência":
        run_power_flow_simulation()
    elif sim_type == "💥 Curto-Circuito":
        run_short_circuit_simulation()
    elif sim_type == "🔀 Contingências N-1":
        run_contingency_simulation()
    elif sim_type == "📐 Dimensionamento de Condutores":
        run_conductor_sizing()
    else:
        run_electrical_calculator()

def run_power_flow_simulation():
    st.markdown("### ⚡ Análise de Fluxo de Potência")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        algorithm = st.selectbox(
            "Algoritmo",
            ["Newton-Raphson", "Gauss-Seidel", "DC (Linearizado)", "Fast-Decoupled"],
            help="NR: mais preciso | DC: mais rápido | GS: mais estável"
        )
    with col2:
        tolerance = st.number_input("Tolerância (MVA)", 1e-12, 1e-2, 1e-8, format="%.1e")
    with col3:
        max_iter = st.number_input("Iterações Máximas", 10, 100, 20)
    with col4:
        enforce_q = st.checkbox("Respeitar limites Q", value=True)
    
    if st.button("▶️ Executar Fluxo de Potência", type="primary", use_container_width=True):
        with st.spinner("🔄 Calculando fluxo de potência..."):
            # Simulação
            import time
            time.sleep(1.5)
            
            # Resultados simulados
            t = st.session_state.tables
            n_buses = len(t["buses"])
            
            # Dados simulados
            bus_results = pd.DataFrame({
                "Barra": t["buses"]["name"].tolist(),
                "V (pu)": np.random.uniform(0.98, 1.02, n_buses),
                "Ângulo (°)": np.random.uniform(-5, 5, n_buses),
                "P (MW)": np.random.uniform(-50, 100, n_buses),
                "Q (MVAr)": np.random.uniform(-20, 40, n_buses),
            })
            
            st.success("✅ Convergiu em 8 iterações")
            
            # Visualização
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 Perfil de Tensão")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=bus_results["Barra"],
                    y=bus_results["V (pu)"],
                    mode='lines+markers',
                    name='Tensão',
                    line=dict(color='#1e3c72', width=3),
                    marker=dict(size=10)
                ))
                fig.add_hline(y=1.05, line_dash="dash", line_color="red", annotation_text="Máx")
                fig.add_hline(y=0.95, line_dash="dash", line_color="red", annotation_text="Mín")
                fig.update_layout(
                    yaxis_title="Tensão (pu)",
                    template="plotly_white",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### 📈 Distribuição de Potência")
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=bus_results["Barra"],
                    y=bus_results["P (MW)"],
                    name="P (MW)",
                    marker_color='#2196F3'
                ))
                fig.add_trace(go.Bar(
                    x=bus_results["Barra"],
                    y=bus_results["Q (MVAr)"],
                    name="Q (MVAr)",
                    marker_color='#FF9800'
                ))
                fig.update_layout(
                    yaxis_title="Potência",
                    template="plotly_white",
                    height=400,
                    barmode='group'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Tabela de resultados
            st.markdown("#### 📋 Resultados Detalhados")
            st.dataframe(bus_results, use_container_width=True)
            
            # Métricas
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Perdas Totais", "3.47 MW", delta="-0.8 MW")
            col2.metric("V mínima", f"{bus_results['V (pu)'].min():.3f} pu", 
                       delta=f"{bus_results['V (pu)'].min() - 0.95:.3f}")
            col3.metric("V máxima", f"{bus_results['V (pu)'].max():.3f} pu",
                       delta=f"{1.05 - bus_results['V (pu)'].max():.3f}")
            col4.metric("Geração Total", f"{bus_results['P (MW)'][bus_results['P (MW)'] > 0].sum():.2f} MW")

def run_short_circuit_simulation():
    st.markdown("### 💥 Análise de Curto-Circuito")
    
    t = st.session_state.tables
    
    col1, col2, col3 = st.columns(3)
    with col1:
        bus_fault = st.selectbox("Barra em falta", t["buses"]["name"].tolist())
        fault_type = st.selectbox("Tipo de falta", 
                                  ["Trifásica", "Bifásica", "Monofásica", "Bifásica-Terra"])
    with col2:
        z_fault = st.number_input("Impedância de falta (Ω)", 0.0, 10.0, 0.0, step=0.1)
        r_x_ratio = st.number_input("Relação R/X da fonte", 0.0, 1.0, 0.1, step=0.05)
    with col3:
        calc_ith = st.checkbox("Calcular corrente térmica (ith)", value=True)
        calc_ip = st.checkbox("Calcular corrente de pico (ip)", value=True)
    
    if st.button("⚡ Calcular Curto-Circuito", type="primary", use_container_width=True):
        with st.spinner("⚡ Calculando correntes de curto..."):
            import time
            time.sleep(1.2)
            
            # Resultados simulados
            results = pd.DataFrame({
                "Barra": t["buses"]["name"].tolist(),
                "Ikss (kA)": np.random.uniform(15, 45, len(t["buses"])),
                "ip (kA)": np.random.uniform(35, 95, len(t["buses"])),
                "Sk'' (MVA)": np.random.uniform(1500, 5000, len(t["buses"])),
            })
            
            st.success(f"✅ Cálculo concluído para falta em **{bus_fault}**")
            
            # Destaque da barra em falta
            fault_idx = results[results["Barra"] == bus_fault].index[0]
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Corrente Subtransitória", f"{results.loc[fault_idx, 'Ikss (kA)']:.2f} kA", 
                       help="Corrente inicial de curto-circuito")
            col2.metric("Corrente de Pico", f"{results.loc[fault_idx, 'ip (kA)']:.2f} kA",
                       help="Valor máximo instantâneo")
            col3.metric("Potência de Curto", f"{results.loc[fault_idx, 'Sk'' (MVA)']:.0f} MVA")
            
            # Gráfico
            st.markdown("#### 📊 Correntes de Curto-Circuito")
            fig = go.Figure()
            
            colors = ['red' if b == bus_fault else '#1e3c72' for b in results["Barra"]]
            
            fig.add_trace(go.Bar(
                x=results["Barra"],
                y=results["Ikss (kA)"],
                marker_color=colors,
                text=[f"{x:.1f} kA" for x in results["Ikss (kA)"]],
                textposition='outside'
            ))
            
            fig.update_layout(
                title=f"Correntes de Curto-Circuito - Falta em {bus_fault}",
                yaxis_title="Ikss (kA)",
                template="plotly_white",
                height=450
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Tabela completa
            st.markdown("#### 📋 Resultados Completos")
            st.dataframe(results, use_container_width=True)

def run_contingency_simulation():
    st.markdown("### 🔀 Análise de Contingências N-1")
    
    st.info("💡 Simula a saída de cada elemento (linhas/transformadores) e verifica violações")
    
    t = st.session_state.tables
    
    col1, col2 = st.columns(2)
    with col1:
        include_lines = st.checkbox("Incluir Linhas", value=True)
        include_trafos = st.checkbox("Incluir Transformadores", value=True)
    with col2:
        v_min = st.number_input("V mínima aceitável (pu)", 0.90, 1.00, 0.95, 0.01)
        load_max = st.number_input("Carregamento máx (%)", 80, 150, 100)
    
    if st.button("🔍 Analisar Contingências", type="primary", use_container_width=True):
        with st.spinner("🔄 Analisando contingências..."):
            import time
            time.sleep(2)
            
            # Contingências simuladas
            contingencies = []
            
            if include_lines:
                for _, line in t["lines"].iterrows():
                    contingencies.append({
                        "Elemento": line["name"],
                        "Tipo": "Linha",
                        "Convergiu": np.random.choice([True, True, True, False]),
                        "Violações V": np.random.randint(0, 3),
                        "Sobrecargas": np.random.randint(0, 2),
                        "Severidade": np.random.choice(["OK", "OK", "Alerta", "Crítico"])
                    })
            
            if include_trafos:
                for _, trafo in t["trafos"].iterrows():
                    contingencies.append({
                        "Elemento": trafo["name"],
                        "Tipo": "Transformador",
                        "Convergiu": np.random.choice([True, True, True, False]),
                        "Violações V": np.random.randint(0, 2),
                        "Sobrecargas": np.random.randint(0, 3),
                        "Severidade": np.random.choice(["OK", "OK", "Alerta", "Crítico"])
                    })
            
            results_df = pd.DataFrame(contingencies)
            
            st.success(f"✅ Analisadas {len(results_df)} contingências")
            
            # Resumo
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Analisado", len(results_df))
            col2.metric("OK", len(results_df[results_df["Severidade"] == "OK"]))
            col3.metric("Alertas", len(results_df[results_df["Severidade"] == "Alerta"]))
            col4.metric("Críticas", len(results_df[results_df["Severidade"] == "Crítico"]))
            
            # Filtros
            severity_filter = st.multiselect(
                "Filtrar por severidade",
                ["OK", "Alerta", "Crítico"],
                default=["Alerta", "Crítico"]
            )
            
            filtered = results_df[results_df["Severidade"].isin(severity_filter)]
            
            # Coloração
            def color_severity(val):
                if val == "OK":
                    return "background-color: #d4edda"
                elif val == "Alerta":
                    return "background-color: #fff3cd"
                else:
                    return "background-color: #f8d7da"
            
            styled = filtered.style.applymap(color_severity, subset=["Severidade"])
            st.dataframe(styled, use_container_width=True)
            
            # Gráfico de pizza
            if not results_df.empty:
                fig = go.Figure(data=[go.Pie(
                    labels=results_df["Severidade"].value_counts().index,
                    values=results_df["Severidade"].value_counts().values,
                    marker=dict(colors=['#28a745', '#ffc107', '#dc3545'])
                )])
                fig.update_layout(
                    title="Distribuição de Severidade",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

def run_conductor_sizing():
    st.markdown("### 📐 Dimensionamento de Condutores")
    
    calc = ElectricalCalculator()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Parâmetros do Sistema")
        power_mva = st.number_input("Potência a transmitir (MVA)", 1.0, 500.0, 50.0, step=5.0)
        voltage_kv = st.number_input("Tensão nominal (kV)", 1.0, 500.0, 138.0, step=10.0)
        distance_km = st.number_input("Distância (km)", 0.1, 500.0, 45.0, step=5.0)
        power_factor = st.slider("Fator de potência", 0.70, 1.0, 0.92, 0.01)
        max_drop = st.number_input("Queda de tensão máxima (%)", 1.0, 10.0, 3.0, 0.5)
    
    with col2:
        st.markdown("#### Condições de Instalação")
        temp = st.selectbox("Temperatura de operação", ["60°C", "75°C", "90°C"], index=1)
        installation = st.selectbox("Tipo de instalação", 
                                    ["Aérea - Circuito Simples", 
                                     "Aérea - Circuito Duplo",
                                     "Subterrânea"])
        parallel = st.number_input("Circuitos em paralelo", 1, 4, 1)
    
    if st.button("🔍 Dimensionar Condutor", type="primary", use_container_width=True):
        # Cálculos
        current_a = calc.nominal_current(power_mva, voltage_kv)
        current_per_circuit = current_a / parallel
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Corrente Total", f"{current_a:.1f} A")
        col2.metric("Corrente por Circuito", f"{current_per_circuit:.1f} A")
        col3.metric("Circuitos", parallel)
        
        # Seleção de condutores
        st.markdown("#### 📊 Condutores Adequados")
        
        conductors = st.session_state.conductor_db
        temp_col = f"ampacity_{temp.split('°')[0]}C" if temp != "60°C" else "ampacity_75C"
        
        suitable = conductors[conductors[temp_col] >= current_per_circuit * 1.25].copy()  # Fator de segurança 25%
        
        if not suitable.empty:
            # Calcular quedas de tensão para cada condutor
            suitable["Queda V (%)"] = suitable.apply(
                lambda row: calc.voltage_drop(
                    current_per_circuit, 
                    row["r_ohm_km_ac"], 
                    0.4,  # Reatância típica
                    voltage_kv, 
                    distance_km, 
                    power_factor
                ), axis=1
            )
            
            suitable["Perdas (kW)"] = suitable.apply(
                lambda row: calc.power_loss(current_per_circuit, row["r_ohm_km_ac"], distance_km),
                axis=1
            )
            
            # Filtrar por queda de tensão
            suitable["Adequado"] = suitable["Queda V (%)"] <= max_drop
            
            # Coloração
            def highlight_adequate(row):
                if row["Adequado"]:
                    return ['background-color: #d4edda'] * len(row)
                return ['background-color: #f8d7da'] * len(row)
            
            display_cols = ["name", temp_col, "r_ohm_km_ac", "Queda V (%)", "Perdas (kW)", "Adequado"]
            styled = suitable[display_cols].style.apply(highlight_adequate, axis=1)
            
            st.dataframe(styled, use_container_width=True)
            
            # Recomendação
            best = suitable[suitable["Adequado"]].sort_values("area_mm2").head(1)
            if not best.empty:
                st.success(f"✅ **Recomendação:** {best.iloc[0]['name']} - "
                          f"Queda: {best.iloc[0]['Queda V (%)']:.2f}%, "
                          f"Perdas: {best.iloc[0]['Perdas (kW)']:.2f} kW")
        else:
            st.error("❌ Nenhum condutor adequado encontrado para esta aplicação")

def run_electrical_calculator():
    st.markdown("### 🧮 Calculadora Elétrica")
    
    calc = ElectricalCalculator()
    
    tool = st.selectbox(
        "Selecione a ferramenta",
        [
            "Corrente Nominal",
            "Queda de Tensão",
            "Perdas de Potência",
            "Corrente de Curto-Circuito",
        ]
    )
    
    st.divider()
    
    if tool == "Corrente Nominal":
        col1, col2, col3 = st.columns(3)
        with col1:
            s_mva = st.number_input("Potência (MVA)", 0.1, 1000.0, 10.0)
        with col2:
            v_kv = st.number_input("Tensão (kV)", 0.1, 765.0, 138.0)
        with col3:
            phases = st.selectbox("Sistema", ["Trifásico", "Monofásico"])
        
        i = calc.nominal_current(s_mva, v_kv, 3 if phases == "Trifásico" else 1)
        st.success(f"### ⚡ Corrente Nominal: **{i:.2f} A**")
    
    elif tool == "Queda de Tensão":
        col1, col2 = st.columns(2)
        with col1:
            i_a = st.number_input("Corrente (A)", 1.0, 10000.0, 500.0)
            r_ohm = st.number_input("Resistência (Ω/km)", 0.001, 1.0, 0.035, format="%.4f")
            x_ohm = st.number_input("Reatância (Ω/km)", 0.001, 1.0, 0.38, format="%.4f")
        with col2:
            v_kv = st.number_input("Tensão (kV)", 1.0, 765.0, 138.0)
            length = st.number_input("Comprimento (km)", 0.1, 500.0, 45.0)
            pf = st.slider("Fator de Potência", 0.70, 1.0, 0.92, 0.01)
        
        drop = calc.voltage_drop(i_a, r_ohm, x_ohm, v_kv, length, pf)
        
        if drop < 3:
            st.success(f"### ✅ Queda de Tensão: **{drop:.2f}%** (Adequado)")
        elif drop < 5:
            st.warning(f"### ⚠️ Queda de Tensão: **{drop:.2f}%** (Atenção)")
        else:
            st.error(f"### ❌ Queda de Tensão: **{drop:.2f}%** (Excessivo)")
    
    elif tool == "Perdas de Potência":
        col1, col2 = st.columns(2)
        with col1:
            i_a = st.number_input("Corrente (A)", 1.0, 10000.0, 500.0)
            r_ohm = st.number_input("Resistência (Ω/km)", 0.001, 1.0, 0.035, format="%.4f")
        with col2:
            length = st.number_input("Comprimento (km)", 0.1, 500.0, 45.0)
            phases = st.selectbox("Fases", [1, 3], index=1)
        
        loss = calc.power_loss(i_a, r_ohm, length, phases)
        st.success(f"### 📉 Perdas de Potência: **{loss:.2f} kW**")
        
        # Custo anual estimado
        cost_kwh = st.number_input("Custo energia (R$/kWh)", 0.0, 2.0, 0.65, 0.01)
        annual_loss = loss * 8760  # horas/ano
        annual_cost = annual_loss * cost_kwh / 1000
        st.info(f"💰 Custo anual estimado: **R$ {annual_cost:,.2f}**")
    
    else:  # Curto-circuito
        col1, col2 = st.columns(2)
        with col1:
            v_kv = st.number_input("Tensão (kV)", 1.0, 765.0, 138.0)
        with col2:
            z_ohm = st.number_input("Impedância equivalente (Ω)", 0.01, 100.0, 2.5, 0.01)
        
        icc = calc.short_circuit_current(v_kv, z_ohm)
        st.error(f"### ⚡ Corrente de Curto-Circuito: **{icc:.2f} kA**")
        
        st.warning(f"💡 Esta corrente deve ser menor que a capacidade de interrupção dos disjuntores")

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
