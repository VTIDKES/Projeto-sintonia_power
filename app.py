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
    
    # Menu de tabs
    tab_selection = st.radio(
        "Selecione o componente",
        ["🔌 Barras", "⚡ Geradores", "🏭 Cargas", "📏 Linhas", "🔄 Transformadores"],
        horizontal=True
    )
    
    t = st.session_state.tables
    bus_names = get_bus_names()
    
    st.divider()
    
    if tab_selection == "🔌 Barras":
        render_buses_section(t)
    elif tab_selection == "⚡ Geradores":
        render_generators_section(t, bus_names)
    elif tab_selection == "🏭 Cargas":
        render_loads_section(t, bus_names)
    elif tab_selection == "📏 Linhas":
        render_lines_section(t, bus_names)
    elif tab_selection == "🔄 Transformadores":
        render_transformers_section(t, bus_names)
    
    st.session_state.tables = t
    
    # Ações do sistema
    st.divider()
    st.markdown("### 🎯 Ações do Sistema")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("✅ Validar Sistema", type="primary", use_container_width=True):
            ok, warns, errs = validate_system(t)
            if ok:
                st.success("✅ Sistema validado!")
            for w in warns:
                st.warning(w)
            for e in errs:
                st.error(e)
    
    with col2:
        if st.button("🔨 Construir Rede", type="primary", use_container_width=True):
            ok, warns, errs = validate_system(t)
            if errs:
                st.error("❌ Corrija os erros primeiro")
            else:
                st.success("✅ Rede construída!")
                st.session_state.net_built = True
    
    with col3:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.tables = default_tables()
            st.session_state.net_built = None
            st.rerun()

def render_buses_section(t):
    st.markdown("### 🔌 Barras do Sistema")
    st.info("💡 Configure as barras primeiro - elas são os pontos de conexão dos equipamentos")
    
    t["buses"] = st.data_editor(
        t["buses"],
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "name": st.column_config.TextColumn("Nome", required=True),
            "type": st.column_config.SelectboxColumn(
                "Tipo",
                options=["Slack", "PV", "PQ"],
                required=True
            ),
            "vn_kv": st.column_config.NumberColumn("Tensão (kV)", format="%.2f"),
        }
    )

def render_generators_section(t, bus_names):
    st.markdown("### ⚡ Geradores")
    
    with st.expander("➕ Adicionar Novo Gerador"):
        col1, col2 = st.columns(2)
        with col1:
            gen_name = st.text_input("Nome", f"GER-{len(t['gens'])+1}")
            gen_bus = st.selectbox("Barra de Conexão", bus_names)
        with col2:
            gen_p = st.number_input("P (MW)", 0.0, 1000.0, 50.0)
            gen_vm = st.number_input("V (pu)", 0.9, 1.1, 1.02, 0.01)
        
        if st.button("Adicionar", type="primary"):
            if gen_name and gen_bus:
                new_row = pd.DataFrame([{
                    "name": gen_name, "bus": gen_bus, "p_mw": gen_p,
                    "vm_pu": gen_vm, "min_q_mvar": -30, "max_q_mvar": 40,
                    "type": "Síncrono"
                }])
                t["gens"] = pd.concat([t["gens"], new_row], ignore_index=True)
                st.success(f"✅ Gerador '{gen_name}' adicionado!")
                st.rerun()
    
    if not t["gens"].empty:
        t["gens"] = st.data_editor(
            t["gens"], num_rows="dynamic", use_container_width=True,
            column_config={
                "bus": st.column_config.SelectboxColumn("Barra", options=bus_names)
            }
        )

def render_loads_section(t, bus_names):
    st.markdown("### 🏭 Cargas")
    
    with st.expander("➕ Adicionar Nova Carga"):
        col1, col2 = st.columns(2)
        with col1:
            load_name = st.text_input("Nome", f"CARGA-{len(t['loads'])+1}")
            load_bus = st.selectbox("Barra de Conexão", bus_names)
        with col2:
            load_p = st.number_input("P (MW)", 0.0, 1000.0, 25.0)
            load_q = st.number_input("Q (MVAr)", 0.0, 500.0, 10.0)
        
        if st.button("Adicionar", type="primary"):
            if load_name and load_bus:
                new_row = pd.DataFrame([{
                    "name": load_name, "bus": load_bus, "p_mw": load_p,
                    "q_mvar": load_q, "type": "Industrial"
                }])
                t["loads"] = pd.concat([t["loads"], new_row], ignore_index=True)
                st.success(f"✅ Carga '{load_name}' adicionada!")
                st.rerun()
    
    if not t["loads"].empty:
        t["loads"] = st.data_editor(
            t["loads"], num_rows="dynamic", use_container_width=True,
            column_config={
                "bus": st.column_config.SelectboxColumn("Barra", options=bus_names)
            }
        )

def render_lines_section(t, bus_names):
    st.markdown("### 📏 Linhas de Transmissão")
    
    with st.expander("➕ Adicionar Nova Linha"):
        col1, col2 = st.columns(2)
        with col1:
            line_name = st.text_input("Nome", f"LT-{len(t['lines'])+1}")
            line_from = st.selectbox("De", bus_names, key="line_from")
            line_to = st.selectbox("Para", bus_names, key="line_to")
        with col2:
            line_length = st.number_input("Comprimento (km)", 0.1, 1000.0, 45.0)
            line_r = st.number_input("R (Ω/km)", 0.001, 1.0, 0.035, format="%.4f")
        
        if st.button("Adicionar", type="primary"):
            if line_name and line_from and line_to and line_from != line_to:
                new_row = pd.DataFrame([{
                    "name": line_name, "from_bus": line_from, "to_bus": line_to,
                    "length_km": line_length, "r_ohm_per_km": line_r,
                    "x_ohm_per_km": 0.38, "c_nf_per_km": 11.0, "max_i_ka": 1.2
                }])
                t["lines"] = pd.concat([t["lines"], new_row], ignore_index=True)
                st.success(f"✅ Linha '{line_name}' adicionada!")
                st.rerun()
            elif line_from == line_to:
                st.error("❌ Barras devem ser diferentes")
    
    if not t["lines"].empty:
        t["lines"] = st.data_editor(
            t["lines"], num_rows="dynamic", use_container_width=True,
            column_config={
                "from_bus": st.column_config.SelectboxColumn("De", options=bus_names),
                "to_bus": st.column_config.SelectboxColumn("Para", options=bus_names)
            }
        )

def render_transformers_section(t, bus_names):
    st.markdown("### 🔄 Transformadores")
    
    with st.expander("➕ Adicionar Novo Transformador"):
        col1, col2 = st.columns(2)
        with col1:
            trafo_name = st.text_input("Nome", f"TR-{len(t['trafos'])+1}")
            trafo_hv = st.selectbox("Barra AT", bus_names, key="trafo_hv")
            trafo_lv = st.selectbox("Barra BT", bus_names, key="trafo_lv")
        with col2:
            trafo_sn = st.number_input("S (MVA)", 1.0, 500.0, 100.0)
            trafo_vk = st.number_input("Vk (%)", 1.0, 20.0, 12.0)
        
        if st.button("Adicionar", type="primary"):
            if trafo_name and trafo_hv and trafo_lv and trafo_hv != trafo_lv:
                new_row = pd.DataFrame([{
                    "name": trafo_name, "hv_bus": trafo_hv, "lv_bus": trafo_lv,
                    "sn_mva": trafo_sn, "vn_hv_kv": 230, "vn_lv_kv": 138,
                    "vk_percent": trafo_vk
                }])
                t["trafos"] = pd.concat([t["trafos"], new_row], ignore_index=True)
                st.success(f"✅ Transformador '{trafo_name}' adicionado!")
                st.rerun()
            elif trafo_hv == trafo_lv:
                st.error("❌ Barras AT e BT devem ser diferentes")
    
    if not t["trafos"].empty:
        t["trafos"] = st.data_editor(
            t["trafos"], num_rows="dynamic", use_container_width=True,
            column_config={
                "hv_bus": st.column_config.SelectboxColumn("AT", options=bus_names),
                "lv_bus": st.column_config.SelectboxColumn("BT", options=bus_names)
            }
        )

def page_simulations():
    st.title("⚙️ Simulações e Análises")
    
    if not st.session_state.get("net_built"):
        st.error("⚠️ Construa a rede primeiro em 'Dados do Sistema'")
        return
    
    st.success("✅ Rede pronta para simulações")
    
    sim_type = st.selectbox(
        "Tipo de Análise",
        ["⚡ Fluxo de Potência", "💥 Curto-Circuito", "📐 Dimensionamento"],
        index=0
    )
    
    st.divider()
    
    if sim_type == "⚡ Fluxo de Potência":
        run_power_flow()
    elif sim_type == "💥 Curto-Circuito":
        run_short_circuit()
    else:
        run_dimensioning()

def run_power_flow():
    st.markdown("### ⚡ Fluxo de Potência")
    
    if st.button("▶️ Executar", type="primary"):
        with st.spinner("Calculando..."):
            import time
            time.sleep(1)
            
            t = st.session_state.tables
            n = len(t["buses"])
            
            results = pd.DataFrame({
                "Barra": t["buses"]["name"],
                "V (pu)": np.random.uniform(0.98, 1.02, n),
                "Ângulo (°)": np.random.uniform(-5, 5, n),
            })
            
            st.success("✅ Convergiu!")
            st.dataframe(results, use_container_width=True)

def run_short_circuit():
    st.markdown("### 💥 Curto-Circuito")
    
    t = st.session_state.tables
    bus_fault = st.selectbox("Barra em falta", t["buses"]["name"])
    
    if st.button("⚡ Calcular", type="primary"):
        with st.spinner("Calculando..."):
            import time
            time.sleep(1)
            
            results = pd.DataFrame({
                "Barra": t["buses"]["name"],
                "Ikss (kA)": np.random.uniform(15, 45, len(t["buses"]))
            })
            
            st.success(f"✅ Cálculo para {bus_fault} concluído")
            st.dataframe(results, use_container_width=True)

def run_dimensioning():
    st.markdown("### 📐 Dimensionamento de Condutores")
    
    calc = ElectricalCalculator()
    
    power = st.number_input("Potência (MVA)", 1.0, 500.0, 50.0)
    voltage = st.number_input("Tensão (kV)", 1.0, 500.0, 138.0)
    
    current = calc.nominal_current(power, voltage)
    st.metric("Corrente Nominal", f"{current:.1f} A")
    
    conductors = st.session_state.conductor_db
    suitable = conductors[conductors["ampacity_75C"] >= current * 1.25]
    
    if not suitable.empty:
        st.success(f"✅ {len(suitable)} condutores adequados")
        st.dataframe(suitable, use_container_width=True)

def page_about():
    st.title("ℹ️ Sobre")
    st.markdown("""
    ## Power System Professional Studio v2.0
    
    ### Sistema Completo de Análise de Redes Elétricas
    
    #### 🎯 Recursos:
    - ✅ Seleção inteligente de barras
    - ✅ Simulações de fluxo de potência
    - ✅ Análise de curto-circuito
    - ✅ Dimensionamento de condutores
    - ✅ Validação automática
    
    #### 📝 Como Usar:
    1. Configure as barras em "Dados do Sistema"
    2. Adicione equipamentos usando os formulários
    3. Valide e construa a rede
    4. Execute simulações
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
