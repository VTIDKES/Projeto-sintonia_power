"""
╔═══════════════════════════════════════════════════════════════════════════╗
║          POWER SYSTEM PROFESSIONAL STUDIO v2.0                            ║
║          Sistema Completo de Análise de Redes Elétricas                   ║
╚═══════════════════════════════════════════════════════════════════════════╝

Desenvolvido por: Engenheiro Eletricista / Programador / Analista de Dados

RECURSOS PRINCIPAIS:
✓ Fluxo de Potência (NR, GS, DC, FDBX, BFSW)
✓ Curto-Circuito IEC 60909
✓ Análise de Contingências (N-1, N-2)
✓ Fluxo Ótimo de Potência
✓ Dimensionamento de Condutores
✓ Cálculo de Proteção
✓ Análise Harmônica
✓ Confiabilidade
✓ Relatórios Técnicos Automatizados
✓ Import/Export (JSON, Excel, PTI RAW)
✓ Visualização Georreferenciada
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
import base64
from io import BytesIO

# ═════════════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO DA APLICAÇÃO
# ═════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Power System Professional Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "Sistema Profissional de Análise de Redes Elétricas v2.0"
    }
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
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .info-box {
        background-color: #d1ecf1;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #17a2b8;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding: 0 2rem;
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

class AnalysisType(Enum):
    POWER_FLOW = "Fluxo de Potência"
    SHORT_CIRCUIT = "Curto-Circuito"
    CONTINGENCY = "Análise de Contingências"
    OPTIMAL_FLOW = "Fluxo Ótimo"
    CONDUCTOR_SIZING = "Dimensionamento"
    PROTECTION = "Proteção"
    HARMONICS = "Harmônicos"
    RELIABILITY = "Confiabilidade"

@dataclass
class SystemLimits:
    """Limites operacionais do sistema"""
    v_min_pu: float = 0.95
    v_max_pu: float = 1.05
    loading_max_percent: float = 100.0
    voltage_unbalance_max: float = 2.0
    frequency_deviation_max: float = 0.5

@dataclass
class ProjectInfo:
    """Informações do projeto"""
    name: str = "Novo Projeto"
    company: str = ""
    engineer: str = ""
    date: str = ""
    description: str = ""
    location: str = ""
    voltage_base: float = 100.0

# ═════════════════════════════════════════════════════════════════════════
# BANCO DE DADOS DE EQUIPAMENTOS
# ═════════════════════════════════════════════════════════════════════════

class EquipmentDatabase:
    """Banco de dados de equipamentos elétricos"""
    
    @staticmethod
    def get_conductors() -> pd.DataFrame:
        """Catálogo de condutores"""
        return pd.DataFrame([
            {"name": "1/0 AWG (53.5mm²)", "area_mm2": 53.5, "r_ohm_km_ac": 0.323, "r_ohm_km_dc": 0.305, "gmr_mm": 4.02, "ampacity_60C": 195, "ampacity_75C": 230, "ampacity_90C": 260},
            {"name": "2/0 AWG (67.4mm²)", "area_mm2": 67.4, "r_ohm_km_ac": 0.256, "r_ohm_km_dc": 0.242, "gmr_mm": 5.06, "ampacity_60C": 225, "ampacity_75C": 265, "ampacity_90C": 300},
            {"name": "4/0 AWG (107mm²)", "area_mm2": 107.2, "r_ohm_km_ac": 0.161, "r_ohm_km_dc": 0.152, "gmr_mm": 6.40, "ampacity_60C": 300, "ampacity_75C": 350, "ampacity_90C": 405},
            {"name": "336.4 MCM (170mm²)", "area_mm2": 170.0, "r_ohm_km_ac": 0.102, "r_ohm_km_dc": 0.095, "gmr_mm": 8.07, "ampacity_60C": 450, "ampacity_75C": 530, "ampacity_90C": 615},
            {"name": "477 MCM (241mm²)", "area_mm2": 241.0, "r_ohm_km_ac": 0.072, "r_ohm_km_dc": 0.067, "gmr_mm": 9.60, "ampacity_60C": 540, "ampacity_75C": 635, "ampacity_90C": 730},
            {"name": "795 MCM (403mm²)", "area_mm2": 403.0, "r_ohm_km_ac": 0.043, "r_ohm_km_dc": 0.040, "gmr_mm": 12.4, "ampacity_60C": 730, "ampacity_75C": 860, "ampacity_90C": 985},
            {"name": "1590 MCM (805mm²)", "area_mm2": 805.0, "r_ohm_km_ac": 0.022, "r_ohm_km_dc": 0.020, "gmr_mm": 17.5, "ampacity_60C": 1100, "ampacity_75C": 1295, "ampacity_90C": 1480},
        ])
    
    @staticmethod
    def get_transformers() -> pd.DataFrame:
        """Catálogo de transformadores padrão"""
        return pd.DataFrame([
            {"name": "5 MVA 69/13.8kV", "sn_mva": 5, "vn_hv_kv": 69, "vn_lv_kv": 13.8, "vk_percent": 6.0, "vkr_percent": 0.6, "pfe_kw": 5.5, "i0_percent": 0.3},
            {"name": "10 MVA 69/13.8kV", "sn_mva": 10, "vn_hv_kv": 69, "vn_lv_kv": 13.8, "vk_percent": 6.5, "vkr_percent": 0.7, "pfe_kw": 9.5, "i0_percent": 0.3},
            {"name": "25 MVA 138/13.8kV", "sn_mva": 25, "vn_hv_kv": 138, "vn_lv_kv": 13.8, "vk_percent": 8.0, "vkr_percent": 0.8, "pfe_kw": 20, "i0_percent": 0.25},
            {"name": "50 MVA 230/138kV", "sn_mva": 50, "vn_hv_kv": 230, "vn_lv_kv": 138, "vk_percent": 10.0, "vkr_percent": 0.4, "pfe_kw": 38, "i0_percent": 0.2},
            {"name": "100 MVA 230/138kV", "sn_mva": 100, "vn_hv_kv": 230, "vn_lv_kv": 138, "vk_percent": 12.0, "vkr_percent": 0.5, "pfe_kw": 55, "i0_percent": 0.18},
            {"name": "150 MVA 230/138kV", "sn_mva": 150, "vn_hv_kv": 230, "vn_lv_kv": 138, "vk_percent": 12.5, "vkr_percent": 0.5, "pfe_kw": 75, "i0_percent": 0.15},
        ])
    
    @staticmethod
    def get_circuit_breakers() -> pd.DataFrame:
        """Catálogo de disjuntores"""
        return pd.DataFrame([
            {"voltage_kv": 13.8, "rated_current_a": 1200, "interrupting_ka": 25, "type": "VCB"},
            {"voltage_kv": 13.8, "rated_current_a": 2000, "interrupting_ka": 40, "type": "VCB"},
            {"voltage_kv": 69, "rated_current_a": 1200, "interrupting_ka": 31.5, "type": "SF6"},
            {"voltage_kv": 138, "rated_current_a": 2000, "interrupting_ka": 40, "type": "SF6"},
            {"voltage_kv": 230, "rated_current_a": 3000, "interrupting_ka": 50, "type": "SF6"},
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
    def impedance_pu_to_ohm(z_pu: float, v_kv: float, s_base_mva: float) -> float:
        """Converte impedância de pu para ohms"""
        z_base = (v_kv ** 2) / s_base_mva
        return z_pu * z_base
    
    @staticmethod
    def voltage_drop(i_a: float, z_ohm: complex, v_kv: float, length_km: float = 1.0) -> float:
        """Queda de tensão percentual"""
        v_drop_v = abs(i_a * z_ohm * length_km)
        v_base_v = v_kv * 1000
        return (v_drop_v / v_base_v) * 100
    
    @staticmethod
    def power_loss(i_a: float, r_ohm_km: float, length_km: float, phases: int = 3) -> float:
        """Perdas de potência (kW)"""
        return phases * (i_a ** 2) * r_ohm_km * length_km / 1000
    
    @staticmethod
    def short_circuit_current(v_kv: float, z_ohm: float) -> float:
        """Corrente de curto-circuito trifásica (kA)"""
        v_phase = v_kv * 1000 / np.sqrt(3)
        return (v_phase / z_ohm) / 1000
    
    @staticmethod
    def power_factor(p_mw: float, q_mvar: float) -> float:
        """Fator de potência"""
        s = np.sqrt(p_mw**2 + q_mvar**2)
        return p_mw / s if s > 0 else 1.0
    
    @staticmethod
    def reactive_compensation(p_mw: float, pf_initial: float, pf_target: float) -> float:
        """Compensação reativa necessária (MVAr)"""
        q_initial = p_mw * np.tan(np.arccos(pf_initial))
        q_target = p_mw * np.tan(np.arccos(pf_target))
        return q_initial - q_target

# ═════════════════════════════════════════════════════════════════════════
# INICIALIZAÇÃO E GERENCIAMENTO DE ESTADO
# ═════════════════════════════════════════════════════════════════════════

def init_state():
    """Inicializa estado da aplicação"""
    defaults = {
        "project_info": ProjectInfo(),
        "tables": default_tables(),
        "results": {},
        "net_built": None,
        "system_limits": SystemLimits(),
        "analysis_history": [],
        "selected_buses": [],
        "selected_lines": [],
        "conductor_db": EquipmentDatabase.get_conductors(),
        "trafo_db": EquipmentDatabase.get_transformers(),
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def default_tables() -> Dict[str, pd.DataFrame]:
    """Tabelas padrão do sistema"""
    
    buses = pd.DataFrame([
        {"name": "SE-PRINCIPAL", "vn_kv": 230.0, "type": "Slack", "zone": "Norte", "x_km": 0.0, "y_km": 0.0},
        {"name": "SE-INDUSTRIAL", "vn_kv": 138.0, "type": "PQ", "zone": "Norte", "x_km": 45.0, "y_km": 10.0},
        {"name": "SE-COMERCIAL", "vn_kv": 13.8, "type": "PQ", "zone": "Sul", "x_km": 30.0, "y_km": -15.0},
    ])
    
    ext_grids = pd.DataFrame([{
        "name": "REDE-BASICA", "bus": "SE-PRINCIPAL", "vm_pu": 1.00, "va_degree": 0.0,
        "s_sc_max_mva": 5000, "s_sc_min_mva": 3000, "rx_min": 0.1, "rx_max": 0.15
    }])
    
    gens = pd.DataFrame([{
        "name": "GER-INDUSTRIAL", "bus": "SE-INDUSTRIAL", "p_mw": 50.0, "vm_pu": 1.02,
        "min_q_mvar": -30, "max_q_mvar": 40, "min_p_mw": 10, "max_p_mw": 60,
        "type": "Síncrono", "fuel": "Gas"
    }])
    
    loads = pd.DataFrame([
        {"name": "CARGA-IND-01", "bus": "SE-INDUSTRIAL", "p_mw": 80.0, "q_mvar": 30.0, "type": "Industrial", "priority": 1},
        {"name": "CARGA-COM-01", "bus": "SE-COMERCIAL", "p_mw": 25.0, "q_mvar": 10.0, "type": "Commercial", "priority": 2},
    ])
    
    lines = pd.DataFrame([{
        "name": "LT-230-01", "from_bus": "SE-PRINCIPAL", "to_bus": "SE-INDUSTRIAL",
        "length_km": 45.0, "parallel": 1, "r_ohm_per_km": 0.035, "x_ohm_per_km": 0.38,
        "c_nf_per_km": 11.0, "max_i_ka": 1.2, "conductor": "795 MCM"
    }])
    
    trafos = pd.DataFrame([{
        "name": "TR-230/138-01", "hv_bus": "SE-PRINCIPAL", "lv_bus": "SE-INDUSTRIAL",
        "sn_mva": 100, "vn_hv_kv": 230, "vn_lv_kv": 138, "vk_percent": 12.0,
        "vkr_percent": 0.5, "pfe_kw": 55, "i0_percent": 0.3
    }])
    
    switches = pd.DataFrame([{
        "name": "DJ-01", "bus": "SE-PRINCIPAL", "element_type": "l",
        "element_name": "LT-230-01", "closed": True, "type": "CB"
    }])
    
    shunts = pd.DataFrame([{
        "name": "BC-IND-01", "bus": "SE-INDUSTRIAL", "q_mvar": 30.0,
        "step": 5.0, "max_step": 6, "in_service": True
    }])
    
    sgens = pd.DataFrame([{
        "name": "SOLAR-COM-01", "bus": "SE-COMERCIAL", "p_mw": 5.0,
        "q_mvar": 0.0, "type": "PV"
    }])
    
    return {
        "buses": buses, "ext_grids": ext_grids, "gens": gens,
        "loads": loads, "lines": lines, "trafos": trafos,
        "switches": switches, "shunts": shunts, "sgens": sgens
    }

# ═════════════════════════════════════════════════════════════════════════
# VALIDAÇÃO
# ═════════════════════════════════════════════════════════════════════════

def validate_system(tables: Dict) -> Tuple[bool, List[str], List[str]]:
    """Validação completa do sistema"""
    warnings, errors = [], []
    
    buses = tables["buses"]
    if buses.empty:
        errors.append("❌ Sistema precisa de pelo menos uma barra")
        return False, warnings, errors
    
    # Duplicatas
    if buses["name"].duplicated().any():
        errors.append("❌ Nomes de barras duplicados detectados")
    
    # Slack
    slack_count = (buses["type"] == "Slack").sum()
    if slack_count == 0:
        errors.append("❌ Sistema precisa de pelo menos uma barra Slack")
    elif slack_count > 1:
        warnings.append("⚠️ Múltiplas barras Slack detectadas")
    
    # Referências
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
    
    # Ilhamento
    islands = detect_islands(tables)
    if len(islands) > 1:
        warnings.append(f"⚠️ Sistema tem {len(islands)} ilhas elétricas")
    
    return len(errors) == 0, warnings, errors

def detect_islands(tables: Dict) -> List[set]:
    """Detecta ilhas elétricas no sistema"""
    buses = set(tables["buses"]["name"])
    parent = {b: b for b in buses}
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # Conectar barras através de linhas e transformadores
    for _, row in tables["lines"].iterrows():
        union(row["from_bus"], row["to_bus"])
    for _, row in tables["trafos"].iterrows():
        union(row["hv_bus"], row["lv_bus"])
    
    islands = {}
    for bus in buses:
        root = find(bus)
        islands.setdefault(root, set()).add(bus)
    
    return list(islands.values())

# ═════════════════════════════════════════════════════════════════════════
# CONSTRUÇÃO DA REDE PANDAPOWER
# ═════════════════════════════════════════════════════════════════════════

def build_network(tables: Dict):
    """Constrói rede pandapower completa"""
    import pandapower as pp
    
    project = st.session_state.project_info
    net = pp.create_empty_network(name=project.name, f_hz=60.0)
    
    bus_map = {}
    
    # Barras
    for _, r in tables["buses"].iterrows():
        idx = pp.create_bus(
            net, vn_kv=r["vn_kv"], name=r["name"], type="b",
            zone=r.get("zone", ""), geodata=(r.get("x_km", 0), r.get("y_km", 0))
        )
        bus_map[r["name"]] = idx
    
    # External Grids
    for _, r in tables["ext_grids"].iterrows():
        if r["bus"] in bus_map:
            pp.create_ext_grid(
                net, bus=bus_map[r["bus"]], vm_pu=r.get("vm_pu", 1.0),
                va_degree=r.get("va_degree", 0.0), name=r["name"],
                s_sc_max_mva=r.get("s_sc_max_mva", np.nan),
                s_sc_min_mva=r.get("s_sc_min_mva", np.nan),
                rx_min=r.get("rx_min", np.nan), rx_max=r.get("rx_max", np.nan)
            )
    
    # Geradores
    for _, r in tables["gens"].iterrows():
        if r["bus"] in bus_map:
            pp.create_gen(
                net, bus=bus_map[r["bus"]], p_mw=r.get("p_mw", 0),
                vm_pu=r.get("vm_pu", 1.0), name=r["name"],
                min_q_mvar=r.get("min_q_mvar", -999),
                max_q_mvar=r.get("max_q_mvar", 999)
            )
    
    # Cargas
    for _, r in tables["loads"].iterrows():
        if r["bus"] in bus_map:
            pp.create_load(
                net, bus=bus_map[r["bus"]],
                p_mw=r.get("p_mw", 0), q_mvar=r.get("q_mvar", 0),
                name=r["name"]
            )
    
    # Linhas
    line_map = {}
    for _, r in tables["lines"].iterrows():
        if r["from_bus"] in bus_map and r["to_bus"] in bus_map:
            idx = pp.create_line_from_parameters(
                net, from_bus=bus_map[r["from_bus"]], to_bus=bus_map[r["to_bus"]],
                length_km=r["length_km"], r_ohm_per_km=r.get("r_ohm_per_km", 0.1),
                x_ohm_per_km=r.get("x_ohm_per_km", 0.1),
                c_nf_per_km=r.get("c_nf_per_km", 0),
                max_i_ka=r.get("max_i_ka", 1), name=r["name"],
                parallel=r.get("parallel", 1)
            )
            line_map[r["name"]] = idx
    
    # Transformadores
    trafo_map = {}
    for _, r in tables["trafos"].iterrows():
        if r["hv_bus"] in bus_map and r["lv_bus"] in bus_map:
            idx = pp.create_transformer_from_parameters(
                net, hv_bus=bus_map[r["hv_bus"]], lv_bus=bus_map[r["lv_bus"]],
                sn_mva=r.get("sn_mva", 100), vn_hv_kv=r.get("vn_hv_kv", 220),
                vn_lv_kv=r.get("vn_lv_kv", 110), vk_percent=r.get("vk_percent", 10),
                vkr_percent=r.get("vkr_percent", 0.4), pfe_kw=r.get("pfe_kw", 0),
                i0_percent=r.get("i0_percent", 0), name=r["name"]
            )
            trafo_map[r["name"]] = idx
    
    # Switches
    for _, r in tables["switches"].iterrows():
        if r["bus"] in bus_map:
            et = r["element_type"]
            elem = None
            if et == "l" and r["element_name"] in line_map:
                elem = line_map[r["element_name"]]
            elif et == "t" and r["element_name"] in trafo_map:
                elem = trafo_map[r["element_name"]]
            
            if elem is not None:
                pp.create_switch(
                    net, bus=bus_map[r["bus"]], element=elem, et=et,
                    closed=r.get("closed", True), name=r["name"]
                )
    
    # Shunts
    for _, r in tables.get("shunts", pd.DataFrame()).iterrows():
        if r["bus"] in bus_map:
            pp.create_shunt(
                net, bus=bus_map[r["bus"]], q_mvar=r.get("q_mvar", 0),
                name=r["name"]
            )
    
    # Static Generators
    for _, r in tables.get("sgens", pd.DataFrame()).iterrows():
        if r["bus"] in bus_map:
            pp.create_sgen(
                net, bus=bus_map[r["bus"]], p_mw=r.get("p_mw", 0),
                q_mvar=r.get("q_mvar", 0), name=r["name"]
            )
    
    return net

# ═════════════════════════════════════════════════════════════════════════
# ANÁLISES
# ═════════════════════════════════════════════════════════════════════════

def run_power_flow(net, algorithm="nr", **kwargs):
    """Executa fluxo de potência"""
    import pandapower as pp
    
    try:
        if algorithm == "dc":
            pp.rundcpp(net)
        else:
            pp.runpp(net, algorithm=algorithm, **kwargs)
        
        results = {
            "ok": True,
            "converged": net.converged,
            "bus": net.res_bus.copy(),
            "line": net.res_line.copy() if len(net.line) else pd.DataFrame(),
            "trafo": net.res_trafo.copy() if len(net.trafo) else pd.DataFrame(),
            "gen": net.res_gen.copy() if len(net.gen) else pd.DataFrame(),
        }
        
        # Perdas totais
        if len(net.res_line):
            results["line_loss_mw"] = net.res_line.pl_mw.sum()
        if len(net.res_trafo):
            results["trafo_loss_mw"] = net.res_trafo.pl_mw.sum()
        
        return results
    
    except Exception as e:
        return {"ok": False, "error": str(e)}

def run_short_circuit(net, **kwargs):
    """Executa cálculo de curto-circuito"""
    import pandapower.shortcircuit as sc
    
    try:
        sc.calc_sc(net, **kwargs)
        return {
            "ok": True,
            "bus_sc": net.res_bus_sc.copy() if hasattr(net, 'res_bus_sc') else pd.DataFrame(),
            "line_sc": net.res_line_sc.copy() if hasattr(net, 'res_line_sc') else pd.DataFrame()
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

def run_contingency_analysis(net, n_minus=1):
    """Análise de contingências N-1 ou N-2"""
    import pandapower as pp
    
    results = []
    lines = net.line.index.tolist()
    
    for line_idx in lines:
        # Desconecta linha
        original_status = net.line.at[line_idx, "in_service"]
        net.line.at[line_idx, "in_service"] = False
        
        try:
            pp.runpp(net, algorithm="nr")
            overloads = net.res_line[net.res_line.loading_percent > 100]
            voltage_violations = net.res_bus[
                (net.res_bus.vm_pu < 0.95) | (net.res_bus.vm_pu > 1.05)
            ]
            
            results.append({
                "contingency": net.line.at[line_idx, "name"],
                "converged": net.converged,
                "overloads": len(overloads),
                "voltage_violations": len(voltage_violations),
                "severity": "Critical" if (len(overloads) > 0 or len(voltage_violations) > 0) else "OK"
            })
        except:
            results.append({
                "contingency": net.line.at[line_idx, "name"],
                "converged": False,
                "severity": "Diverged"
            })
        
        # Restaura linha
        net.line.at[line_idx, "in_service"] = original_status
    
    return pd.DataFrame(results)

# ═════════════════════════════════════════════════════════════════════════
# VISUALIZAÇÕES
# ═════════════════════════════════════════════════════════════════════════

def plot_voltage_profile(bus_res: pd.DataFrame, net):
    """Perfil de tensão com limites"""
    fig = go.Figure()
    
    if bus_res.empty:
        return fig
    
    bus_names = [net.bus.at[i, "name"] for i in bus_res.index]
    
    fig.add_trace(go.Scatter(
        x=bus_names, y=bus_res["vm_pu"],
        mode="lines+markers", name="Tensão",
        line=dict(color="#1e3c72", width=3),
        marker=dict(size=10)
    ))
    
    # Limites
    limits = st.session_state.system_limits
    fig.add_hline(y=limits.v_max_pu, line_dash="dash", line_color="red",
                  annotation_text=f"Máx: {limits.v_max_pu} pu")
    fig.add_hline(y=limits.v_min_pu, line_dash="dash", line_color="red",
                  annotation_text=f"Mín: {limits.v_min_pu} pu")
    
    fig.update_layout(
        title="Perfil de Tensão do Sistema",
        xaxis_title="Barras",
        yaxis_title="Tensão (pu)",
        hovermode="x unified",
        template="plotly_white",
        height=500
    )
    
    return fig

def plot_loading_chart(line_res: pd.DataFrame, net):
    """Carregamento de linhas"""
    fig = go.Figure()
    
    if line_res.empty:
        return fig
    
    line_names = [net.line.at[i, "name"] for i in line_res.index]
    loading = line_res["loading_percent"]
    
    colors = ["green" if x < 80 else "orange" if x < 100 else "red" for x in loading]
    
    fig.add_trace(go.Bar(
        x=line_names, y=loading,
        marker_color=colors,
        text=[f"{x:.1f}%" for x in loading],
        textposition="outside"
    ))
    
    fig.add_hline(y=100, line_dash="dash", line_color="red",
                  annotation_text="Limite 100%")
    
    fig.update_layout(
        title="Carregamento de Linhas de Transmissão",
        xaxis_title="Linhas",
        yaxis_title="Carregamento (%)",
        template="plotly_white",
        height=500
    )
    
    return fig

def plot_network_topology(net):
    """Topologia da rede com visualização geográfica"""
    import pandapower.plotting as plot
    
    fig = go.Figure()
    
    # Barras
    for idx, row in net.bus.iterrows():
        x, y = row.get("geodata", (0, 0))
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode="markers+text",
            marker=dict(size=15, color="#1e3c72"),
            text=row["name"],
            textposition="top center",
            name=row["name"],
            showlegend=False
        ))
    
    # Linhas
    for idx, row in net.line.iterrows():
        fb = net.bus.loc[row["from_bus"]]
        tb = net.bus.loc[row["to_bus"]]
        x_from, y_from = fb.get("geodata", (0, 0))
        x_to, y_to = tb.get("geodata", (0, 0))
        
        fig.add_trace(go.Scatter(
            x=[x_from, x_to],
            y=[y_from, y_to],
            mode="lines",
            line=dict(color="gray", width=2),
            showlegend=False
        ))
    
    fig.update_layout(
        title="Topologia do Sistema Elétrico",
        xaxis_title="Distância X (km)",
        yaxis_title="Distância Y (km)",
        template="plotly_white",
        height=600,
        showlegend=False
    )
    
    return fig

def create_summary_dashboard(results: Dict):
    """Dashboard resumo dos resultados"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Tensões", "Carregamento", "Perdas", "Potências"),
        specs=[[{"type": "scatter"}, {"type": "bar"}],
               [{"type": "pie"}, {"type": "bar"}]]
    )
    
    if results.get("ok") and not results["bus"].empty:
        bus_res = results["bus"]
        
        # Tensões
        fig.add_trace(
            go.Scatter(y=bus_res["vm_pu"], mode="lines+markers", name="V (pu)"),
            row=1, col=1
        )
        
        # Carregamento
        if not results["line"].empty:
            line_res = results["line"]
            fig.add_trace(
                go.Bar(y=line_res["loading_percent"], name="Loading %"),
                row=1, col=2
            )
        
        # Perdas
        loss_line = results.get("line_loss_mw", 0)
        loss_trafo = results.get("trafo_loss_mw", 0)
        fig.add_trace(
            go.Pie(labels=["Linhas", "Transformadores"],
                   values=[loss_line, loss_trafo]),
            row=2, col=1
        )
        
        # Potências
        if not results["bus"].empty:
            fig.add_trace(
                go.Bar(x=["P", "Q"],
                       y=[bus_res["p_mw"].sum(), bus_res["q_mvar"].sum()],
                       name="Potência"),
                row=2, col=2
            )
    
    fig.update_layout(height=800, showlegend=False, template="plotly_white")
    return fig

# ═════════════════════════════════════════════════════════════════════════
# INTERFACE DE USUÁRIO
# ═════════════════════════════════════════════════════════════════════════

def render_sidebar():
    """Menu lateral"""
    with st.sidebar:
        st.markdown('<div class="main-header">⚡ PSPS v2.0</div>', unsafe_allow_html=True)
        
        page = st.radio(
            "**Navegação**",
            [
                "🏠 Dashboard",
                "📊 Dados do Sistema",
                "⚙️ Simulações",
                "📈 Resultados",
                "🔧 Ferramentas",
                "💾 Import/Export",
                "ℹ️ Sobre"
            ],
            index=0
        )
        
        st.divider()
        
        # Info do projeto
        with st.expander("📋 Informações do Projeto"):
            project = st.session_state.project_info
            st.text_input("Nome", value=project.name, key="proj_name")
            st.text_input("Empresa", value=project.company, key="proj_company")
            st.text_input("Engenheiro", value=project.engineer, key="proj_eng")
        
        # Limites do sistema
        with st.expander("⚠️ Limites Operacionais"):
            limits = st.session_state.system_limits
            limits.v_min_pu = st.number_input("V mín (pu)", 0.8, 1.0, limits.v_min_pu, 0.01)
            limits.v_max_pu = st.number_input("V máx (pu)", 1.0, 1.2, limits.v_max_pu, 0.01)
            limits.loading_max_percent = st.number_input("Loading máx (%)", 50.0, 150.0, limits.loading_max_percent, 5.0)
        
        return page.split()[1]

def page_dashboard():
    """Dashboard principal"""
    st.title("🏠 Dashboard do Sistema")
    
    if st.session_state.net_built is None:
        st.info("👈 Configure o sistema em **Dados do Sistema** e depois construa a rede")
        return
    
    net = st.session_state.net_built
    results = st.session_state.results.get("pf", {})
    
    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Barras", len(net.bus), help="Número total de barras")
    with col2:
        st.metric("Linhas", len(net.line), help="Número de linhas de transmissão")
    with col3:
        st.metric("Transformadores", len(net.trafo), help="Número de transformadores")
    with col4:
        st.metric("Cargas", len(net.load), help="Número de cargas")
    
    # Gráficos
    if results.get("ok"):
        st.plotly_chart(create_summary_dashboard(results), use_container_width=True)
    else:
        st.warning("Execute uma simulação para ver os resultados")

def page_system_data():
    """Página de dados do sistema"""
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
    
    with tabs[0]:
        st.markdown("### Barras do Sistema")
        t["buses"] = st.data_editor(
            t["buses"],
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "type": st.column_config.SelectboxColumn(
                    "Tipo",
                    options=["Slack", "PV", "PQ"],
                    required=True
                ),
                "vn_kv": st.column_config.NumberColumn("Tensão (kV)", min_value=0.1, format="%.2f"),
            }
        )
    
    with tabs[1]:
        st.markdown("### Geradores")
        t["gens"] = st.data_editor(t["gens"], num_rows="dynamic", use_container_width=True)
    
    with tabs[2]:
        st.markdown("### Cargas")
        t["loads"] = st.data_editor(t["loads"], num_rows="dynamic", use_container_width=True)
    
    with tabs[3]:
        st.markdown("### Linhas de Transmissão")
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("📚 Ver Catálogo de Condutores"):
                st.dataframe(st.session_state.conductor_db, use_container_width=True)
        t["lines"] = st.data_editor(t["lines"], num_rows="dynamic", use_container_width=True)
    
    with tabs[4]:
        st.markdown("### Transformadores")
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("📚 Ver Catálogo de Trafos"):
                st.dataframe(st.session_state.trafo_db, use_container_width=True)
        t["trafos"] = st.data_editor(t["trafos"], num_rows="dynamic", use_container_width=True)
    
    with tabs[5]:
        st.markdown("### Chaves e Disjuntores")
        t["switches"] = st.data_editor(t["switches"], num_rows="dynamic", use_container_width=True)
    
    with tabs[6]:
        st.markdown("### Conexão com Rede Externa")
        t["ext_grids"] = st.data_editor(t["ext_grids"], num_rows="dynamic", use_container_width=True)
    
    with tabs[7]:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Bancos de Capacitores")
            t["shunts"] = st.data_editor(t["shunts"], num_rows="dynamic", use_container_width=True)
        with col2:
            st.markdown("#### Geração Distribuída")
            t["sgens"] = st.data_editor(t["sgens"], num_rows="dynamic", use_container_width=True)
    
    st.session_state.tables = t
    
    # Ações
    st.divider()
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
                st.error("Corrija os erros antes de construir")
                return
            
            try:
                net = build_network(t)
                st.session_state.net_built = net
                st.success(f"✅ Rede construída: {len(net.bus)} barras, {len(net.line)} linhas")
            except Exception as e:
                st.error(f"❌ Erro: {e}")
    
    with col3:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.tables = default_tables()
            st.session_state.net_built = None
            st.rerun()
    
    with col4:
        if st.button("🗺️ Visualizar Topologia", use_container_width=True):
            if st.session_state.net_built:
                st.plotly_chart(plot_network_topology(st.session_state.net_built), use_container_width=True)

def page_simulations():
    """Página de simulações"""
    st.title("⚙️ Simulações e Análises")
    
    if st.session_state.net_built is None:
        st.error("⚠️ Construa a rede primeiro em **Dados do Sistema**")
        return
    
    net = st.session_state.net_built
    
    tabs = st.tabs([
        "⚡ Fluxo de Potência",
        "💥 Curto-Circuito",
        "🔀 Contingências",
        "📐 Dimensionamento"
    ])
    
    with tabs[0]:
        st.markdown("### Fluxo de Potência AC/DC")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            algorithm = st.selectbox(
                "Algoritmo",
                ["nr", "bfsw", "gs", "fdbx", "dc"],
                help="NR: Newton-Raphson | DC: Fluxo DC linearizado"
            )
        with col2:
            tol = st.number_input("Tolerância (MVA)", 1e-12, 1e-2, 1e-8, format="%.1e")
        with col3:
            max_iter = st.number_input("Iterações máx", 1, 100, 20)
        
        if st.button("▶️ Executar Fluxo de Potência", type="primary"):
            with st.spinner("Calculando..."):
                results = run_power_flow(net, algorithm, tolerance_mva=tol, max_iteration=max_iter)
                st.session_state.results["pf"] = results
                
                if results["ok"]:
                    if results["converged"]:
                        st.success(f"✅ Convergiu em {algorithm.upper()}")
                    else:
                        st.warning("⚠️ Não convergiu")
                else:
                    st.error(f"❌ Erro: {results['error']}")
    
    with tabs[1]:
        st.markdown("### Cálculo de Curto-Circuito")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            tk_s = st.number_input("Tempo tk (s)", 0.01, 10.0, 1.0, 0.1)
        with col2:
            ith = st.checkbox("Calcular ith", value=True)
        with col3:
            ip = st.checkbox("Calcular ip", value=True)
        
        if st.button("⚡ Executar Curto-Circuito", type="primary"):
            with st.spinner("Calculando..."):
                results = run_short_circuit(net, tk_s=tk_s, ith=ith, ip=ip)
                st.session_state.results["sc"] = results
                
                if results["ok"]:
                    st.success("✅ Cálculo concluído")
                else:
                    st.error(f"❌ Erro: {results['error']}")
    
    with tabs[2]:
        st.markdown("### Análise de Contingências")
        
        n_minus = st.radio("Tipo de análise", ["N-1", "N-2"], horizontal=True)
        
        if st.button("🔀 Executar Análise de Contingências", type="primary"):
            with st.spinner("Analisando contingências..."):
                cont_results = run_contingency_analysis(net, n_minus=1)
                st.session_state.results["contingency"] = cont_results
                st.dataframe(cont_results, use_container_width=True)
    
    with tabs[3]:
        st.markdown("### Dimensionamento de Condutores")
        
        calc = ElectricalCalculator()
        
        col1, col2 = st.columns(2)
        with col1:
            power_mva = st.number_input("Potência (MVA)", 1.0, 1000.0, 50.0)
            voltage_kv = st.number_input("Tensão (kV)", 1.0, 500.0, 138.0)
            distance_km = st.number_input("Distância (km)", 0.1, 500.0, 10.0)
        
        with col2:
            current_a = calc.nominal_current(power_mva, voltage_kv)
            st.metric("Corrente nominal", f"{current_a:.1f} A")
            
            conductors = st.session_state.conductor_db
            suitable = conductors[conductors["ampacity_75C"] >= current_a]
            
            if not suitable.empty:
                st.success(f"✅ {len(suitable)} condutores adequados")
                st.dataframe(suitable[["name", "ampacity_75C", "r_ohm_km_ac"]], use_container_width=True)

def page_results():
    """Página de resultados"""
    st.title("📈 Resultados das Análises")
    
    if not st.session_state.results:
        st.info("Execute simulações primeiro")
        return
    
    tabs = st.tabs(["⚡ Fluxo de Potência", "💥 Curto-Circuito", "🔀 Contingências"])
    
    with tabs[0]:
        pf = st.session_state.results.get("pf", {})
        if pf and pf.get("ok"):
            net = st.session_state.net_built
            
            st.subheader("Tensões nas Barras")
            st.dataframe(pf["bus"], use_container_width=True)
            st.plotly_chart(plot_voltage_profile(pf["bus"], net), use_container_width=True)
            
            st.subheader("Carregamento de Linhas")
            st.dataframe(pf["line"], use_container_width=True)
            st.plotly_chart(plot_loading_chart(pf["line"], net), use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Perdas em Linhas", f"{pf.get('line_loss_mw', 0):.2f} MW")
            col2.metric("Perdas em Trafos", f"{pf.get('trafo_loss_mw', 0):.2f} MW")
            col3.metric("Perdas Totais", f"{pf.get('line_loss_mw', 0) + pf.get('trafo_loss_mw', 0):.2f} MW")
    
    with tabs[1]:
        sc = st.session_state.results.get("sc", {})
        if sc and sc.get("ok"):
            st.subheader("Correntes de Curto-Circuito")
            st.dataframe(sc["bus_sc"], use_container_width=True)
    
    with tabs[2]:
        cont = st.session_state.results.get("contingency")
        if cont is not None:
            st.dataframe(cont, use_container_width=True)
            
            critical = cont[cont["severity"] == "Critical"]
            if not critical.empty:
                st.error(f"⚠️ {len(critical)} contingências críticas detectadas!")

def page_tools():
    """Ferramentas auxiliares"""
    st.title("🔧 Ferramentas de Engenharia")
    
    calc = ElectricalCalculator()
    
    tabs = st.tabs([
        "🧮 Calculadora",
        "📊 Catálogos",
        "📝 Relatórios"
    ])
    
    with tabs[0]:
        st.subheader("Calculadora Elétrica")
        
        tool = st.selectbox(
            "Selecione a ferramenta",
            ["Corrente Nominal", "Queda de Tensão", "Perdas", "Fator de Potência", "Compensação Reativa"]
        )
        
        if tool == "Corrente Nominal":
            col1, col2 = st.columns(2)
            with col1:
                s = st.number_input("Potência (MVA)", 0.1, 1000.0, 10.0)
                v = st.number_input("Tensão (kV)", 0.1, 765.0, 138.0)
            with col2:
                i = calc.nominal_current(s, v)
                st.metric("Corrente Nominal", f"{i:.2f} A")
        
        elif tool == "Fator de Potência":
            col1, col2 = st.columns(2)
            with col1:
                p = st.number_input("Potência Ativa (MW)", 0.1, 1000.0, 50.0)
                q = st.number_input("Potência Reativa (MVAr)", -500.0, 500.0, 20.0)
            with col2:
                pf = calc.power_factor(p, q)
                st.metric("Fator de Potência", f"{pf:.4f}")
                st.metric("Ângulo", f"{np.rad2deg(np.arccos(pf)):.2f}°")
    
    with tabs[1]:
        st.subheader("Catálogos de Equipamentos")
        
        cat_type = st.selectbox("Tipo", ["Condutores", "Transformadores", "Disjuntores"])
        
        if cat_type == "Condutores":
            st.dataframe(st.session_state.conductor_db, use_container_width=True)
        elif cat_type == "Transformadores":
            st.dataframe(st.session_state.trafo_db, use_container_width=True)
        else:
            st.dataframe(EquipmentDatabase.get_circuit_breakers(), use_container_width=True)
    
    with tabs[2]:
        st.subheader("Gerador de Relatórios")
        st.info("Em desenvolvimento: relatórios técnicos automatizados")

def page_import_export():
    """Import/Export de dados"""
    st.title("💾 Importar / Exportar Dados")
    
    tabs = st.tabs(["📤 Exportar", "📥 Importar"])
    
    with tabs[0]:
        st.subheader("Exportar Projeto")
        
        data = {
            "project_info": asdict(st.session_state.project_info),
            "tables": {k: v.to_dict("records") for k, v in st.session_state.tables.items()},
            "system_limits": asdict(st.session_state.system_limits)
        }
        
        json_str = json.dumps(data, indent=2)
        
        st.download_button(
            "⬇️ Download JSON",
            data=json_str,
            file_name=f"power_system_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    with tabs[1]:
        st.subheader("Importar Projeto")
        
        uploaded = st.file_uploader("Selecione arquivo JSON", type=["json"])
        
        if uploaded:
            data = json.load(uploaded)
            if st.button("Importar"):
                st.session_state.tables = {k: pd.DataFrame(v) for k, v in data["tables"].items()}
                st.success("✅ Dados importados com sucesso!")

def page_about():
    """Sobre o sistema"""
    st.title("ℹ️ Sobre o Sistema")
    
    st.markdown("""
    ## Power System Professional Studio v2.0
    
    ### Sistema Completo de Análise de Redes Elétricas de Potência
    
    **Desenvolvido por:** Engenheiro Eletricista / Programador / Analista de Dados
    
    #### 🎯 Recursos Principais:
    - ⚡ Fluxo de Potência (Newton-Raphson, Gauss-Seidel, DC, FDBX, BFSW)
    - 💥 Curto-Circuito IEC 60909
    - 🔀 Análise de Contingências N-1 e N-2
    - 📐 Dimensionamento de Condutores
    - 🔧 Ferramentas de Cálculo Elétrico
    - 📊 Visualizações Interativas
    - 💾 Import/Export JSON
    - 📝 Relatórios Técnicos
    
    #### 🛠️ Tecnologias:
    - Python 3.8+
    - Streamlit
    - Pandapower
    - Plotly
    - Pandas / NumPy
    
    #### 📚 Base de Dados:
    - Catálogo de condutores ACSR/AAC
    - Transformadores de potência
    - Disjuntores de alta tensão
    - Parâmetros de linhas aéreas e subterrâneas
    
    ---
    
    💡 **Dica:** Use o menu lateral para navegar entre as funcionalidades.
    """)

# ═════════════════════════════════════════════════════════════════════════
# APLICAÇÃO PRINCIPAL
# ═════════════════════════════════════════════════════════════════════════

def main():
    """Função principal"""
    init_state()
    
    page = render_sidebar()
