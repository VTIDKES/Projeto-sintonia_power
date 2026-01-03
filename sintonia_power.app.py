"""
POWER SYSTEM STUDIO v4.0 - Sistema Completo Avançado
Aplicação profissional para análise de sistemas elétricos de potência

Salve este arquivo como: power_system_studio.py
Execute com: streamlit run power_system_studio.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandapower as pp
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

st.set_page_config(
    page_title="Power System Studio v4.0",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .violation-warning {
        background: #fff3cd;
        padding: 0.75rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 0.5rem 0;
    }
    .violation-error {
        background: #f8d7da;
        padding: 0.75rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
        margin: 0.5rem 0;
    }
    .element-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MODELOS DE DADOS
# ============================================================================

@dataclass
class BusNode:
    id: str
    label: str
    x: float
    y: float
    vn_kv: float
    bus_type: str
    zone: Optional[str] = None
    def to_dict(self):
        return asdict(self)

@dataclass
class LoadNode:
    id: str
    label: str
    parent_bus: str
    p_mw: float
    q_mvar: float
    scaling: float = 1.0
    def to_dict(self):
        return asdict(self)

@dataclass
class GenNode:
    id: str
    label: str
    parent_bus: str
    p_mw: float
    vm_pu: float
    min_q_mvar: float = -50.0
    max_q_mvar: float = 50.0
    def to_dict(self):
        return asdict(self)

@dataclass
class LineEdge:
    id: str
    source: str
    target: str
    length_km: float
    std_type: str
    parallel: int = 1
    in_service: bool = True
    def to_dict(self):
        return asdict(self)

@dataclass
class TransformerEdge:
    id: str
    source: str
    target: str
    std_type: str
    tap_pos: int = 0
    in_service: bool = True
    def to_dict(self):
        return asdict(self)

class PowerSystemModel:
    def __init__(self):
        self.metadata = {"version": "4.0", "created": datetime.now().isoformat(), "name": "Projeto Sem Título"}
        self.buses: Dict[str, BusNode] = {}
        self.loads: Dict[str, LoadNode] = {}
        self.generators: Dict[str, GenNode] = {}
        self.lines: Dict[str, LineEdge] = {}
        self.transformers: Dict[str, TransformerEdge] = {}
    
    def add_bus(self, bus: BusNode):
        self.buses[bus.id] = bus
    
    def add_load(self, load: LoadNode):
        if load.parent_bus not in self.buses:
            raise ValueError(f"Barra '{load.parent_bus}' não existe")
        self.loads[load.id] = load
    
    def add_generator(self, gen: GenNode):
        if gen.parent_bus not in self.buses:
            raise ValueError(f"Barra '{gen.parent_bus}' não existe")
        self.generators[gen.id] = gen
    
    def add_line(self, line: LineEdge):
        if line.source not in self.buses or line.target not in self.buses:
            raise ValueError("Barras de origem/destino não existem")
        self.lines[line.id] = line
    
    def add_transformer(self, trafo: TransformerEdge):
        if trafo.source not in self.buses or trafo.target not in self.buses:
            raise ValueError("Barras de primário/secundário não existem")
        self.transformers[trafo.id] = trafo
    
    def remove_bus(self, bus_id: str):
        if bus_id in self.buses:
            self.loads = {k: v for k, v in self.loads.items() if v.parent_bus != bus_id}
            self.generators = {k: v for k, v in self.generators.items() if v.parent_bus != bus_id}
            self.lines = {k: v for k, v in self.lines.items() if v.source != bus_id and v.target != bus_id}
            self.transformers = {k: v for k, v in self.transformers.items() if v.source != bus_id and v.target != bus_id}
            del self.buses[bus_id]
    
    def to_json(self) -> str:
        return json.dumps({
            "metadata": self.metadata,
            "buses": [b.to_dict() for b in self.buses.values()],
            "loads": [l.to_dict() for l in self.loads.values()],
            "generators": [g.to_dict() for g in self.generators.values()],
            "lines": [l.to_dict() for l in self.lines.values()],
            "transformers": [t.to_dict() for t in self.transformers.values()]
        }, indent=2)
    
    @staticmethod
    def from_json(json_str: str) -> 'PowerSystemModel':
        data = json.loads(json_str)
        model = PowerSystemModel()
        model.metadata = data.get("metadata", model.metadata)
        for b in data.get("buses", []):
            model.add_bus(BusNode(**b))
        for l in data.get("loads", []):
            model.add_load(LoadNode(**l))
        for g in data.get("generators", []):
            model.add_generator(GenNode(**g))
        for ln in data.get("lines", []):
            model.add_line(LineEdge(**ln))
        for t in data.get("transformers", []):
            model.add_transformer(TransformerEdge(**t))
        return model

# ============================================================================
# VALIDAÇÃO E CONVERSÃO
# ============================================================================

class NetworkValidator:
    @staticmethod
    def validate(model: PowerSystemModel) -> Tuple[bool, List[str]]:
        errors = []
        warnings = []
        
        if len(model.buses) == 0:
            errors.append("⚠️ Rede sem barras")
            return False, errors
        
        slack_buses = [b for b in model.buses.values() if b.bus_type == "slack"]
        if len(slack_buses) == 0:
            errors.append("⚠️ Rede sem barra slack")
        elif len(slack_buses) > 1:
            warnings.append(f"ℹ️ {len(slack_buses)} barras slack (usará primeira)")
        
        if not NetworkValidator._is_connected(model):
            errors.append("⚠️ Rede possui ilhas elétricas")
        
        for line in model.lines.values():
            bus_from = model.buses[line.source]
            bus_to = model.buses[line.target]
            if abs(bus_from.vn_kv - bus_to.vn_kv) > 0.1:
                errors.append(f"⚠️ Linha '{line.id}' conecta tensões diferentes - use transformador")
        
        return len(errors) == 0, errors + warnings
    
    @staticmethod
    def _is_connected(model: PowerSystemModel) -> bool:
        if len(model.buses) == 0:
            return True
        graph = {bus_id: [] for bus_id in model.buses.keys()}
        for line in model.lines.values():
            if line.in_service:
                graph[line.source].append(line.target)
                graph[line.target].append(line.source)
        for trafo in model.transformers.values():
            if trafo.in_service:
                graph[trafo.source].append(trafo.target)
                graph[trafo.target].append(trafo.source)
        visited = set()
        start = next(iter(model.buses.keys()))
        NetworkValidator._dfs(start, graph, visited)
        return len(visited) == len(model.buses)
    
    @staticmethod
    def _dfs(node: str, graph: Dict[str, List[str]], visited: set):
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                NetworkValidator._dfs(neighbor, graph, visited)

class PandapowerConverter:
    @staticmethod
    def to_pandapower(model: PowerSystemModel) -> Tuple[pp.pandapowerNet, Dict]:
        net = pp.create_empty_network()
        bus_map = {}
        
        for bus_id, bus in model.buses.items():
            pp_idx = pp.create_bus(net, vn_kv=bus.vn_kv, name=bus.label, geodata=(bus.x, bus.y))
            bus_map[bus_id] = pp_idx
        
        slack_created = False
        for bus_id, bus in model.buses.items():
            if bus.bus_type == "slack" and not slack_created:
                pp.create_ext_grid(net, bus=bus_map[bus_id], vm_pu=1.0, name=f"Grid_{bus.label}")
                slack_created = True
                break
        
        if not slack_created and len(bus_map) > 0:
            pp.create_ext_grid(net, bus=list(bus_map.values())[0], vm_pu=1.0, name="Grid_Auto")
        
        for line in model.lines.values():
            try:
                pp.create_line(net, from_bus=bus_map[line.source], to_bus=bus_map[line.target],
                             length_km=line.length_km, std_type=line.std_type, name=line.id,
                             parallel=line.parallel, in_service=line.in_service)
            except Exception as e:
                st.warning(f"Linha '{line.id}': {str(e)}")
        
        for trafo in model.transformers.values():
            try:
                pp.create_transformer(net, hv_bus=bus_map[trafo.source], lv_bus=bus_map[trafo.target],
                                    std_type=trafo.std_type, name=trafo.id, tap_pos=trafo.tap_pos,
                                    in_service=trafo.in_service)
            except Exception as e:
                st.warning(f"Trafo '{trafo.id}': {str(e)}")
        
        for load in model.loads.values():
            pp.create_load(net, bus=bus_map[load.parent_bus], p_mw=load.p_mw, q_mvar=load.q_mvar,
                         scaling=load.scaling, name=load.label)
        
        for gen in model.generators.values():
            pp.create_gen(net, bus=bus_map[gen.parent_bus], p_mw=gen.p_mw, vm_pu=gen.vm_pu,
                        min_q_mvar=gen.min_q_mvar, max_q_mvar=gen.max_q_mvar, name=gen.label)
        
        return net, bus_map

# ============================================================================
# MOTOR DE SIMULAÇÃO
# ============================================================================

class SimulationEngine:
    @staticmethod
    def run_power_flow(model: PowerSystemModel) -> Dict:
        is_valid, errors = NetworkValidator.validate(model)
        if not is_valid:
            return {"success": False, "converged": False, "errors": errors}
        
        try:
            net, bus_map = PandapowerConverter.to_pandapower(model)
        except Exception as e:
            return {"success": False, "converged": False, "errors": [f"Conversão: {str(e)}"]}
        
        try:
            pp.runpp(net, algorithm='nr', calculate_voltage_angles=True)
            converged = net.converged
        except Exception as e:
            return {"success": False, "converged": False, "errors": [f"Simulação: {str(e)}"]}
        
        if converged:
            results = SimulationEngine._extract_results(net, model, bus_map)
            results["success"] = True
            results["converged"] = True
            results["errors"] = []
            return results
        else:
            return {"success": False, "converged": False, "errors": ["Não convergiu"]}
    
    @staticmethod
    def run_short_circuit(model: PowerSystemModel, fault_bus_id: str) -> Dict:
        try:
            net, bus_map = PandapowerConverter.to_pandapower(model)
            pp.runpp(net)
            pp.shortcircuit.calc_sc(net, fault='3ph', case='max')
            
            fault_idx = bus_map[fault_bus_id]
            ikss_ka = net.res_bus_sc.at[fault_idx, 'ikss_ka']
            
            all_buses_sc = {}
            for bus_id, pp_idx in bus_map.items():
                all_buses_sc[bus_id] = {
                    "ikss_ka": float(net.res_bus_sc.at[pp_idx, 'ikss_ka']),
                    "ip_ka": float(net.res_bus_sc.at[pp_idx, 'ip_ka']) if 'ip_ka' in net.res_bus_sc.columns else 0.0
                }
            
            return {"success": True, "fault_bus": fault_bus_id, "ikss_ka": float(ikss_ka), "all_buses": all_buses_sc}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    @staticmethod
    def _extract_results(net: pp.pandapowerNet, model: PowerSystemModel, bus_map: Dict) -> Dict:
        reverse_bus_map = {v: k for k, v in bus_map.items()}
        results = {"buses": {}, "lines": {}, "transformers": {}, "loads": {}, "generators": {}, "violations": []}
        
        for pp_idx, bus_id in reverse_bus_map.items():
            vm_pu = net.res_bus.at[pp_idx, 'vm_pu']
            va_deg = net.res_bus.at[pp_idx, 'va_degree']
            
            results["buses"][bus_id] = {
                "vm_pu": float(vm_pu),
                "va_degree": float(va_deg),
                "p_mw": float(net.res_bus.at[pp_idx, 'p_mw']),
                "q_mvar": float(net.res_bus.at[pp_idx, 'q_mvar'])
            }
            
            bus_label = model.buses[bus_id].label
            if vm_pu < 0.95:
                results["violations"].append({
                    "type": "subtensão", "severity": "error" if vm_pu < 0.9 else "warning",
                    "element": bus_label, "value": float(vm_pu), "limit": 0.95
                })
            elif vm_pu > 1.05:
                results["violations"].append({
                    "type": "sobretensão", "severity": "error" if vm_pu > 1.1 else "warning",
                    "element": bus_label, "value": float(vm_pu), "limit": 1.05
                })
        
        for pp_idx in net.line.index:
            line_name = net.line.at[pp_idx, 'name']
            loading = net.res_line.at[pp_idx, 'loading_percent']
            results["lines"][line_name] = {
                "loading_percent": float(loading),
                "p_from_mw": float(net.res_line.at[pp_idx, 'p_from_mw']),
                "q_from_mvar": float(net.res_line.at[pp_idx, 'q_from_mvar']),
                "i_ka": float(net.res_line.at[pp_idx, 'i_ka'])
            }
            if loading > 100:
                results["violations"].append({
                    "type": "sobrecarga_linha", "severity": "error" if loading > 120 else "warning",
                    "element": line_name, "value": float(loading), "limit": 100
                })
        
        for pp_idx in net.trafo.index:
            trafo_name = net.trafo.at[pp_idx, 'name']
            loading = net.res_trafo.at[pp_idx, 'loading_percent']
            results["transformers"][trafo_name] = {
                "loading_percent": float(loading),
                "p_hv_mw": float(net.res_trafo.at[pp_idx, 'p_hv_mw']),
                "q_hv_mvar": float(net.res_trafo.at[pp_idx, 'q_hv_mvar'])
            }
            if loading > 100:
                results["violations"].append({
                    "type": "sobrecarga_trafo", "severity": "error" if loading > 120 else "warning",
                    "element": trafo_name, "value": float(loading), "limit": 100
                })
        
        return results

# ============================================================================
# BIBLIOTECA
# ============================================================================

class LibraryManager:
    @staticmethod
    def get_line_types() -> List[str]:
        return ["NAYY 4x50 SE", "NAYY 4x120 SE", "NAYY 4x150 SE", "NA2XS2Y 1x95 RM/25 12/20 kV",
                "NA2XS2Y 1x185 RM/25 12/20 kV", "NA2XS2Y 1x240 RM/25 12/20 kV",
                "15-AL1/3-ST1A 0.4", "24-AL1/4-ST1A 0.4", "48-AL1/8-ST1A 10.0",
                "94-AL1/15-ST1A 10.0", "149-AL1/24-ST1A 10.0", "184-AL1/30-ST1A 10.0"]
    
    @staticmethod
    def get_transformer_types() -> List[str]:
        return ["0.25 MVA 20/0.4 kV", "0.4 MVA 20/0.4 kV", "0.63 MVA 20/0.4 kV",
                "25 MVA 110/20 kV", "40 MVA 110/20 kV", "63 MVA 110/20 kV", "100 MVA 220/110 kV"]
    
    @staticmethod
    def get_standard_voltages() -> List[float]:
        return [0.4, 13.8, 34.5, 69.0, 88.0, 138.0, 230.0, 345.0, 500.0]

# ============================================================================
# VISUALIZAÇÃO
# ============================================================================

def create_network_diagram(model: PowerSystemModel, results: Optional[Dict] = None):
    fig = go.Figure()
    color_map = {"slack": "#dc3545", "pv": "#28a745", "pq": "#007bff"}
    
    if results and results.get("success") and results.get("buses"):
        bus_colors = []
        hover_texts = []
        for bus in model.buses.values():
            if bus.id in results["buses"]:
                vm_pu = results["buses"][bus.id]["vm_pu"]
                va_deg = results["buses"][bus.id]["va_degree"]
                if vm_pu < 0.95:
                    color = "#ffc107"
                elif vm_pu > 1.05:
                    color = "#fd7e14"
                else:
                    color = color_map.get(bus.bus_type, "#6c757d")
                bus_colors.append(color)
                hover_texts.append(f"<b>{bus.label}</b><br>Tensão: {vm_pu:.4f} pu ({bus.vn_kv * vm_pu:.2f} kV)<br>Ângulo: {va_deg:.2f}°<br>Tipo: {bus.bus_type.upper()}")
            else:
                bus_colors.append(color_map.get(bus.bus_type, "#6c757d"))
                hover_texts.append(f"<b>{bus.label}</b><br>Tensão: {bus.vn_kv} kV<br>Tipo: {bus.bus_type.upper()}")
    else:
        bus_colors = [color_map.get(bus.bus_type, "#6c757d") for bus in model.buses.values()]
        hover_texts = [f"<b>{bus.label}</b><br>Tensão: {bus.vn_kv} kV<br>Tipo: {bus.bus_type.upper()}" for bus in model.buses.values()]
    
    for line in model.lines.values():
        if line.source not in model.buses or line.target not in model.buses:
            continue
            
        bus_from = model.buses[line.source]
        bus_to = model.buses[line.target]
        line_color = "#6c757d"
        line_width = 3
        line_hover = f"{line.id}<br>{line.length_km} km"
        if results and results.get("success") and results.get("lines") and line.id in results["lines"]:
            loading = results["lines"][line.id]["loading_percent"]
            line_hover = f"<b>{line.id}</b><br>Carga: {loading:.1f}%<br>P: {results['lines'][line.id]['p_from_mw']:.2f} MW"
            if loading > 100:
                line_color = "#dc3545"
                line_width = 5
            elif loading > 80:
                line_color = "#ffc107"
                line_width = 4
        fig.add_trace(go.Scatter(x=[bus_from.x, bus_to.x], y=[bus_from.y, bus_to.y], mode="lines",
                                line=dict(color=line_color, width=line_width), hoverinfo="text",
                                hovertext=line_hover, showlegend=False))
    
    for trafo in model.transformers.values():
        if trafo.source not in model.buses or trafo.target not in model.buses:
            continue
            
        bus_from = model.buses[trafo.source]
        bus_to = model.buses[trafo.target]
        trafo_color = "#fd7e14"
        trafo_hover = f"<b>🔄 {trafo.id}</b><br>Transformador"
        if results and results.get("success") and results.get("transformers") and trafo.id in results["transformers"]:
            loading = results["transformers"][trafo.id]["loading_percent"]
            trafo_hover = f"<b>🔄 {trafo.id}</b><br>Carga: {loading:.1f}%"
            if loading > 100:
                trafo_color = "#dc3545"
        fig.add_trace(go.Scatter(x=[bus_from.x, bus_to.x], y=[bus_from.y, bus_to.y], mode="lines",
                                line=dict(color=trafo_color, width=4, dash="dash"), hoverinfo="text",
                                hovertext=trafo_hover, showlegend=False))
    
    if len(model.buses) > 0:
        fig.add_trace(go.Scatter(
            x=[bus.x for bus in model.buses.values()],
            y=[bus.y for bus in model.buses.values()],
            mode="markers+text",
            marker=dict(size=30, color=bus_colors, line=dict(width=3, color="white"), symbol="circle"),
            text=[bus.label for bus in model.buses.values()],
            textposition="top center",
            textfont=dict(size=11, color="black", family="Arial Black"),
            hoverinfo="text",
            hovertext=hover_texts,
            showlegend=False
        ))
    
    for load in model.loads.values():
        if load.parent_bus not in model.buses:
            continue
            
        parent_bus = model.buses[load.parent_bus]
        fig.add_trace(go.Scatter(
            x=[parent_bus.x], y=[parent_bus.y - 15], mode="markers+text",
            marker=dict(size=20, color="#ffc107", symbol="triangle-down", line=dict(width=2, color="white")),
            text=["⚡"], textfont=dict(size=14), hoverinfo="text",
            hovertext=f"<b>📊 {load.label}</b><br>P: {load.p_mw} MW<br>Q: {load.q_mvar} MVAr",
            showlegend=False
        ))
    
    for gen in model.generators.values():
        if gen.parent_bus not in model.buses:
            continue
            
        parent_bus = model.buses[gen.parent_bus]
        fig.add_trace(go.Scatter(
            x=[parent_bus.x], y=[parent_bus.y + 15], mode="markers+text",
            marker=dict(size=20, color="#28a745", symbol="square", line=dict(width=2, color="white")),
            text=["🔋"], textfont=dict(size=14), hoverinfo="text",
            hovertext=f"<b>⚙️ {gen.label}</b><br>P: {gen.p_mw} MW<br>V: {gen.vm_pu} pu",
            showlegend=False
        ))
    
    fig.update_layout(
        height=600,
        xaxis=dict(title="Posição X (m)", showgrid=True, zeroline=True, gridcolor="#e0e0e0"),
        yaxis=dict(title="Posição Y (m)", showgrid=True, zeroline=True, gridcolor="#e0e0e0", scaleanchor="x", scaleratio=1),
        hovermode='closest', plot_bgcolor='#f8f9fa', paper_bgcolor='white',
        dragmode='pan', margin=dict(l=50, r=50, t=30, b=50)
    )
    return fig

def display_violations(violations: List[Dict]):
    if not violations:
        st.success("✅ Nenhuma violação detectada!")
        return
    st.error(f"⚠️ {len(violations)} violação(ões) detectada(s)")
    for v in violations:
        severity = v.get("severity", "warning")
        css_class = "violation-error" if severity == "error" else "violation-warning"
        icon = "🔴" if severity == "error" else "⚠️"
        st.markdown(f"""<div class="{css_class}">
            {icon} <b>{v['type'].upper()}</b> em <b>{v['element']}</b><br>
            Valor: {v['value']:.4f} | Limite: {v['limit']:.4f}
        </div>""", unsafe_allow_html=True)

# ============================================================================
# SIMULAÇÃO SIMPLIFICADA
# ============================================================================

def calcular_fluxo_potencia_simplificado(model: PowerSystemModel):
    """Calcula o fluxo de potência do sistema montado (versão simplificada)"""
    if not model.buses:
        return None
    
    geradores = model.generators.values()
    cargas = model.loads.values()
    
    if not geradores:
        return None
    
    # Calcular potências
    P_gerada = sum([g.p_mw for g in geradores])
    Q_gerada = sum([g.vm_pu * 10 for g in geradores])  # Aproximação
    P_carga = sum([l.p_mw for l in cargas])
    Q_carga = sum([l.q_mvar for l in cargas])
    
    # Calcular perdas (simplificado)
    P_perdas = 0
    for linha in model.lines.values():
        # Estimativa de corrente baseada na potência
        I_estimado = P_carga / (len(model.lines) if model.lines else 1)
        P_perdas += 0.01 * (I_estimado ** 2)  # Aproximação
    
    resultados = {
        'P_gerada': P_gerada,
        'Q_gerada': Q_gerada,
        'P_carga': P_carga,
        'Q_carga': Q_carga,
        'P_perdas': P_perdas,
        'eficiencia': (P_carga / P_gerada * 100) if P_gerada > 0 else 0,
        'n_geradores': len(geradores),
        'n_cargas': len(cargas),
        'n_barramentos': len(model.buses),
        'n_linhas': len(model.lines),
        'n_transformers': len(model.transformers),
        'elementos': []
    }
    
    # Detalhes de cada elemento
    for elem in list(model.buses.values()) + list(model.generators.values()) + \
                 list(model.loads.values()) + list(model.lines.values()) + \
                 list(model.transformers.values()):
        if isinstance(elem, BusNode):
            tipo = f"Barra {elem.bus_type.upper()}"
            params = f"Tensão: {elem.vn_kv} kV"
        elif isinstance(elem, GenNode):
            tipo = "Gerador"
            params = f"P: {elem.p_mw} MW, V: {elem.vm_pu} pu"
        elif isinstance(elem, LoadNode):
            tipo = "Carga"
            params = f"P: {elem.p_mw} MW, Q: {elem.q_mvar} MVAr"
        elif isinstance(elem, LineEdge):
            tipo = "Linha"
            params = f"Comprimento: {elem.length_km} km"
        else:
            tipo = "Transformador"
            params = f"Tipo: {elem.std_type}"
        
        resultados['elementos'].append({
            'nome': elem.label if hasattr(elem, 'label') else elem.id,
            'tipo': tipo,
            'parametros': params
        })
    
    return resultados

# ============================================================================
# INICIALIZAÇÃO
# ============================================================================

def init_session_state():
    if "ps_model" not in st.session_state:
        st.session_state.ps_model = PowerSystemModel()
    if "simulation_results" not in st.session_state:
        st.session_state.simulation_results = None
    if "simple_results" not in st.session_state:
        st.session_state.simple_results = None
    if "show_validation" not in st.session_state:
        st.session_state.show_validation = False
    if "custom_voltage" not in st.session_state:
        st.session_state.custom_voltage = 138.0
    if "voltage_mode" not in st.session_state:
        st.session_state.voltage_mode = "Tensões Padrão"
    if "simulation_mode" not in st.session_state:
        st.session_state.simulation_mode = "Pandapower"

init_session_state()

# ============================================================================
# INTERFACE PRINCIPAL
# ============================================================================

st.markdown('<h1 class="main-header">⚡ Power System Studio v4.0</h1>', unsafe_allow_html=True)
st.markdown("**Plataforma Avançada para Análise de Sistemas Elétricos de Potência**")

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.metric("Barras", len(st.session_state.ps_model.buses))
with col_m2:
    st.metric("Linhas", len(st.session_state.ps_model.lines) + len(st.session_state.ps_model.transformers))
with col_m3:
    st.metric("Cargas", len(st.session_state.ps_model.loads))
with col_m4:
    st.metric("Geradores", len(st.session_state.ps_model.generators))

st.markdown("---")

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.header("🔧 Painel de Controle")
    tool = st.radio("Selecione o elemento:", ["🔵 Barra", "➖ Linha", "🔄 Transformador", "📊 Carga", "⚙️ Gerador", "🗑️ Remover"], key="tool_selector")
    st.markdown("---")
    
    # ADICIONAR BARRA
    if "Barra" in tool:
        with st.form("add_bus", clear_on_submit=True):
            st.subheader("➕ Adicionar Barra")
            bus_id = st.text_input("ID único", value=f"bus_{len(st.session_state.ps_model.buses)+1}")
            bus_label = st.text_input("Nome", value="Nova Barra")
            col1, col2 = st.columns(2)
            with col1:
                x = st.number_input("Posição X", value=0.0, step=10.0)
            with col2:
                y = st.number_input("Posição Y", value=0.0, step=10.0)
            
            # Tensão nominal com seletor manual (slider)
            st.markdown("**Tensão Nominal**")
            voltage_mode = st.radio(
                "Modo de seleção:",
                ["Tensões Padrão", "Ajuste Manual"],
                horizontal=True,
                key="voltage_mode"
            )
            
            if voltage_mode == "Tensões Padrão":
                standard_voltages = LibraryManager.get_standard_voltages()
                
                # Usar 138 kV como padrão
                default_voltage = 138.0 if 138.0 in standard_voltages else standard_voltages[0]
                
                vn_kv = st.select_slider(
                    "Selecione a tensão:",
                    options=standard_voltages,
                    value=default_voltage,
                    format_func=lambda x: f"{x} kV",
                    key="voltage_standard_slider"
                )
                st.success(f"✅ Tensão selecionada: **{vn_kv} kV**")
            else:
                # Modo manual com slider contínuo
                vn_kv = st.slider(
                    "Ajuste a tensão (kV):",
                    min_value=0.4,
                    max_value=500.0,
                    value=st.session_state.get('custom_voltage', 138.0),
                    step=0.1,
                    format="%.1f kV",
                    key="voltage_manual_slider"
                )
                st.session_state.custom_voltage = vn_kv
                st.info(f"ℹ️ Tensão personalizada: **{vn_kv} kV**")
            
            bus_type = st.selectbox("Tipo", ["pq", "pv", "slack"], 
                                   help="Slack: Barra de referência\nPV: Barra de geração com controle de tensão\nPQ: Barra de carga")
            
            if st.form_submit_button("➕ Adicionar Barra", use_container_width=True, type="primary"):
                try:
                    if bus_id in st.session_state.ps_model.buses:
                        st.error(f"❌ ID '{bus_id}' já existe!")
                    else:
                        bus = BusNode(id=bus_id, label=bus_label, x=x, y=y, vn_kv=vn_kv, bus_type=bus_type)
                        st.session_state.ps_model.add_bus(bus)
                        st.success(f"✅ Barra '{bus_label}' adicionada!")
                        st.rerun()
                except Exception as e:
                    st.error(f"❌ {str(e)}")
    
    # ADICIONAR LINHA
    elif "Linha" in tool:
        with st.form("add_line", clear_on_submit=True):
            st.subheader("➕ Adicionar Linha")
            if len(st.session_state.ps_model.buses) >= 2:
                bus_ids = list(st.session_state.ps_model.buses.keys())
                bus_labels = {bid: st.session_state.ps_model.buses[bid].label for bid in bus_ids}
                line_id = st.text_input("ID", value=f"line_{len(st.session_state.ps_model.lines)+1}")
                source = st.selectbox("De (origem)", bus_ids, format_func=lambda x: f"{x} ({bus_labels[x]})")
                target = st.selectbox("Para (destino)", bus_ids, format_func=lambda x: f"{x} ({bus_labels[x]})")
                length_km = st.number_input("Comprimento (km)", value=10.0, min_value=0.1, step=1.0)
                std_type = st.selectbox("Tipo de Condutor", LibraryManager.get_line_types())
                parallel = st.number_input("Circuitos em Paralelo", value=1, min_value=1, max_value=4)
                if st.form_submit_button("➕ Adicionar Linha", use_container_width=True, type="primary"):
                    try:
                        if source == target:
                            st.error("❌ Origem e destino devem ser diferentes!")
                        else:
                            st.session_state.ps_model.add_line(LineEdge(line_id, source, target, length_km, std_type, parallel))
                            st.success(f"✅ Linha adicionada!")
                            st.rerun()
                    except Exception as e:
                        st.error(f"❌ {str(e)}")
            else:
                st.info("ℹ️ Adicione pelo menos 2 barras primeiro")
                st.form_submit_button("➕ Adicionar", disabled=True)
    
    # ADICIONAR TRANSFORMADOR
    elif "Transformador" in tool:
        with st.form("add_trafo", clear_on_submit=True):
            st.subheader("➕ Adicionar Transformador")
            if len(st.session_state.ps_model.buses) >= 2:
                bus_ids = list(st.session_state.ps_model.buses.keys())
                bus_labels = {bid: st.session_state.ps_model.buses[bid].label for bid in bus_ids}
                trafo_id = st.text_input("ID", value=f"trafo_{len(st.session_state.ps_model.transformers)+1}")
                source = st.selectbox("Primário (alta tensão)", bus_ids, format_func=lambda x: f"{x} ({bus_labels[x]})")
                target = st.selectbox("Secundário (baixa tensão)", bus_ids, format_func=lambda x: f"{x} ({bus_labels[x]})")
                std_type = st.selectbox("Tipo", LibraryManager.get_transformer_types())
                tap_pos = st.slider("Posição do Tap", -10, 10, 0)
                if st.form_submit_button("➕ Adicionar Transformador", use_container_width=True, type="primary"):
                    try:
                        if source == target:
                            st.error("❌ Primário e secundário devem ser diferentes!")
                        else:
                            st.session_state.ps_model.add_transformer(TransformerEdge(trafo_id, source, target, std_type, tap_pos))
                            st.success(f"✅ Transformador adicionado!")
                            st.rerun()
                    except Exception as e:
                        st.error(f"❌ {str(e)}")
            else:
                st.info("ℹ️ Adicione pelo menos 2 barras primeiro")
                st.form_submit_button("➕ Adicionar", disabled=True)
    
    # ADICIONAR CARGA
    elif "Carga" in tool:
        with st.form("add_load", clear_on_submit=True):
            st.subheader("➕ Adicionar Carga")
            if len(st.session_state.ps_model.buses) > 0:
                bus_ids = list(st.session_state.ps_model.buses.keys())
                bus_labels = {bid: st.session_state.ps_model.buses[bid].label for bid in bus_ids}
                load_id = st.text_input("ID", value=f"load_{len(st.session_state.ps_model.loads)+1}")
                load_label = st.text_input("Nome", value="Nova Carga")
                parent_bus = st.selectbox("Barra", bus_ids, format_func=lambda x: f"{x} ({bus_labels[x]})")
                col1, col2 = st.columns(2)
                with col1:
                    p_mw = st.number_input("P (MW)", value=10.0, min_value=0.0, step=1.0)
                with col2:
                    q_mvar = st.number_input("Q (MVAr)", value=5.0, step=1.0)
                scaling = st.slider("Fator de Escala", 0.0, 2.0, 1.0, 0.1)
                if st.form_submit_button("➕ Adicionar Carga", use_container_width=True, type="primary"):
                    try:
                        st.session_state.ps_model.add_load(LoadNode(load_id, load_label, parent_bus, p_mw, q_mvar, scaling))
                        st.success(f"✅ Carga '{load_label}' adicionada!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ {str(e)}")
            else:
                st.info("ℹ️ Adicione barras primeiro")
                st.form_submit_button("➕ Adicionar", disabled=True)
    
    # ADICIONAR GERADOR
    elif "Gerador" in tool:
        with st.form("add_gen", clear_on_submit=True):
            st.subheader("➕ Adicionar Gerador")
            if len(st.session_state.ps_model.buses) > 0:
                bus_ids = list(st.session_state.ps_model.buses.keys())
                bus_labels = {bid: st.session_state.ps_model.buses[bid].label for bid in bus_ids}
                gen_id = st.text_input("ID", value=f"gen_{len(st.session_state.ps_model.generators)+1}")
                gen_label = st.text_input("Nome", value="Novo Gerador")
                parent_bus = st.selectbox("Barra", bus_ids, format_func=lambda x: f"{x} ({bus_labels[x]})")
                p_mw = st.number_input("Potência (MW)", value=50.0, min_value=0.0, step=5.0)
                vm_pu = st.number_input("Tensão (pu)", value=1.02, min_value=0.9, max_value=1.1, step=0.01)
                col1, col2 = st.columns(2)
                with col1:
                    min_q = st.number_input("Q mín (MVAr)", value=-50.0, step=10.0)
                with col2:
                    max_q = st.number_input("Q máx (MVAr)", value=50.0, step=10.0)
                if st.form_submit_button("➕ Adicionar Gerador", use_container_width=True, type="primary"):
                    try:
                        st.session_state.ps_model.add_generator(GenNode(gen_id, gen_label, parent_bus, p_mw, vm_pu, min_q, max_q))
                        st.success(f"✅ Gerador '{gen_label}' adicionado!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ {str(e)}")
            else:
                st.info("ℹ️ Adicione barras primeiro")
                st.form_submit_button("➕ Adicionar", disabled=True)
    
    # REMOVER ELEMENTOS
    elif "Remover" in tool:
        st.subheader("🗑️ Remover Elementos")
        remove_type = st.selectbox("Tipo", ["Barra", "Linha", "Transformador", "Carga", "Gerador"])
        
        if remove_type == "Barra" and st.session_state.ps_model.buses:
            bus_to_remove = st.selectbox("Selecione a barra", list(st.session_state.ps_model.buses.keys()))
            if st.button("🗑️ Remover Barra", type="secondary"):
                st.session_state.ps_model.remove_bus(bus_to_remove)
                st.success("✅ Barra removida (e elementos dependentes)")
                st.rerun()
        elif remove_type == "Linha" and st.session_state.ps_model.lines:
            line_to_remove = st.selectbox("Selecione a linha", list(st.session_state.ps_model.lines.keys()))
            if st.button("🗑️ Remover Linha", type="secondary"):
                del st.session_state.ps_model.lines[line_to_remove]
                st.success("✅ Linha removida")
                st.rerun()
        elif remove_type == "Transformador" and st.session_state.ps_model.transformers:
            trafo_to_remove = st.selectbox("Selecione", list(st.session_state.ps_model.transformers.keys()))
            if st.button("🗑️ Remover Transformador", type="secondary"):
                del st.session_state.ps_model.transformers[trafo_to_remove]
                st.success("✅ Transformador removido")
                st.rerun()
        elif remove_type == "Carga" and st.session_state.ps_model.loads:
            load_to_remove = st.selectbox("Selecione", list(st.session_state.ps_model.loads.keys()))
            if st.button("🗑️ Remover Carga", type="secondary"):
                del st.session_state.ps_model.loads[load_to_remove]
                st.success("✅ Carga removida")
                st.rerun()
        elif remove_type == "Gerador" and st.session_state.ps_model.generators:
            gen_to_remove = st.selectbox("Selecione", list(st.session_state.ps_model.generators.keys()))
            if st.button("🗑️ Remover Gerador", type="secondary"):
                del st.session_state.ps_model.generators[gen_to_remove]
                st.success("✅ Gerador removida")
                st.rerun()
        else:
            st.info("ℹ️ Nenhum elemento deste tipo para remover")
    
    st.markdown("---")
    
    # GERENCIAMENTO DE PROJETO
    st.header("💾 Projeto")
    project_name = st.text_input("Nome do Projeto", value=st.session_state.ps_model.metadata["name"], key="project_name_input")
    if project_name != st.session_state.ps_model.metadata["name"]:
        st.session_state.ps_model.metadata["name"] = project_name
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Salvar", use_container_width=True):
            json_data = st.session_state.ps_model.to_json()
            st.download_button(
                label="📥 Download JSON",
                data=json_data,
                file_name=f"{project_name.replace(' ', '_')}.json",
                mime="application/json",
                use_container_width=True
            )
    
    with col2:
        uploaded_file = st.file_uploader("📂 Carregar", type=["json"], label_visibility="collapsed")
        if uploaded_file:
            try:
                json_str = uploaded_file.read().decode("utf-8")
                st.session_state.ps_model = PowerSystemModel.from_json(json_str)
                st.session_state.simulation_results = None
                st.success("✅ Projeto carregado!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Erro ao carregar: {str(e)}")
    
    if st.button("🗑️ Limpar Tudo", use_container_width=True):
        st.session_state.ps_model = PowerSystemModel()
        st.session_state.simulation_results = None
        st.session_state.simple_results = None
        st.success("✅ Projeto limpo!")
        st.rerun()
    
    st.markdown("---")
    
    # SIMULAÇÃO
    st.header("⚡ Simulação")
    
    # Modo de simulação
    st.subheader("📊 Método de Cálculo")
    st.session_state.simulation_mode = st.selectbox(
        "Escolha o método:",
        ["Pandapower (Avançado)", "Simplificado"]
    )
    
    if st.button("🔍 Validar Rede", use_container_width=True):
        st.session_state.show_validation = True
    
    if st.button("⚡ Calcular Fluxo de Potência", use_container_width=True, type="primary"):
        if st.session_state.simulation_mode == "Pandapower (Avançado)":
            with st.spinner("⚙️ Calculando com Pandapower..."):
                results = SimulationEngine.run_power_flow(st.session_state.ps_model)
                st.session_state.simulation_results = results
                if results["success"]:
                    st.success("✅ Simulação convergiu!")
                else:
                    st.error("❌ Simulação falhou")
                    for err in results.get("errors", []):
                        st.warning(err)
        else:
            with st.spinner("⚙️ Calculando (método simplificado)..."):
                st.session_state.simple_results = calcular_fluxo_potencia_simplificado(st.session_state.ps_model)
                if st.session_state.simple_results:
                    st.success("✅ Cálculo realizado!")
                else:
                    st.error("❌ Erro no cálculo")
        st.rerun()
    
    # CURTO-CIRCUITO
    if len(st.session_state.ps_model.buses) > 0:
        st.markdown("---")
        st.subheader("⚡ Curto-Circuito")
        fault_bus = st.selectbox("Barra de falta", list(st.session_state.ps_model.buses.keys()), key="sc_bus")
        if st.button("⚡ Calcular CC", use_container_width=True):
            with st.spinner("Calculando curto-circuito..."):
                sc_results = SimulationEngine.run_short_circuit(st.session_state.ps_model, fault_bus)
                if sc_results["success"]:
                    st.success(f"✅ Ikss = {sc_results['ikss_ka']:.2f} kA")
                    with st.expander("Ver todas as correntes"):
                        for bus_id, data in sc_results["all_buses"].items():
                            st.write(f"{bus_id}: {data['ikss_ka']:.2f} kA")
                else:
                    st.error(f"❌ {sc_results.get('error', 'Erro desconhecido')}")

# ============================================================================
# ÁREA PRINCIPAL - TABS
# ============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Diagrama", "📋 Elementos", "📈 Resultados", "📊 Estatísticas", "ℹ️ Ajuda"])

with tab1:
    st.subheader("Diagrama do Sistema Elétrico")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        fig = create_network_diagram(st.session_state.ps_model, st.session_state.simulation_results)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})
    
    with col2:
        st.markdown("### 🎨 Legenda")
        st.markdown("🔴 **Barra Slack**")
        st.markdown("🟢 **Barra PV**")
        st.markdown("🔵 **Barra PQ**")
        st.markdown("⚡ **Carga**")
        st.markdown("🔋 **Gerador**")
        st.markdown("➖ **Linha**")
        st.markdown("🔄 **Transformador**")
        
        if st.session_state.simulation_results and st.session_state.simulation_results.get("violations"):
            st.markdown("---")
            st.markdown("### ⚠️ Violações")
            for v in st.session_state.simulation_results["violations"][:3]:
                icon = "🔴" if v.get("severity") == "error" else "🟡"
                st.markdown(f"{icon} {v['element']}: {v['type']}")

with tab2:
    st.subheader("Elementos do Sistema")
    
    if st.session_state.ps_model.buses:
        # Tabela de barras
        with st.expander("🔵 Barras", expanded=True):
            bus_data = []
            for bus in st.session_state.ps_model.buses.values():
                bus_data.append({
                    "ID": bus.id,
                    "Nome": bus.label,
                    "Tipo": bus.bus_type.upper(),
                    "Tensão (kV)": bus.vn_kv,
                    "Posição": f"({bus.x}, {bus.y})"
                })
            st.dataframe(pd.DataFrame(bus_data), use_container_width=True)
        
        # Tabela de cargas
        if st.session_state.ps_model.loads:
            with st.expander("📊 Cargas"):
                load_data = []
                for load in st.session_state.ps_model.loads.values():
                    load_data.append({
                        "ID": load.id,
                        "Nome": load.label,
                        "Barra": load.parent_bus,
                        "P (MW)": load.p_mw,
                        "Q (MVAr)": load.q_mvar,
                        "Escala": load.scaling
                    })
                st.dataframe(pd.DataFrame(load_data), use_container_width=True)
        
        # Tabela de geradores
        if st.session_state.ps_model.generators:
            with st.expander("⚙️ Geradores"):
                gen_data = []
                for gen in st.session_state.ps_model.generators.values():
                    gen_data.append({
                        "ID": gen.id,
                        "Nome": gen.label,
                        "Barra": gen.parent_bus,
                        "P (MW)": gen.p_mw,
                        "V (pu)": gen.vm_pu,
                        "Q min": gen.min_q_mvar,
                        "Q max": gen.max_q_mvar
                    })
                st.dataframe(pd.DataFrame(gen_data), use_container_width=True)
        
        # Tabela de linhas
        if st.session_state.ps_model.lines:
            with st.expander("➖ Linhas"):
                line_data = []
                for line in st.session_state.ps_model.lines.values():
                    line_data.append({
                        "ID": line.id,
                        "De": line.source,
                        "Para": line.target,
                        "Comprimento (km)": line.length_km,
                        "Tipo": line.std_type,
                        "Paralelo": line.parallel
                    })
                st.dataframe(pd.DataFrame(line_data), use_container_width=True)
        
        # Tabela de transformadores
        if st.session_state.ps_model.transformers:
            with st.expander("🔄 Transformadores"):
                trafo_data = []
                for trafo in st.session_state.ps_model.transformers.values():
                    trafo_data.append({
                        "ID": trafo.id,
                        "Primário": trafo.source,
                        "Secundário": trafo.target,
                        "Tipo": trafo.std_type,
                        "Tap": trafo.tap_pos
                    })
                st.dataframe(pd.DataFrame(trafo_data), use_container_width=True)
    else:
        st.info("👈 Use o painel lateral para adicionar elementos ao sistema")

with tab3:
    st.subheader("Resultados da Simulação")
    
    if st.session_state.simulation_results and st.session_state.simulation_results.get("success"):
        results = st.session_state.simulation_results
        
        if results["violations"]:
            display_violations(results["violations"])
        else:
            st.success("✅ Sem violações!")
        
        st.markdown("---")
        
        with st.expander("🔌 Tensões nas Barras", expanded=True):
            bus_data = []
            for bus_id, data in results["buses"].items():
                bus_label = st.session_state.ps_model.buses[bus_id].label
                vn_kv = st.session_state.ps_model.buses[bus_id].vn_kv
                vm_pu = data['vm_pu']
                status = "✅"
                if vm_pu < 0.95:
                    status = "🟡"
                elif vm_pu > 1.05:
                    status = "🟠"
                bus_data.append({
                    "Status": status,
                    "Barra": bus_label,
                    "V (pu)": f"{vm_pu:.4f}",
                    "V (kV)": f"{vn_kv * vm_pu:.2f}",
                    "θ (°)": f"{data['va_degree']:.2f}"
                })
            st.dataframe(bus_data, use_container_width=True, hide_index=True)
        
        if results["lines"]:
            with st.expander("➖ Carregamento de Linhas"):
                line_data = []
                for line_id, data in results["lines"].items():
                    loading = data['loading_percent']
                    status = "✅"
                    if loading > 100:
                        status = "🔴"
                    elif loading > 80:
                        status = "🟡"
                    line_data.append({
                        "Status": status,
                        "Linha": line_id,
                        "Carga": f"{loading:.1f}%",
                        "P (MW)": f"{data['p_from_mw']:.2f}",
                        "I (kA)": f"{data['i_ka']:.3f}"
                    })
                st.dataframe(line_data, use_container_width=True, hide_index=True)
        
        if results["transformers"]:
            with st.expander("🔄 Transformadores"):
                trafo_data = []
                for trafo_id, data in results["transformers"].items():
                    loading = data['loading_percent']
                    status = "✅" if loading <= 100 else "🔴"
                    trafo_data.append({
                        "Status": status,
                        "Trafo": trafo_id,
                        "Carga": f"{loading:.1f}%",
                        "P (MW)": f"{data['p_hv_mw']:.2f}"
                    })
                st.dataframe(trafo_data, use_container_width=True, hide_index=True)
    
    elif st.session_state.simple_results:
        res = st.session_state.simple_results
        
        # Métricas principais
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Potência Gerada (P)", f"{res['P_gerada']:.2f} MW")
        with col2:
            st.metric("Potência Consumida (P)", f"{res['P_carga']:.2f} MW")
        with col3:
            st.metric("Perdas (P)", f"{res['P_perdas']:.2f} MW")
        with col4:
            st.metric("Eficiência", f"{res['eficiencia']:.1f} %")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Potência Reativa Gerada (Q)", f"{res['Q_gerada']:.2f} MVAr")
        with col2:
            st.metric("Potência Reativa Consumida (Q)", f"{res['Q_carga']:.2f} MVAr")
        
        # Gráficos com Plotly
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Distribuição de Potência Ativa")
            labels = ['Gerada', 'Consumida', 'Perdas']
            sizes = [res['P_gerada'], res['P_carga'], res['P_perdas']]
            colors = ['#FF6B6B', '#4ECDC4', '#FFA07A']
            
            fig = go.Figure(data=[go.Pie(labels=labels, values=sizes, hole=0.3)])
            fig.update_traces(marker=dict(colors=colors))
            fig.update_layout(showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Composição do Sistema")
            labels = ['Geradores', 'Cargas', 'Barramentos', 'Linhas', 'Transformadores']
            sizes = [res['n_geradores'], res['n_cargas'], res['n_barramentos'], res['n_linhas'], res['n_transformers']]
            colors = ['#FF6B6B', '#95E1D3', '#4ECDC4', '#F38181', '#FFA07A']
            
            fig = go.Figure(data=[go.Bar(x=labels, y=sizes, marker_color=colors)])
            fig.update_layout(
                xaxis_title="Elementos",
                yaxis_title="Quantidade",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("👈 Use o botão 'Calcular Fluxo de Potência' no painel lateral")

with tab4:
    st.subheader("📊 Estatísticas e Análises")
    
    if st.session_state.ps_model.buses:
        # Resumo do sistema
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total de Elementos", 
                     len(st.session_state.ps_model.buses) + 
                     len(st.session_state.ps_model.loads) + 
                     len(st.session_state.ps_model.generators) + 
                     len(st.session_state.ps_model.lines) + 
                     len(st.session_state.ps_model.transformers))
        
        with col2:
            total_p_load = sum(load.p_mw for load in st.session_state.ps_model.loads.values())
            st.metric("Demanda Total (P)", f"{total_p_load:.2f} MW")
        
        with col3:
            total_q_load = sum(load.q_mvar for load in st.session_state.ps_model.loads.values())
            st.metric("Demanda Total (Q)", f"{total_q_load:.2f} MVAr")
        
        st.markdown("---")
        
        # Distribuição por tipo de barra
        if st.session_state.ps_model.buses:
            bus_types = {}
            for bus in st.session_state.ps_model.buses.values():
                bus_types[bus.bus_type] = bus_types.get(bus.bus_type, 0) + 1
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Distribuição por Tipo de Barra")
                labels = [f"{k.upper()} ({v})" for k, v in bus_types.items()]
                sizes = list(bus_types.values())
                colors = ['#FF6B6B', '#4ECDC4', '#95E1D3'][:len(sizes)]
                
                fig = go.Figure(data=[go.Pie(labels=labels, values=sizes, hole=0.3)])
                fig.update_traces(marker=dict(colors=colors))
                fig.update_layout(showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Distribuição por Nível de Tensão")
                voltage_levels = {}
                for bus in st.session_state.ps_model.buses.values():
                    level = f"{bus.vn_kv:.0f} kV"
                    voltage_levels[level] = voltage_levels.get(level, 0) + 1
                
                fig = go.Figure(data=[go.Bar(
                    x=list(voltage_levels.keys()),
                    y=list(voltage_levels.values()),
                    marker_color='#007bff'
                )])
                fig.update_layout(
                    xaxis_title="Tensão Nominal",
                    yaxis_title="Quantidade",
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Exportar dados
        st.markdown("---")
        st.subheader("📤 Exportar Dados")
        
        if st.button("📊 Exportar Resultados CSV"):
            if st.session_state.simulation_results:
                bus_df = pd.DataFrame([
                    {
                        "Barra": st.session_state.ps_model.buses[bid].label,
                        "V_pu": data['vm_pu'],
                        "Angulo_deg": data['va_degree'],
                        "P_MW": data['p_mw'],
                        "Q_MVAr": data['q_mvar']
                    }
                    for bid, data in st.session_state.simulation_results["buses"].items()
                ])
                csv = bus_df.to_csv(index=False)
                st.download_button("📥 Download CSV", csv, "resultados_fluxo_potencia.csv", "text/csv", use_container_width=True)
            elif st.session_state.simple_results:
                df = pd.DataFrame(st.session_state.simple_results['elementos'])
                csv = df.to_csv(index=False)
                st.download_button("📥 Download CSV", csv, "resultados_simplificados.csv", "text/csv", use_container_width=True)
    else:
        st.info("👈 Adicione elementos ao sistema para ver estatísticas")

with tab5:
    st.subheader("ℹ️ Informações e Ajuda")
    
    st.markdown("""
    ### Power System Studio v4.0
    
    **Sistema Integrado para Análise de Sistemas Elétricos de Potência**
    
    ### 🎯 Características Principais:
    
    1. **Modelagem Avançada**
       - Barras (Slack, PV, PQ)
       - Geradores com controle de tensão
       - Cargas ativas e reativas
       - Linhas de transmissão com parâmetros reais
       - Transformadores com controle de tap
    
    2. **Análises Disponíveis**
       - Fluxo de potência (métodos NR e simplificado)
       - Cálculo de curto-circuito
       - Detecção de violações
       - Análise de carregamento
    
    3. **Visualização**
       - Diagramas interativos (Plotly)
       - Resultados em tempo real
       - Gráficos e métricas
    
    ### 🚀 Como usar:
    
    1. **Adicionar Elementos**: Use o painel lateral para criar seu sistema
    2. **Configurar**: Ajuste parâmetros de cada elemento
    3. **Conectar**: Estabeleça conexões entre elementos
    4. **Simular**: Execute análises de fluxo de potência
    5. **Analisar**: Visualize resultados e estatísticas
    
    ### 📊 Métodos de Cálculo:
    
    - **Pandapower (Avançado)**: Algoritmo Newton-Raphson completo
    - **Simplificado**: Cálculos rápidos para estimativas
    
    ### 💾 Gerenciamento:
    
    - Salve seus projetos em JSON
    - Carregue projetos existentes
    - Exporte resultados para CSV
    
    ### 🔧 Requisitos:
    
    ```bash
    pip install streamlit pandas numpy plotly pandapower
    ```
    
    ### 📚 Recursos:
    
    - [Documentação Pandapower](https://pandapower.readthedocs.io/)
    - [IEEE Power Systems](https://www.ieee.org/)
    - [Tutoriais de Sistemas de Potência](https://www.powerworld.com/)
    
    ---
    
    **Desenvolvido para engenheiros e estudantes de sistemas elétricos de potência**
    
    Versão 4.0 | © 2024 Power System Studio
    """)

# ============================================================================
# EXEMPLOS RÁPIDOS
# ============================================================================

with st.expander("🚀 Exemplos Rápidos - Carregar Sistema Padrão"):
    st.markdown("### Sistemas de Teste")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📘 IEEE 4 Barras", use_container_width=True):
            model = PowerSystemModel()
            model.metadata["name"] = "IEEE 4 Barras"
            model.add_bus(BusNode("b1", "Barra 1 (Slack)", 0, 100, 138.0, "slack"))
            model.add_bus(BusNode("b2", "Barra 2", 150, 100, 138.0, "pq"))
            model.add_bus(BusNode("b3", "Barra 3", 150, 0, 13.8, "pq"))
            model.add_bus(BusNode("b4", "Barra 4", 300, 50, 13.8, "pq"))
            model.add_line(LineEdge("l1", "b1", "b2", 50.0, "94-AL1/15-ST1A 10.0"))
            model.add_transformer(TransformerEdge("t1", "b2", "b3", "25 MVA 110/20 kV"))
            model.add_line(LineEdge("l2", "b3", "b4", 10.0, "NAYY 4x150 SE"))
            model.add_load(LoadNode("load1", "Carga Industrial", "b4", 20.0, 10.0))
            model.add_generator(GenNode("gen1", "Gerador Local", "b3", 5.0, 1.0))
            st.session_state.ps_model = model
            st.success("✅ Sistema IEEE 4 Barras carregado!")
            st.rerun()
    
    with col2:
        if st.button("📗 Sistema Simples 3 Barras", use_container_width=True):
            model = PowerSystemModel()
            model.metadata["name"] = "Sistema Simples 3 Barras"
            model.add_bus(BusNode("b1", "Subestação", 0, 0, 138.0, "slack"))
            model.add_bus(BusNode("b2", "Indústria", 100, 0, 138.0, "pq"))
            model.add_bus(BusNode("b3", "Comércio", 200, 0, 138.0, "pq"))
            model.add_line(LineEdge("l1", "b1", "b2", 25.0, "94-AL1/15-ST1A 10.0"))
            model.add_line(LineEdge("l2", "b2", "b3", 25.0, "94-AL1/15-ST1A 10.0"))
            model.add_load(LoadNode("load1", "Carga Indústria", "b2", 30.0, 15.0))
            model.add_load(LoadNode("load2", "Carga Comércio", "b3", 20.0, 10.0))
            st.session_state.ps_model = model
            st.success("✅ Sistema 3 Barras carregado!")
            st.rerun()
    
    with col3:
        if st.button("📙 Rede com Geração Distribuída", use_container_width=True):
            model = PowerSystemModel()
            model.metadata["name"] = "Rede com GD"
            model.add_bus(BusNode("b1", "Concessionária", 0, 50, 138.0, "slack"))
            model.add_bus(BusNode("b2", "Alimentador", 100, 50, 13.8, "pq"))
            model.add_bus(BusNode("b3", "Solar Farm", 200, 80, 13.8, "pv"))
            model.add_bus(BusNode("b4", "Consumidor", 200, 20, 13.8, "pq"))
            model.add_transformer(TransformerEdge("t1", "b1", "b2", "25 MVA 110/20 kV"))
            model.add_line(LineEdge("l1", "b2", "b3", 15.0, "NAYY 4x150 SE"))
            model.add_line(LineEdge("l2", "b2", "b4", 15.0, "NAYY 4x150 SE"))
            model.add_generator(GenNode("solar1", "Usina Solar", "b3", 10.0, 1.0))
            model.add_load(LoadNode("load1", "Carga Residencial", "b4", 15.0, 7.0))
            st.session_state.ps_model = model
            st.success("✅ Rede com GD carregada!")
            st.rerun()

# ============================================================================
# RODAPÉ
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem; padding: 1rem;">
    <strong>Power System Studio v4.0</strong> | Sistema Integrado Avançado<br>
    🔧 Núcleo: Python + Pandapower | 🎨 Interface: Streamlit + Plotly<br>
    Desenvolvido para análise técnica de sistemas elétricos de potência<br>
    <br>
    <em>Funcionalidades: Fluxo de Potência • Curto-Circuito • Validação Elétrica • Detecção de Violações • Análise Simplificada</em>
</div>
""", unsafe_allow_html=True)
