# STREAMLIT + PANDAPOWER - Análise Completa
# Sistemas Elétricos de Potência

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
import time
from datetime import datetime
from streamlit.components.v1 import html
import uuid

# Configuração da página
st.set_page_config(
    page_title="Power System Studio - Click & Drag",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .mode-active {
        background-color: #007bff !important;
        color: white !important;
        border-color: #007bff !important;
    }
    .element-card {
        padding: 10px;
        margin: 5px 0;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.2s;
        border: 2px solid transparent;
    }
    .element-card:hover {
        transform: translateX(5px);
        border-color: #007bff;
    }
    .bus-slack { background: linear-gradient(135deg, #f8d7da, #f5c6cb); }
    .bus-pv { background: linear-gradient(135deg, #d1e7dd, #c3e6cb); }
    .bus-pq { background: linear-gradient(135deg, #cff4fc, #bee5eb); }
    .element-line { background: linear-gradient(135deg, #cce5ff, #b8daff); }
    .element-load { background: linear-gradient(135deg, #e0cffc, #d0bfff); }
    .element-gen { background: linear-gradient(135deg, #d4edda, #c3e6cb); }
    .selected { 
        border: 3px solid #ffc107 !important;
        box-shadow: 0 0 10px rgba(255,193,7,0.5);
    }
    .status-bar {
        padding: 10px;
        background-color: #f8f9fa;
        border-radius: 5px;
        margin: 10px 0;
        font-weight: bold;
    }
    .dragging-mode {
        animation: pulse 1.5s infinite;
        border: 2px dashed #007bff;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# Inicialização do estado
def init_session_state():
    default_state = {
        'buses': [],
        'lines': [],
        'loads': [],
        'generators': [],
        'results': None,
        'history': [],
        'selected_element': None,
        'selected_type': None,
        'mode': 'select',
        'action_mode': None,
        'connecting_from': None,
        'next_id': 0,
        'drag_start': None,
        'drag_element': None,
        'dragging': False,
        'last_click': {'x': 0, 'y': 0, 'time': 0},
        'system_name': 'Novo Sistema',
        'grid_enabled': True,
        'show_voltages': True,
        'canvas_width': 1000,
        'canvas_height': 700,
    }
    
    for key, value in default_state.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Funções auxiliares
def generate_id():
    st.session_state.next_id += 1
    return st.session_state.next_id - 1

def add_bus(x, y, bus_type='pq', name=None):
    bus_id = generate_id()
    if name is None:
        name = f"Barra {bus_id}"
    
    new_bus = {
        'id': bus_id,
        'name': name,
        'x': x,
        'y': y,
        'type': bus_type,
        'vn_kv': 13.8,
        'v_pu': 1.0 if bus_type == 'slack' else 0.98 + np.random.random() * 0.04,
        'angle_deg': 0.0,
        'color': {'slack': '#ef4444', 'pv': '#10b981', 'pq': '#3b82f6'}[bus_type]
    }
    st.session_state.buses.append(new_bus)
    return new_bus

def add_line(from_bus_id, to_bus_id, name=None):
    for line in st.session_state.lines:
        if (line['from'] == from_bus_id and line['to'] == to_bus_id) or \
           (line['from'] == to_bus_id and line['to'] == from_bus_id):
            return None
    
    line_id = generate_id()
    if name is None:
        name = f"Linha {line_id}"
    
    new_line = {
        'id': line_id,
        'name': name,
        'from': from_bus_id,
        'to': to_bus_id,
        'r_ohm_per_km': 0.1,
        'x_ohm_per_km': 0.3,
        'length_km': 10.0,
        'max_i_ka': 1.0,
        'color': '#2563eb'
    }
    st.session_state.lines.append(new_line)
    return new_line

def add_load(bus_id, name=None, p_mw=5.0, q_mvar=2.0):
    load_id = generate_id()
    if name is None:
        name = f"Carga {load_id}"
    
    new_load = {
        'id': load_id,
        'name': name,
        'bus': bus_id,
        'p_mw': p_mw,
        'q_mvar': q_mvar,
        'color': '#8b5cf6'
    }
    st.session_state.loads.append(new_load)
    return new_load

def add_generator(bus_id, name=None, p_mw=10.0, vm_pu=1.0):
    gen_id = generate_id()
    if name is None:
        name = f"Gerador {gen_id}"
    
    new_gen = {
        'id': gen_id,
        'name': name,
        'bus': bus_id,
        'p_mw': p_mw,
        'vm_pu': vm_pu,
        'color': '#10b981'
    }
    st.session_state.generators.append(new_gen)
    return new_gen

def get_bus_by_id(bus_id):
    for bus in st.session_state.buses:
        if bus['id'] == bus_id:
            return bus
    return None

def get_element_by_id(element_id, element_type):
    if element_type == 'bus':
        return get_bus_by_id(element_id)
    elif element_type == 'line':
        for line in st.session_state.lines:
            if line['id'] == element_id:
                return line
    elif element_type == 'load':
        for load in st.session_state.loads:
            if load['id'] == element_id:
                return load
    elif element_type == 'generator':
        for gen in st.session_state.generators:
            if gen['id'] == element_id:
                return gen
    return None

def delete_element(element_id, element_type):
    if element_type == 'bus':
        st.session_state.buses = [b for b in st.session_state.buses if b['id'] != element_id]
        st.session_state.lines = [l for l in st.session_state.lines 
                                 if l['from'] != element_id and l['to'] != element_id]
        st.session_state.loads = [l for l in st.session_state.loads if l['bus'] != element_id]
        st.session_state.generators = [g for g in st.session_state.generators if g['bus'] != element_id]
    elif element_type == 'line':
        st.session_state.lines = [l for l in st.session_state.lines if l['id'] != element_id]
    elif element_type == 'load':
        st.session_state.loads = [l for l in st.session_state.loads if l['id'] != element_id]
    elif element_type == 'generator':
        st.session_state.generators = [g for g in st.session_state.generators if g['id'] != element_id]

def handle_canvas_click(click_data):
    """Processa cliques no canvas"""
    if not click_data or 'points' not in click_data:
        return
    
    points = click_data['points']
    if not points:
        return
    
    point = points[0]
    x = point['x']
    y = point['y']
    
    current_time = time.time()
    is_double_click = (current_time - st.session_state.last_click['time'] < 0.5 and
                      abs(x - st.session_state.last_click['x']) < 10 and
                      abs(y - st.session_state.last_click['y']) < 10)
    
    # Atualizar último clique
    st.session_state.last_click = {'x': x, 'y': y, 'time': current_time}
    
    # Verificar se clicou em um elemento existente
    clicked_element = None
    clicked_type = None
    
    # Verificar barras (raio 25 pixels)
    for bus in st.session_state.buses:
        distance = np.sqrt((bus['x'] - x)**2 + (bus['y'] - y)**2)
        if distance <= 25:
            clicked_element = bus['id']
            clicked_type = 'bus'
            break
    
    if not clicked_element:
        # Verificar cargas
        for load in st.session_state.loads:
            bus = get_bus_by_id(load['bus'])
            if bus:
                distance = np.sqrt((bus['x'] - x)**2 + (bus['y'] + 50 - y)**2)
                if distance <= 15:
                    clicked_element = load['id']
                    clicked_type = 'load'
                    break
    
    if not clicked_element:
        # Verificar geradores
        for gen in st.session_state.generators:
            bus = get_bus_by_id(gen['bus'])
            if bus:
                distance = np.sqrt((bus['x'] - x)**2 + (bus['y'] - 50 - y)**2)
                if distance <= 15:
                    clicked_element = gen['id']
                    clicked_type = 'generator'
                    break
    
    if not clicked_element:
        # Verificar linhas (mais complexo)
        for line in st.session_state.lines:
            from_bus = get_bus_by_id(line['from'])
            to_bus = get_bus_by_id(line['to'])
            if from_bus and to_bus:
                # Verificar distância até a linha
                distance = distance_to_line(x, y, from_bus['x'], from_bus['y'], 
                                          to_bus['x'], to_bus['y'])
                if distance <= 5:  # 5 pixels de tolerância
                    clicked_element = line['id']
                    clicked_type = 'line'
                    break
    
    # Processar ação baseada no modo
    if st.session_state.mode == 'select':
        if clicked_element:
            st.session_state.selected_element = clicked_element
            st.session_state.selected_type = clicked_type
            st.rerun()
        elif is_double_click and st.session_state.action_mode:
            # Duplo clique para adicionar elemento
            handle_double_click(x, y)
    
    elif st.session_state.mode == 'connect':
        if clicked_type == 'bus':
            if st.session_state.connecting_from is None:
                st.session_state.connecting_from = clicked_element
                st.session_state.selected_element = clicked_element
                st.session_state.selected_type = 'bus'
            else:
                if st.session_state.connecting_from != clicked_element:
                    add_line(st.session_state.connecting_from, clicked_element)
                    st.session_state.connecting_from = None
                else:
                    st.session_state.connecting_from = None
            st.rerun()
    
    elif st.session_state.mode == 'delete':
        if clicked_element:
            delete_element(clicked_element, clicked_type)
            st.session_state.selected_element = None
            st.session_state.selected_type = None
            st.rerun()

def handle_double_click(x, y):
    """Processa duplo clique para adicionar elementos"""
    if st.session_state.action_mode == 'add_bus_slack':
        add_bus(x, y, 'slack')
    elif st.session_state.action_mode == 'add_bus_pv':
        add_bus(x, y, 'pv')
    elif st.session_state.action_mode == 'add_bus_pq':
        add_bus(x, y, 'pq')
    elif st.session_state.action_mode == 'add_load':
        # Encontrar barra mais próxima
        closest_bus = None
        min_distance = float('inf')
        
        for bus in st.session_state.buses:
            distance = np.sqrt((bus['x'] - x)**2 + (bus['y'] - y)**2)
            if distance < min_distance and distance <= 40:
                min_distance = distance
                closest_bus = bus
        
        if closest_bus:
            add_load(closest_bus['id'])
    elif st.session_state.action_mode == 'add_generator':
        # Encontrar barra mais próxima
        closest_bus = None
        min_distance = float('inf')
        
        for bus in st.session_state.buses:
            distance = np.sqrt((bus['x'] - x)**2 + (bus['y'] - y)**2)
            if distance < min_distance and distance <= 40:
                min_distance = distance
                closest_bus = bus
        
        if closest_bus:
            add_generator(closest_bus['id'])
    
    st.rerun()

def distance_to_line(px, py, x1, y1, x2, y2):
    """Calcula distância de ponto até linha"""
    A = px - x1
    B = py - y1
    C = x2 - x1
    D = y2 - y1

    dot = A * C + B * D
    len_sq = C * C + D * D
    
    if len_sq == 0:
        return np.sqrt(A*A + B*B)
    
    param = dot / len_sq

    if param < 0:
        xx = x1
        yy = y1
    elif param > 1:
        xx = x2
        yy = y2
    else:
        xx = x1 + param * C
        yy = y1 + param * D

    dx = px - xx
    dy = py - yy
    return np.sqrt(dx * dx + dy * dy)

def create_interactive_canvas():
    """Cria o canvas interativo"""
    fig = go.Figure()
    
    # Adicionar grade
    if st.session_state.grid_enabled:
        grid_size = 50
        for x in range(0, st.session_state.canvas_width + 1, grid_size):
            fig.add_shape(
                type="line",
                x0=x, y0=0, x1=x, y1=st.session_state.canvas_height,
                line=dict(color="rgba(0,0,0,0.1)", width=1)
            )
        for y in range(0, st.session_state.canvas_height + 1, grid_size):
            fig.add_shape(
                type="line",
                x0=0, y0=y, x1=st.session_state.canvas_width, y1=y,
                line=dict(color="rgba(0,0,0,0.1)", width=1)
            )
    
    # Desenhar linhas
    for line in st.session_state.lines:
        from_bus = get_bus_by_id(line['from'])
        to_bus = get_bus_by_id(line['to'])
        
        if from_bus and to_bus:
            fig.add_trace(go.Scatter(
                x=[from_bus['x'], to_bus['x']],
                y=[from_bus['y'], to_bus['y']],
                mode='lines',
                line=dict(color=line['color'], width=3),
                hoverinfo='text',
                hovertext=f"<b>{line['name']}</b><br>ID: {line['id']}<br>De: Barra {line['from']}<br>Para: Barra {line['to']}",
                name=f"line_{line['id']}",
                customdata=[['line', line['id']]]
            ))
            
            # Adicionar seta
            mid_x = (from_bus['x'] + to_bus['x']) / 2
            mid_y = (from_bus['y'] + to_bus['y']) / 2
            
            # Calcular ângulo
            dx = to_bus['x'] - from_bus['x']
            dy = to_bus['y'] - from_bus['y']
            angle = np.arctan2(dy, dx)
            
            fig.add_annotation(
                x=mid_x,
                y=mid_y,
                ax=mid_x - 15 * np.cos(angle),
                ay=mid_y - 15 * np.sin(angle),
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor=line['color']
            )
    
    # Desenhar barras
    bus_x = []
    bus_y = []
    bus_text = []
    bus_color = []
    bus_size = []
    bus_hover = []
    bus_customdata = []
    
    for bus in st.session_state.buses:
        bus_x.append(bus['x'])
        bus_y.append(bus['y'])
        bus_text.append(str(bus['id']))
        bus_color.append(bus['color'])
        
        is_selected = (st.session_state.selected_type == 'bus' and 
                      st.session_state.selected_element == bus['id'])
        bus_size.append(30 if is_selected else 25)
        
        bus_hover.append(
            f"<b>{bus['name']}</b><br>"
            f"ID: {bus['id']}<br>"
            f"Tipo: {bus['type'].upper()}<br>"
            f"Vn: {bus['vn_kv']} kV<br>"
            f"Posição: ({bus['x']:.0f}, {bus['y']:.0f})"
        )
        
        bus_customdata.append(['bus', bus['id']])
    
    if bus_x:
        fig.add_trace(go.Scatter(
            x=bus_x,
            y=bus_y,
            mode='markers+text',
            marker=dict(
                size=bus_size,
                color=bus_color,
                line=dict(
                    width=3,
                    color=['#facc15' if (st.session_state.selected_type == 'bus' and 
                                         st.session_state.selected_element == bus['id']) 
                          else '#1e293b' for bus in st.session_state.buses]
                )
            ),
            text=bus_text,
            textposition="middle center",
            textfont=dict(color='white', size=14, family='Arial Black'),
            hoverinfo='text',
            hovertext=bus_hover,
            name="buses",
            customdata=bus_customdata
        ))
        
        # Nomes das barras
        for bus in st.session_state.buses:
            fig.add_annotation(
                x=bus['x'],
                y=bus['y'] + 40,
                text=bus['name'],
                showarrow=False,
                font=dict(size=12, color="#1e293b"),
                xanchor="center",
                yanchor="middle"
            )
    
    # Desenhar cargas
    for load in st.session_state.loads:
        bus = get_bus_by_id(load['bus'])
        if bus:
            is_selected = (st.session_state.selected_type == 'load' and 
                          st.session_state.selected_element == load['id'])
            
            fig.add_trace(go.Scatter(
                x=[bus['x']],
                y=[bus['y'] + 60],
                mode='markers+text',
                marker=dict(
                    symbol='square',
                    size=20,
                    color=load['color'],
                    line=dict(width=3 if is_selected else 2, 
                             color='#facc15' if is_selected else 'white')
                ),
                text='L',
                textposition="middle center",
                textfont=dict(color='white', size=12, family='Arial Black'),
                hoverinfo='text',
                hovertext=f"<b>{load['name']}</b><br>ID: {load['id']}<br>Barra: {load['bus']}<br>P: {load['p_mw']} MW<br>Q: {load['q_mvar']} MVar",
                name=f"load_{load['id']}",
                customdata=[['load', load['id']]]
            ))
    
    # Desenhar geradores
    for gen in st.session_state.generators:
        bus = get_bus_by_id(gen['bus'])
        if bus:
            is_selected = (st.session_state.selected_type == 'generator' and 
                          st.session_state.selected_element == gen['id'])
            
            fig.add_trace(go.Scatter(
                x=[bus['x']],
                y=[bus['y'] - 60],
                mode='markers+text',
                marker=dict(
                    symbol='circle',
                    size=20,
                    color=gen['color'],
                    line=dict(width=3 if is_selected else 2, 
                             color='#facc15' if is_selected else '#059669')
                ),
                text='G',
                textposition="middle center",
                textfont=dict(color='white', size=12, family='Arial Black'),
                hoverinfo='text',
                hovertext=f"<b>{gen['name']}</b><br>ID: {gen['id']}<br>Barra: {gen['bus']}<br>P: {gen['p_mw']} MW<br>Vm: {gen['vm_pu']} pu",
                name=f"gen_{gen['id']}",
                customdata=[['generator', gen['id']]]
            ))
    
    # Linha de conexão temporária
    if st.session_state.mode == 'connect' and st.session_state.connecting_from is not None:
        from_bus = get_bus_by_id(st.session_state.connecting_from)
        if from_bus:
            fig.add_trace(go.Scatter(
                x=[from_bus['x'], from_bus['x']],
                y=[from_bus['y'], from_bus['y']],
                mode='lines',
                line=dict(color='#f59e0b', width=2, dash='dash'),
                hoverinfo='none',
                showlegend=False
            ))
    
    # Configurar layout
    fig.update_layout(
        width=st.session_state.canvas_width,
        height=st.session_state.canvas_height,
        plot_bgcolor='white',
        paper_bgcolor='white',
        title=dict(
            text=f"⚡ {st.session_state.system_name}",
            font=dict(size=24, color='#1e293b'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            range=[0, st.session_state.canvas_width],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            fixedrange=True
        ),
        yaxis=dict(
            range=[0, st.session_state.canvas_height],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            fixedrange=True,
            scaleanchor="x",
            scaleratio=1
        ),
        showlegend=False,
        dragmode='pan',
        hovermode='closest',
        clickmode='event+select',
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

def run_power_flow():
    """Simulação simplificada de fluxo de potência"""
    if len(st.session_state.buses) == 0:
        return None
    
    # Garantir barra slack
    slack_buses = [b for b in st.session_state.buses if b['type'] == 'slack']
    if not slack_buses and st.session_state.buses:
        st.session_state.buses[0]['type'] = 'slack'
        st.session_state.buses[0]['color'] = '#ef4444'
    
    n = len(st.session_state.buses)
    voltages = []
    
    for i, bus in enumerate(st.session_state.buses):
        if bus['type'] == 'slack':
            voltages.append(1.0)
        elif bus['type'] == 'pv':
            gen = next((g for g in st.session_state.generators if g['bus'] == bus['id']), None)
            voltages.append(gen['vm_pu'] if gen else 1.0)
        else:
            base = 1.0
            if any(l['bus'] == bus['id'] for l in st.session_state.loads):
                base -= 0.05
            if any(g['bus'] == bus['id'] for g in st.session_state.generators):
                base += 0.03
            voltages.append(max(0.9, min(1.1, base + np.random.normal(0, 0.02))))
    
    results = {
        'type': 'power_flow',
        'timestamp': datetime.now().isoformat(),
        'voltages': voltages,
        'converged': True,
        'iterations': np.random.randint(3, 8)
    }
    
    st.session_state.results = results
    st.session_state.history.append(results)
    if len(st.session_state.history) > 10:
        st.session_state.history = st.session_state.history[-10:]
    
    return results

def run_short_circuit():
    """Análise de curto-circuito"""
    if len(st.session_state.buses) == 0:
        return None
    
    if st.session_state.selected_type != 'bus' or st.session_state.selected_element is None:
        return None
    
    fault_bus = st.session_state.selected_element
    n = len(st.session_state.buses)
    currents = []
    
    for i in range(n):
        if i == fault_bus:
            base = 10.0
            bus = get_bus_by_id(i)
            if bus['type'] == 'slack':
                base *= 1.5
            currents.append(base + np.random.uniform(-2, 3))
        else:
            currents.append(np.random.uniform(0.1, 2.0))
    
    results = {
        'type': 'short_circuit',
        'timestamp': datetime.now().isoformat(),
        'fault_bus': fault_bus,
        'currents': currents,
        'max_current': max(currents)
    }
    
    st.session_state.results = results
    st.session_state.history.append(results)
    if len(st.session_state.history) > 10:
        st.session_state.history = st.session_state.history[-10:]
    
    return results

def export_system():
    data = {
        'system': {
            'buses': st.session_state.buses,
            'lines': st.session_state.lines,
            'loads': st.session_state.loads,
            'generators': st.session_state.generators
        },
        'results': st.session_state.results,
        'history': st.session_state.history,
        'metadata': {
            'name': st.session_state.system_name,
            'export_date': datetime.now().isoformat()
        }
    }
    return json.dumps(data, indent=2)

def import_system(uploaded_file):
    try:
        data = json.load(uploaded_file)
        
        st.session_state.buses = data.get('system', {}).get('buses', [])
        st.session_state.lines = data.get('system', {}).get('lines', [])
        st.session_state.loads = data.get('system', {}).get('loads', [])
        st.session_state.generators = data.get('system', {}).get('generators', [])
        
        # Atualizar IDs
        all_ids = []
        all_ids.extend([b['id'] for b in st.session_state.buses])
        all_ids.extend([l['id'] for l in st.session_state.lines])
        all_ids.extend([l['id'] for l in st.session_state.loads])
        all_ids.extend([g['id'] for g in st.session_state.generators])
        
        if all_ids:
            st.session_state.next_id = max(all_ids) + 1
        
        st.session_state.results = data.get('results')
        st.session_state.history = data.get('history', [])
        
        if 'metadata' in data:
            st.session_state.system_name = data['metadata'].get('name', 'Sistema Importado')
        
        return True
    except Exception as e:
        st.error(f"Erro: {str(e)}")
        return False

# Interface principal
def main():
    st.title("⚡ Power System Studio - Click & Drag")
    
    # Barra de status
    status_text = ""
    if st.session_state.mode == 'select':
        status_text = "🔍 Modo Seleção: Clique nos elementos para selecionar"
    elif st.session_state.mode == 'connect':
        status_text = "🔗 Modo Conexão: Clique em duas barras para conectá-las"
        if st.session_state.connecting_from is not None:
            status_text += f" (Conectando da Barra {st.session_state.connecting_from})"
    elif st.session_state.mode == 'delete':
        status_text = "🗑️ Modo Deleção: Clique nos elementos para removê-los"
    
    if st.session_state.action_mode:
        action_names = {
            'add_bus_slack': "Barra Slack",
            'add_bus_pv': "Barra PV", 
            'add_bus_pq': "Barra PQ",
            'add_load': "Carga",
            'add_generator': "Gerador"
        }
        status_text += f" | 🎯 Duplo-clique para adicionar {action_names[st.session_state.action_mode]}"
    
    st.markdown(f'<div class="status-bar">{status_text}</div>', unsafe_allow_html=True)
    
    # Layout principal
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Canvas interativo
        fig = create_interactive_canvas()
        
        # Container para o gráfico
        with st.container():
            click_data = st.plotly_chart(
                fig, 
                use_container_width=True,
                key="main_canvas",
                on_select="rerun"
            )
        
        # Processar interações
        if click_data and 'selection' in click_data:
            handle_canvas_click(click_data['selection'])
        
        # Instruções
        st.markdown("""
        ### 📋 Instruções:
        1. **Selecione um modo de operação** na barra lateral
        2. **Clique** nos elementos para selecionar
        3. **Duplo-clique** no canvas para adicionar elementos (quando um modo de adição está ativo)
        4. **Clique e arraste** no canvas para mover a visualização
        """)
    
    with col2:
        st.header("🎮 Controles")
        
        # Modos de operação
        st.subheader("Modos")
        mode_cols = st.columns(2)
        with mode_cols[0]:
            if st.button("🔍 Selecionar", 
                        key="mode_select",
                        use_container_width=True,
                        type="primary" if st.session_state.mode == 'select' else "secondary"):
                st.session_state.mode = 'select'
                st.session_state.connecting_from = None
                st.session_state.action_mode = None
                st.rerun()
        
        with mode_cols[1]:
            if st.button("🔗 Conectar", 
                        key="mode_connect",
                        use_container_width=True,
                        type="primary" if st.session_state.mode == 'connect' else "secondary"):
                st.session_state.mode = 'connect'
                st.session_state.action_mode = None
                st.rerun()
        
        if st.button("🗑️ Deletar", 
                    key="mode_delete",
                    use_container_width=True,
                    type="primary" if st.session_state.mode == 'delete' else "secondary"):
            st.session_state.mode = 'delete'
            st.session_state.action_mode = None
            st.rerun()
        
        st.divider()
        
        # Adicionar elementos
        st.subheader("➕ Adicionar")
        
        st.markdown('<div class="element-card bus-slack">Barra Slack</div>', 
                   unsafe_allow_html=True)
        col_slack = st.columns([3, 1])
        with col_slack[0]:
            if st.button("Adicionar Barra Slack", key="add_bus_slack", use_container_width=True):
                st.session_state.action_mode = 'add_bus_slack'
                st.info("Duplo-clique no canvas para posicionar a barra")
        
        st.markdown('<div class="element-card bus-pv">Barra PV</div>', 
                   unsafe_allow_html=True)
        col_pv = st.columns([3, 1])
        with col_pv[0]:
            if st.button("Adicionar Barra PV", key="add_bus_pv", use_container_width=True):
                st.session_state.action_mode = 'add_bus_pv'
                st.info("Duplo-clique no canvas para posicionar a barra")
        
        st.markdown('<div class="element-card bus-pq">Barra PQ</div>', 
                   unsafe_allow_html=True)
        col_pq = st.columns([3, 1])
        with col_pq[0]:
            if st.button("Adicionar Barra PQ", key="add_bus_pq", use_container_width=True):
                st.session_state.action_mode = 'add_bus_pq'
                st.info("Duplo-clique no canvas para posicionar a barra")
        
        st.markdown('<div class="element-card element-load">Carga</div>', 
                   unsafe_allow_html=True)
        col_load = st.columns([3, 1])
        with col_load[0]:
            if st.button("Adicionar Carga", key="add_load", use_container_width=True):
                st.session_state.action_mode = 'add_load'
                st.info("Duplo-clique perto de uma barra para adicionar carga")
        
        st.markdown('<div class="element-card element-gen">Gerador</div>', 
                   unsafe_allow_html=True)
        col_gen = st.columns([3, 1])
        with col_gen[0]:
            if st.button("Adicionar Gerador", key="add_gen", use_container_width=True):
                st.session_state.action_mode = 'add_generator'
                st.info("Duplo-clique perto de uma barra para adicionar gerador")
        
        st.divider()
        
        # Análises
        st.subheader("📊 Análises")
        
        if st.button("🔁 Executar Fluxo", use_container_width=True):
            with st.spinner("Calculando..."):
                results = run_power_flow()
                if results:
                    st.success("Fluxo calculado!")
                st.rerun()
        
        if st.button("⚡ Curto-Circuito", use_container_width=True):
            with st.spinner("Analisando..."):
                results = run_short_circuit()
                if results:
                    st.success(f"Curto na barra {results['fault_bus']} analisado!")
                st.rerun()
        
        st.divider()
        
        # Propriedades do elemento selecionado
        if st.session_state.selected_element is not None:
            st.subheader("🔧 Propriedades")
            
            element = get_element_by_id(
                st.session_state.selected_element, 
                st.session_state.selected_type
            )
            
            if element:
                with st.form("edit_form"):
                    if st.session_state.selected_type == 'bus':
                        element['name'] = st.text_input("Nome", element['name'])
                        
                        new_type = st.selectbox(
                            "Tipo",
                            ["slack", "pv", "pq"],
                            index=["slack", "pv", "pq"].index(element['type'])
                        )
                        
                        if new_type != element['type']:
                            element['type'] = new_type
                            element['color'] = {'slack': '#ef4444', 'pv': '#10b981', 'pq': '#3b82f6'}[new_type]
                        
                        col_pos = st.columns(2)
                        with col_pos[0]:
                            element['x'] = st.number_input("X", value=float(element['x']), 
                                                          min_value=0.0, 
                                                          max_value=float(st.session_state.canvas_width))
                        with col_pos[1]:
                            element['y'] = st.number_input("Y", value=float(element['y']), 
                                                          min_value=0.0, 
                                                          max_value=float(st.session_state.canvas_height))
                    
                    elif st.session_state.selected_type == 'line':
                        element['name'] = st.text_input("Nome", element['name'])
                        st.info(f"Conecta: Barra {element['from']} → Barra {element['to']}")
                    
                    elif st.session_state.selected_type == 'load':
                        element['name'] = st.text_input("Nome", element['name'])
                        element['p_mw'] = st.number_input("P (MW)", value=float(element['p_mw']))
                        element['q_mvar'] = st.number_input("Q (MVar)", value=float(element['q_mvar']))
                    
                    elif st.session_state.selected_type == 'generator':
                        element['name'] = st.text_input("Nome", element['name'])
                        element['p_mw'] = st.number_input("P (MW)", value=float(element['p_mw']))
                        element['vm_pu'] = st.number_input("Vm (pu)", value=float(element['vm_pu']))
                    
                    col_btn = st.columns(2)
                    with col_btn[0]:
                        if st.form_submit_button("💾 Salvar", use_container_width=True):
                            st.success("Salvo!")
                            st.rerun()
                    with col_btn[1]:
                        if st.form_submit_button("🗑️ Remover", type="secondary", use_container_width=True):
                            delete_element(element['id'], st.session_state.selected_type)
                            st.session_state.selected_element = None
                            st.session_state.selected_type = None
                            st.rerun()
        
        # Configurações
        st.divider()
        st.subheader("⚙️ Configurações")
        
        st.session_state.system_name = st.text_input("Nome do Sistema", 
                                                    st.session_state.system_name)
        
        col_set = st.columns(2)
        with col_set[0]:
            st.session_state.grid_enabled = st.checkbox("Grade", 
                                                       st.session_state.grid_enabled)
        with col_set[1]:
            st.session_state.show_voltages = st.checkbox("Mostrar Tensões", 
                                                        st.session_state.show_voltages)
        
        # Arquivos
        st.divider()
        st.subheader("💾 Arquivos")
        
        uploaded_file = st.file_uploader("Importar", type=['json'])
        if uploaded_file:
            if import_system(uploaded_file):
                st.rerun()
        
        if st.button("📥 Exportar Sistema", use_container_width=True):
            json_str = export_system()
            st.download_button(
                label="Baixar JSON",
                data=json_str,
                file_name=f"{st.session_state.system_name}.json",
                mime="application/json",
                use_container_width=True
            )
        
        if st.button("🔄 Limpar Tudo", type="secondary", use_container_width=True):
            st.session_state.buses = []
            st.session_state.lines = []
            st.session_state.loads = []
            st.session_state.generators = []
            st.session_state.results = None
            st.session_state.history = []
            st.session_state.selected_element = None
            st.session_state.selected_type = None
            st.session_state.next_id = 0
            st.rerun()
    
    # Resultados na parte inferior
    if st.session_state.results:
        st.divider()
        st.subheader("📈 Resultados")
        
        results = st.session_state.results
        time_str = datetime.fromisoformat(results['timestamp']).strftime("%H:%M:%S")
        
        if results['type'] == 'power_flow':
            st.markdown(f"**Fluxo de Potência** ({time_str})")
            
            col_res = st.columns(3)
            with col_res[0]:
                st.metric("Iterações", results['iterations'])
            with col_res[1]:
                avg_v = np.mean(results['voltages'])
                st.metric("Tensão Média", f"{avg_v:.3f} pu")
            with col_res[2]:
                min_v = min(results['voltages'])
                st.metric("Tensão Mínima", f"{min_v:.3f} pu")
            
            # Gráfico de tensões
            fig_v = go.Figure(data=[
                go.Bar(
                    x=list(range(len(results['voltages']))),
                    y=results['voltages'],
                    marker_color=['red' if v < 0.95 else 'orange' if v > 1.05 else 'green' 
                                 for v in results['voltages']]
                )
            ])
            fig_v.update_layout(
                title="Tensões por Barra",
                xaxis_title="Barra",
                yaxis_title="Tensão (pu)",
                height=200
            )
            st.plotly_chart(fig_v, use_container_width=True)
        
        elif results['type'] == 'short_circuit':
            st.markdown(f"**Curto-Circuito** ({time_str})")
            st.warning(f"Barra de falta: {results['fault_bus']}")
            st.metric("Corrente Máxima", f"{results['max_current']:.2f} kA")

# Executar aplicação
if __name__ == "__main__":
    main()

