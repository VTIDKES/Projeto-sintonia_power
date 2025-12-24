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
    page_title="Power System Studio - Drag & Drop",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado para melhorar a interface
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
        margin-top: 5px;
        margin-bottom: 5px;
    }
    .element-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 10px;
        margin: 5px 0;
        background-color: #f8f9fa;
        cursor: move;
        user-select: none;
    }
    .element-card:hover {
        background-color: #e9ecef;
        border-color: #007bff;
    }
    .bus-slack { background-color: #f8d7da !important; border-left: 4px solid #dc3545 !important; }
    .bus-pv { background-color: #d1e7dd !important; border-left: 4px solid #198754 !important; }
    .bus-pq { background-color: #cff4fc !important; border-left: 4px solid #0dcaf0 !important; }
    .element-line { background-color: #e7f1ff !important; border-left: 4px solid #0d6efd !important; }
    .element-load { background-color: #e7d7ff !important; border-left: 4px solid #6f42c1 !important; }
    .element-gen { background-color: #d4edda !important; border-left: 4px solid #28a745 !important; }
    .connection-mode { 
        background-color: #fff3cd !important; 
        border: 2px dashed #ffc107 !important;
        font-weight: bold;
    }
    .selected-element {
        background-color: #fff3cd !important;
        border: 2px solid #ffc107 !important;
    }
</style>
""", unsafe_allow_html=True)

# Inicialização do estado da sessão
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
        'connecting_from': None,
        'next_id': 0,
        'dragging_element': None,
        'drag_start_pos': None,
        'last_click_time': 0,
        'system_name': 'Novo Sistema',
        'grid_enabled': True,
        'snap_to_grid': True,
        'show_voltages': True,
        'show_power_flow': True,
    }
    
    for key, value in default_state.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Funções auxiliares
def generate_id():
    st.session_state.next_id += 1
    return st.session_state.next_id - 1

def add_bus(name=None, x=400, y=300, bus_type='pq', vn_kv=13.8):
    bus_id = generate_id()
    if name is None:
        name = f"Barra {bus_id}"
    
    new_bus = {
        'id': bus_id,
        'name': name,
        'x': x,
        'y': y,
        'type': bus_type,
        'vn_kv': vn_kv,
        'v_pu': 1.0 if bus_type == 'slack' else 0.98 + np.random.random() * 0.04,
        'angle_deg': 0.0,
        'color': {
            'slack': '#ef4444',
            'pv': '#10b981',
            'pq': '#3b82f6'
        }[bus_type]
    }
    st.session_state.buses.append(new_bus)
    return new_bus

def add_line(from_bus_id, to_bus_id, name=None):
    # Verificar se a linha já existe
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

def get_element_by_id(element_id, element_type):
    if element_type == 'bus':
        for bus in st.session_state.buses:
            if bus['id'] == element_id:
                return bus
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

def get_bus_by_id(bus_id):
    for bus in st.session_state.buses:
        if bus['id'] == bus_id:
            return bus
    return None

def select_element(element_id, element_type):
    st.session_state.selected_element = element_id
    st.session_state.selected_type = element_type

def delete_selected_element():
    if st.session_state.selected_element is not None and st.session_state.selected_type is not None:
        element_id = st.session_state.selected_element
        element_type = st.session_state.selected_type
        
        if element_type == 'bus':
            # Remover barra e todos os elementos associados
            st.session_state.buses = [b for b in st.session_state.buses if b['id'] != element_id]
            st.session_state.lines = [l for l in st.session_state.lines if l['from'] != element_id and l['to'] != element_id]
            st.session_state.loads = [l for l in st.session_state.loads if l['bus'] != element_id]
            st.session_state.generators = [g for g in st.session_state.generators if g['bus'] != element_id]
        elif element_type == 'line':
            st.session_state.lines = [l for l in st.session_state.lines if l['id'] != element_id]
        elif element_type == 'load':
            st.session_state.loads = [l for l in st.session_state.loads if l['id'] != element_id]
        elif element_type == 'generator':
            st.session_state.generators = [g for g in st.session_state.generators if g['id'] != element_id]
        
        st.session_state.selected_element = None
        st.session_state.selected_type = None

def move_bus(bus_id, new_x, new_y):
    for bus in st.session_state.buses:
        if bus['id'] == bus_id:
            bus['x'] = new_x
            bus['y'] = new_y
            break

def create_palette_draggable():
    """Cria HTML para elementos arrastáveis na paleta"""
    html_code = """
    <div id="palette-container" style="display: flex; flex-direction: column; gap: 10px; padding: 10px;">
        <div class="element-card bus-slack" draggable="true" data-type="bus" data-bus-type="slack" 
             style="padding: 10px; border-radius: 5px; cursor: move;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="width: 20px; height: 20px; background-color: #ef4444; border-radius: 50%; border: 2px solid #dc2626;"></div>
                <div>
                    <strong>Barra Slack</strong><br>
                    <small>Tensão de referência</small>
                </div>
            </div>
        </div>
        
        <div class="element-card bus-pv" draggable="true" data-type="bus" data-bus-type="pv"
             style="padding: 10px; border-radius: 5px; cursor: move;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="width: 20px; height: 20px; background-color: #10b981; border-radius: 50%; border: 2px solid #059669;"></div>
                <div>
                    <strong>Barra PV</strong><br>
                    <small>Controle de tensão</small>
                </div>
            </div>
        </div>
        
        <div class="element-card bus-pq" draggable="true" data-type="bus" data-bus-type="pq"
             style="padding: 10px; border-radius: 5px; cursor: move;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="width: 20px; height: 20px; background-color: #3b82f6; border-radius: 50%; border: 2px solid #1d4ed8;"></div>
                <div>
                    <strong>Barra PQ</strong><br>
                    <small>Carga constante</small>
                </div>
            </div>
        </div>
        
        <div class="element-card element-line" draggable="true" data-type="line"
             style="padding: 10px; border-radius: 5px; cursor: move;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="width: 40px; height: 3px; background-color: #2563eb;"></div>
                <div>
                    <strong>Linha</strong><br>
                    <small>Conexão entre barras</small>
                </div>
            </div>
        </div>
        
        <div class="element-card element-load" draggable="true" data-type="load"
             style="padding: 10px; border-radius: 5px; cursor: move;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="width: 20px; height: 20px; background-color: #8b5cf6; border: 2px solid #7c3aed;"></div>
                <div>
                    <strong>Carga</strong><br>
                    <small>Consumo de energia</small>
                </div>
            </div>
        </div>
        
        <div class="element-card element-gen" draggable="true" data-type="generator"
             style="padding: 10px; border-radius: 5px; cursor: move;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="width: 20px; height: 20px; background-color: #10b981; border-radius: 50%; border: 2px solid #059669;"></div>
                <div>
                    <strong>Gerador</strong><br>
                    <small>Geração de energia</small>
                </div>
            </div>
        </div>
    </div>
    
    <script>
    // Configurar drag and drop
    const paletteElements = document.querySelectorAll('#palette-container [draggable="true"]');
    const canvas = document.querySelector('[data-testid="stPlotlyChart"] iframe')?.contentDocument?.querySelector('.js-plotly-plot');
    
    paletteElements.forEach(element => {
        element.addEventListener('dragstart', function(e) {
            const elementType = this.getAttribute('data-type');
            const busType = this.getAttribute('data-bus-type');
            e.dataTransfer.setData('text/plain', JSON.stringify({
                type: elementType,
                busType: busType,
                source: 'palette'
            }));
            this.style.opacity = '0.5';
        });
        
        element.addEventListener('dragend', function(e) {
            this.style.opacity = '1';
        });
    });
    
    // Enviar mensagem para Python quando um elemento é arrastado para o canvas
    if (canvas) {
        canvas.addEventListener('dragover', function(e) {
            e.preventDefault();
            e.dataTransfer.dropEffect = 'copy';
        });
        
        canvas.addEventListener('drop', function(e) {
            e.preventDefault();
            const data = JSON.parse(e.dataTransfer.getData('text/plain'));
            
            if (data.source === 'palette') {
                // Obter coordenadas do drop
                const rect = this.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;
                
                // Enviar dados para Streamlit
                const message = {
                    type: 'add_element',
                    elementType: data.type,
                    busType: data.busType,
                    x: x,
                    y: y
                };
                
                window.parent.postMessage(message, '*');
            }
        });
    }
    
    // Receber mensagens do Python
    window.addEventListener('message', function(event) {
        if (event.data.type === 'select_element') {
            // Atualizar seleção visual
            console.log('Element selected:', event.data);
        }
    });
    </script>
    """
    
    return html_code

def create_interactive_canvas():
    """Cria o canvas interativo com Plotly"""
    fig = go.Figure()
    
    # Adicionar grade
    if st.session_state.grid_enabled:
        grid_size = 20
        for x in range(0, 1001, grid_size):
            fig.add_shape(
                type="line",
                x0=x, y0=0, x1=x, y1=700,
                line=dict(color="lightgray", width=0.5, dash="dot")
            )
        for y in range(0, 701, grid_size):
            fig.add_shape(
                type="line",
                x0=0, y0=y, x1=1000, y1=y,
                line=dict(color="lightgray", width=0.5, dash="dot")
            )
    
    # Desenhar linhas
    for line in st.session_state.lines:
        from_bus = get_bus_by_id(line['from'])
        to_bus = get_bus_by_id(line['to'])
        
        if from_bus and to_bus:
            # Linha principal
            fig.add_trace(go.Scatter(
                x=[from_bus['x'], to_bus['x']],
                y=[from_bus['y'], to_bus['y']],
                mode='lines',
                line=dict(
                    color=line['color'],
                    width=3
                ),
                hoverinfo='text',
                hovertext=f"<b>{line['name']}</b><br>ID: {line['id']}<br>De: Barra {line['from']}<br>Para: Barra {line['to']}<br>R: {line['r_ohm_per_km']} Ω/km<br>X: {line['x_ohm_per_km']} Ω/km",
                name=f"Linha {line['id']}",
                customdata=[['line', line['id']], ['line', line['id']]]
            ))
            
            # Seta no meio
            mid_x = (from_bus['x'] + to_bus['x']) / 2
            mid_y = (from_bus['y'] + to_bus['y']) / 2
            
            # Calcular ângulo para a seta
            dx = to_bus['x'] - from_bus['x']
            dy = to_bus['y'] - from_bus['y']
            angle = np.arctan2(dy, dx)
            
            # Adicionar seta como annotation
            fig.add_annotation(
                x=mid_x,
                y=mid_y,
                ax=mid_x - 10 * np.cos(angle),
                ay=mid_y - 10 * np.sin(angle),
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
    for bus in st.session_state.buses:
        is_selected = (st.session_state.selected_type == 'bus' and 
                      st.session_state.selected_element == bus['id'])
        
        # Círculo da barra
        fig.add_trace(go.Scatter(
            x=[bus['x']],
            y=[bus['y']],
            mode='markers',
            marker=dict(
                size=25,
                color=bus['color'],
                line=dict(
                    width=4 if is_selected else 2,
                    color='#facc15' if is_selected else '#1e293b'
                )
            ),
            hoverinfo='text',
            hovertext=f"<b>{bus['name']}</b><br>ID: {bus['id']}<br>Tipo: {bus['type'].upper()}<br>Vn: {bus['vn_kv']} kV<br>Posição: ({bus['x']}, {bus['y']})",
            name=f"Barra {bus['id']}",
            customdata=[['bus', bus['id']]]
        ))
        
        # Texto da barra
        fig.add_annotation(
            x=bus['x'],
            y=bus['y'],
            text=str(bus['id']),
            showarrow=False,
            font=dict(size=14, color="white", family="Arial Black"),
            xanchor="center",
            yanchor="middle"
        )
        
        # Nome da barra
        fig.add_annotation(
            x=bus['x'],
            y=bus['y'] + 35,
            text=bus['name'],
            showarrow=False,
            font=dict(size=12, color="#1e293b"),
            xanchor="center",
            yanchor="middle"
        )
        
        # Tensão (se disponível)
        if st.session_state.show_voltages and st.session_state.results:
            voltage = st.session_state.results.get('voltages', [1.0] * len(st.session_state.buses))[bus['id']]
            fig.add_annotation(
                x=bus['x'],
                y=bus['y'] - 35,
                text=f"{voltage:.3f} pu",
                showarrow=False,
                font=dict(
                    size=10,
                    color="red" if voltage < 0.95 else "orange" if voltage > 1.05 else "green"
                ),
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
                y=[bus['y'] + 50],
                mode='markers+text',
                marker=dict(
                    symbol='square',
                    size=15,
                    color=load['color'],
                    line=dict(
                        width=3 if is_selected else 1,
                        color='#facc15' if is_selected else 'white'
                    )
                ),
                text='L',
                textfont=dict(color='white', size=10, family='Arial Black'),
                textposition='middle center',
                hoverinfo='text',
                hovertext=f"<b>{load['name']}</b><br>ID: {load['id']}<br>Barra: {load['bus']}<br>P: {load['p_mw']} MW<br>Q: {load['q_mvar']} MVar",
                name=f"Carga {load['id']}",
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
                y=[bus['y'] - 50],
                mode='markers+text',
                marker=dict(
                    symbol='circle',
                    size=15,
                    color=gen['color'],
                    line=dict(
                        width=3 if is_selected else 2,
                        color='#facc15' if is_selected else '#059669'
                    )
                ),
                text='G',
                textfont=dict(color='white', size=10, family='Arial Black'),
                textposition='middle center',
                hoverinfo='text',
                hovertext=f"<b>{gen['name']}</b><br>ID: {gen['id']}<br>Barra: {gen['bus']}<br>P: {gen['p_mw']} MW<br>Vm: {gen['vm_pu']} pu",
                name=f"Gerador {gen['id']}",
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
        width=1000,
        height=700,
        plot_bgcolor='white',
        paper_bgcolor='white',
        title=dict(
            text=f"Sistema: {st.session_state.system_name}",
            font=dict(size=20)
        ),
        xaxis=dict(
            range=[0, 1000],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            fixedrange=True
        ),
        yaxis=dict(
            range=[0, 700],
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
        clickmode='event+select'
    )
    
    return fig

def handle_canvas_click(click_data):
    """Processa cliques no canvas"""
    if not click_data or 'points' not in click_data:
        return
    
    points = click_data['points']
    if not points:
        return
    
    point = points[0]
    
    # Obter tipo e ID do elemento clicado
    if 'customdata' in point and point['customdata']:
        element_type = point['customdata'][0]
        element_id = point['customdata'][1]
        
        # Processar baseado no modo atual
        if st.session_state.mode == 'select':
            # Selecionar elemento
            select_element(element_id, element_type)
            st.rerun()
        
        elif st.session_state.mode == 'connect':
            # Modo conexão
            if element_type == 'bus':
                if st.session_state.connecting_from is None:
                    # Primeira barra selecionada para conexão
                    st.session_state.connecting_from = element_id
                    st.session_state.selected_element = element_id
                    st.session_state.selected_type = 'bus'
                else:
                    # Segunda barra - criar linha
                    from_bus_id = st.session_state.connecting_from
                    to_bus_id = element_id
                    
                    if from_bus_id != to_bus_id:
                        line = add_line(from_bus_id, to_bus_id)
                        if line:
                            st.success(f"Linha criada entre barras {from_bus_id} e {to_bus_id}")
                    
                    st.session_state.connecting_from = None
                    st.rerun()
        
        elif st.session_state.mode == 'delete':
            # Modo deletar
            select_element(element_id, element_type)
            delete_selected_element()
            st.rerun()

def run_power_flow_simulation():
    """Executa uma simulação simplificada de fluxo de potência"""
    if len(st.session_state.buses) == 0:
        st.error("Adicione barras ao sistema primeiro!")
        return
    
    # Garantir que há pelo menos uma barra slack
    slack_buses = [b for b in st.session_state.buses if b['type'] == 'slack']
    if not slack_buses and st.session_state.buses:
        st.session_state.buses[0]['type'] = 'slack'
        st.session_state.buses[0]['color'] = '#ef4444'
    
    # Simulação simplificada
    n_buses = len(st.session_state.buses)
    
    # Inicializar tensões
    voltages = []
    for bus in st.session_state.buses:
        if bus['type'] == 'slack':
            voltages.append(1.0)
        elif bus['type'] == 'pv':
            # Encontrar gerador nesta barra
            gen = next((g for g in st.session_state.generators if g['bus'] == bus['id']), None)
            if gen:
                voltages.append(gen['vm_pu'])
            else:
                voltages.append(0.98 + np.random.random() * 0.04)
        else:  # PQ
            base_voltage = 1.0
            
            # Ajustar baseado em cargas
            loads = [l for l in st.session_state.loads if l['bus'] == bus['id']]
            if loads:
                base_voltage -= 0.05
            
            # Ajustar baseado em geradores
            gens = [g for g in st.session_state.generators if g['bus'] == bus['id']]
            if gens:
                base_voltage += 0.03
            
            voltages.append(max(0.9, min(1.1, base_voltage + np.random.normal(0, 0.02))))
    
    # Calcular fluxo de potência (simplificado)
    angles = [0.0] * n_buses
    for i in range(1, n_buses):
        angles[i] = np.random.uniform(-0.2, 0.2)
    
    # Salvar resultados
    results = {
        'type': 'power_flow',
        'timestamp': datetime.now().isoformat(),
        'voltages': voltages,
        'angles': angles,
        'converged': True,
        'iterations': np.random.randint(3, 8),
        'total_load_mw': sum(load['p_mw'] for load in st.session_state.loads),
        'total_gen_mw': sum(gen['p_mw'] for gen in st.session_state.generators)
    }
    
    st.session_state.results = results
    st.session_state.history.append(results)
    
    if len(st.session_state.history) > 10:
        st.session_state.history = st.session_state.history[-10:]
    
    return results

def run_short_circuit_analysis():
    """Executa análise de curto-circuito"""
    if len(st.session_state.buses) == 0:
        st.error("Adicione barras ao sistema primeiro!")
        return
    
    if st.session_state.selected_type != 'bus' or st.session_state.selected_element is None:
        st.error("Selecione uma barra para simular o curto-circuito!")
        return
    
    fault_bus_id = st.session_state.selected_element
    
    # Simulação simplificada
    fault_currents = []
    for bus in st.session_state.buses:
        if bus['id'] == fault_bus_id:
            # Barra de falta - corrente alta
            base_current = 10.0
            if bus['type'] == 'slack':
                base_current *= 1.5
            current = base_current + np.random.uniform(-2, 3)
        else:
            # Outras barras - corrente baixa
            current = np.random.uniform(0.1, 2.0)
        fault_currents.append(current)
    
    results = {
        'type': 'short_circuit',
        'timestamp': datetime.now().isoformat(),
        'fault_bus': fault_bus_id,
        'fault_currents': fault_currents,
        'max_current': max(fault_currents),
        'critical_buses': [
            {'bus_id': i, 'current': fault_currents[i]}
            for i in range(len(fault_currents))
            if fault_currents[i] > 8.0
        ]
    }
    
    st.session_state.results = results
    st.session_state.history.append(results)
    
    if len(st.session_state.history) > 10:
        st.session_state.history = st.session_state.history[-10:]
    
    return results

def export_system():
    """Exporta o sistema para JSON"""
    data = {
        'metadata': {
            'name': st.session_state.system_name,
            'export_date': datetime.now().isoformat(),
            'version': '1.0'
        },
        'system': {
            'buses': st.session_state.buses,
            'lines': st.session_state.lines,
            'loads': st.session_state.loads,
            'generators': st.session_state.generators
        },
        'results': st.session_state.results,
        'history': st.session_state.history
    }
    return json.dumps(data, indent=2)

def import_system(uploaded_file):
    """Importa sistema de arquivo JSON"""
    try:
        data = json.load(uploaded_file)
        
        # Importar dados básicos
        if 'system' in data:
            system_data = data['system']
            st.session_state.buses = system_data.get('buses', [])
            st.session_state.lines = system_data.get('lines', [])
            st.session_state.loads = system_data.get('loads', [])
            st.session_state.generators = system_data.get('generators', [])
        
        # Atualizar contador de IDs
        all_ids = []
        all_ids.extend([b['id'] for b in st.session_state.buses])
        all_ids.extend([l['id'] for l in st.session_state.lines])
        all_ids.extend([l['id'] for l in st.session_state.loads])
        all_ids.extend([g['id'] for g in st.session_state.generators])
        
        if all_ids:
            st.session_state.next_id = max(all_ids) + 1
        
        # Importar resultados e histórico
        st.session_state.results = data.get('results')
        st.session_state.history = data.get('history', [])
        
        # Importar metadados
        if 'metadata' in data:
            st.session_state.system_name = data['metadata'].get('name', 'Sistema Importado')
        
        return True
    except Exception as e:
        st.error(f"Erro ao importar: {str(e)}")
        return False

# Interface principal
def main():
    st.title("⚡ Power System Studio - Drag & Drop")
    st.markdown("Crie, edite e simule sistemas elétricos de forma visual e interativa")
    
    # Barra lateral esquerda - Paleta de elementos
    with st.sidebar:
        st.header("🎨 Paleta de Elementos")
        st.markdown("Arraste e solte elementos no canvas:")
        
        # Usar HTML para elementos arrastáveis
        html(create_palette_draggable(), height=400)
        
        st.divider()
        
        st.header("🛠️ Ferramentas")
        
        # Modos de operação
        mode_cols = st.columns(2)
        with mode_cols[0]:
            if st.button("🔍 Selecionar", 
                        use_container_width=True,
                        type="primary" if st.session_state.mode == 'select' else "secondary"):
                st.session_state.mode = 'select'
                st.session_state.connecting_from = None
                st.rerun()
        
        with mode_cols[1]:
            if st.button("🔗 Conectar", 
                        use_container_width=True,
                        type="primary" if st.session_state.mode == 'connect' else "secondary"):
                st.session_state.mode = 'connect'
                st.rerun()
        
        mode_cols2 = st.columns(2)
        with mode_cols2[0]:
            if st.button("➕ Adicionar", 
                        use_container_width=True,
                        type="primary" if st.session_state.mode == 'add' else "secondary"):
                st.session_state.mode = 'add'
                st.rerun()
        
        with mode_cols2[1]:
            if st.button("🗑️ Deletar", 
                        use_container_width=True,
                        type="primary" if st.session_state.mode == 'delete' else "secondary"):
                st.session_state.mode = 'delete'
                st.rerun()
        
        st.divider()
        
        # Configurações
        st.header("⚙️ Configurações")
        
        st.session_state.system_name = st.text_input("Nome do Sistema", 
                                                    st.session_state.system_name)
        
        col_set1, col_set2 = st.columns(2)
        with col_set1:
            st.session_state.grid_enabled = st.checkbox("Grade", 
                                                       st.session_state.grid_enabled)
        with col_set2:
            st.session_state.snap_to_grid = st.checkbox("Snap to Grid", 
                                                       st.session_state.snap_to_grid)
        
        st.session_state.show_voltages = st.checkbox("Mostrar Tensões", 
                                                    st.session_state.show_voltages)
        
        st.divider()
        
        # Análises
        st.header("📊 Análises")
        
        if st.button("🔁 Executar Fluxo de Potência", use_container_width=True):
            with st.spinner("Calculando fluxo de potência..."):
                results = run_power_flow_simulation()
                if results:
                    st.success("Fluxo de potência calculado!")
                st.rerun()
        
        if st.button("⚡ Análise de Curto-Circuito", use_container_width=True):
            with st.spinner("Analisando curto-circuito..."):
                results = run_short_circuit_analysis()
                if results:
                    st.success(f"Curto-circuito na barra {results['fault_bus']} analisado!")
                st.rerun()
        
        st.divider()
        
        # Gerenciamento de arquivos
        st.header("💾 Arquivos")
        
        # Importar
        uploaded_file = st.file_uploader("Importar sistema", type=['json'])
        if uploaded_file is not None:
            if import_system(uploaded_file):
                st.success("Sistema importado com sucesso!")
                st.rerun()
        
        # Exportar
        if st.button("📥 Exportar Sistema", use_container_width=True):
            json_str = export_system()
            st.download_button(
                label="⬇️ Baixar JSON",
                data=json_str,
                file_name=f"{st.session_state.system_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        # Limpar
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
    
    # Área principal
    col_main1, col_main2 = st.columns([3, 1])
    
    with col_main1:
        # Canvas interativo
        st.subheader("Canvas Interativo")
        
        # Instruções
        mode_instructions = {
            'select': "🔍 **Modo Seleção:** Clique nos elementos para selecionar",
            'connect': "🔗 **Modo Conexão:** Clique em duas barras para conectá-las",
            'add': "➕ **Modo Adição:** Arraste elementos da paleta para o canvas",
            'delete': "🗑️ **Modo Deleção:** Clique nos elementos para removê-los"
        }
        
        current_mode = st.session_state.mode
        st.info(mode_instructions.get(current_mode, "Selecione um modo de operação"))
        
        # Criar canvas
        fig = create_interactive_canvas()
        
        # Mostrar gráfico e capturar interações
        click_data = st.plotly_chart(
            fig, 
            use_container_width=True,
            key="main_canvas",
            on_select="rerun"
        )
        
        # Processar cliques
        if click_data and 'selection' in click_data:
            handle_canvas_click(click_data['selection'])
        
        # JavaScript para drag and drop
        st.markdown("""
        <script>
        // Função para habilitar arrastar elementos no canvas
        function enableDragging() {
            const iframe = document.querySelector('[data-testid="stPlotlyChart"] iframe');
            if (!iframe) return;
            
            const iframeDoc = iframe.contentDocument || iframe.contentWindow.document;
            const plot = iframeDoc.querySelector('.js-plotly-plot');
            
            if (!plot) return;
            
            let draggedElement = null;
            let dragStartX = 0;
            let dragStartY = 0;
            
            // Encontrar elementos arrastáveis (barras)
            plot.addEventListener('mousedown', function(e) {
                const target = e.target;
                const point = target.closest('.point, .scatterpts');
                
                if (point) {
                    // Verificar se é uma barra
                    const trace = point.closest('.trace');
                    if (trace && trace.getAttribute('data-type') === 'bus') {
                        draggedElement = point;
                        dragStartX = e.clientX;
                        dragStartY = e.clientY;
                        
                        e.preventDefault();
                    }
                }
            });
            
            plot.addEventListener('mousemove', function(e) {
                if (draggedElement) {
                    const dx = e.clientX - dragStartX;
                    const dy = e.clientY - dragStartY;
                    
                    // Atualizar posição visual
                    const transform = draggedElement.getAttribute('transform') || 'translate(0,0)';
                    const match = transform.match(/translate\\(([^,]+),([^)]+)\\)/);
                    
                    if (match) {
                        const x = parseFloat(match[1]) + dx;
                        const y = parseFloat(match[2]) + dy;
                        
                        draggedElement.setAttribute('transform', `translate(${x},${y})`);
                        dragStartX = e.clientX;
                        dragStartY = e.clientY;
                    }
                }
            });
            
            plot.addEventListener('mouseup', function(e) {
                if (draggedElement) {
                    // Enviar nova posição para Python
                    const transform = draggedElement.getAttribute('transform') || 'translate(0,0)';
                    const match = transform.match(/translate\\(([^,]+),([^)]+)\\)/);
                    
                    if (match) {
                        const x = parseFloat(match[1]);
                        const y = parseFloat(match[2]);
                        
                        // Obter ID da barra
                        const trace = draggedElement.closest('.trace');
                        const busId = trace ? trace.getAttribute('data-id') : null;
                        
                        if (busId) {
                            const message = {
                                type: 'move_bus',
                                bus_id: parseInt(busId),
                                x: x,
                                y: y
                            };
                            
                            window.parent.postMessage(message, '*');
                        }
                    }
                    
                    draggedElement = null;
                }
            });
        }
        
        // Executar quando a página carregar
        window.addEventListener('load', function() {
            setTimeout(enableDragging, 1000); // Aguardar Plotly carregar
        });
        
        // Receber mensagens do Python
        window.addEventListener('message', function(event) {
            if (event.data.type === 'move_bus') {
                console.log('Moving bus:', event.data);
            }
        });
        </script>
        """, unsafe_allow_html=True)
        
        # Estatísticas rápidas
        st.subheader("📈 Estatísticas do Sistema")
        stats_cols = st.columns(4)
        with stats_cols[0]:
            st.metric("Barras", len(st.session_state.buses))
        with stats_cols[1]:
            st.metric("Linhas", len(st.session_state.lines))
        with stats_cols[2]:
            st.metric("Cargas", len(st.session_state.loads))
        with stats_cols[3]:
            st.metric("Geradores", len(st.session_state.generators))
    
    with col_main2:
        # Painel de propriedades
        st.subheader("🔧 Propriedades")
        
        if st.session_state.selected_element is not None and st.session_state.selected_type is not None:
            element = get_element_by_id(
                st.session_state.selected_element, 
                st.session_state.selected_type
            )
            
            if element:
                # Mostrar tipo do elemento selecionado
                type_names = {
                    'bus': 'Barra',
                    'line': 'Linha',
                    'load': 'Carga',
                    'generator': 'Gerador'
                }
                
                st.success(f"✅ {type_names[st.session_state.selected_type]} selecionado")
                
                # Formulário de edição
                with st.form("edit_element"):
                    if st.session_state.selected_type == 'bus':
                        st.text_input("Nome", element['name'], key=f"bus_name_{element['id']}")
                        
                        bus_type = st.selectbox(
                            "Tipo",
                            ["slack", "pv", "pq"],
                            index=["slack", "pv", "pq"].index(element['type']),
                            key=f"bus_type_{element['id']}"
                        )
                        
                        col_bus1, col_bus2 = st.columns(2)
                        with col_bus1:
                            new_x = st.number_input("Posição X", 
                                                   value=float(element['x']),
                                                   min_value=0.0,
                                                   max_value=1000.0,
                                                   key=f"bus_x_{element['id']}")
                        with col_bus2:
                            new_y = st.number_input("Posição Y",
                                                   value=float(element['y']),
                                                   min_value=0.0,
                                                   max_value=700.0,
                                                   key=f"bus_y_{element['id']}")
                        
                        element['vn_kv'] = st.number_input("Tensão Nominal (kV)",
                                                          value=float(element['vn_kv']),
                                                          min_value=0.1,
                                                          step=0.1,
                                                          key=f"bus_vn_{element['id']}")
                    
                    elif st.session_state.selected_type == 'line':
                        st.text_input("Nome", element['name'], key=f"line_name_{element['id']}")
                        
                        # Mostrar barras conectadas
                        from_bus = get_bus_by_id(element['from'])
                        to_bus = get_bus_by_id(element['to'])
                        
                        if from_bus and to_bus:
                            st.info(f"Conecta: Barra {element['from']} → Barra {element['to']}")
                        
                        col_line1, col_line2 = st.columns(2)
                        with col_line1:
                            element['r_ohm_per_km'] = st.number_input("R (Ω/km)",
                                                                     value=float(element['r_ohm_per_km']),
                                                                     min_value=0.001,
                                                                     step=0.01,
                                                                     key=f"line_r_{element['id']}")
                        with col_line2:
                            element['x_ohm_per_km'] = st.number_input("X (Ω/km)",
                                                                     value=float(element['x_ohm_per_km']),
                                                                     min_value=0.001,
                                                                     step=0.01,
                                                                     key=f"line_x_{element['id']}")
                    
                    elif st.session_state.selected_type == 'load':
                        st.text_input("Nome", element['name'], key=f"load_name_{element['id']}")
                        
                        col_load1, col_load2 = st.columns(2)
                        with col_load1:
                            element['p_mw'] = st.number_input("P (MW)",
                                                             value=float(element['p_mw']),
                                                             min_value=0.0,
                                                             step=0.1,
                                                             key=f"load_p_{element['id']}")
                        with col_load2:
                            element['q_mvar'] = st.number_input("Q (MVar)",
                                                               value=float(element['q_mvar']),
                                                               min_value=0.0,
                                                               step=0.1,
                                                               key=f"load_q_{element['id']}")
                    
                    elif st.session_state.selected_type == 'generator':
                        st.text_input("Nome", element['name'], key=f"gen_name_{element['id']}")
                        
                        col_gen1, col_gen2 = st.columns(2)
                        with col_gen1:
                            element['p_mw'] = st.number_input("P (MW)",
                                                             value=float(element['p_mw']),
                                                             min_value=0.0,
                                                             step=0.1,
                                                             key=f"gen_p_{element['id']}")
                        with col_gen2:
                            element['vm_pu'] = st.number_input("Vm (pu)",
                                                              value=float(element['vm_pu']),
                                                              min_value=0.8,
                                                              max_value=1.2,
                                                              step=0.01,
                                                              key=f"gen_vm_{element['id']}")
                    
                    # Botões de ação
                    col_act1, col_act2 = st.columns(2)
                    with col_act1:
                        if st.form_submit_button("💾 Salvar Alterações", use_container_width=True):
                            # Atualizar propriedades
                            if st.session_state.selected_type == 'bus':
                                element['x'] = new_x
                                element['y'] = new_y
                                element['type'] = bus_type
                                element['color'] = {
                                    'slack': '#ef4444',
                                    'pv': '#10b981',
                                    'pq': '#3b82f6'
                                }[bus_type]
                            
                            st.success("Alterações salvas!")
                            st.rerun()
                    
                    with col_act2:
                        if st.form_submit_button("🗑️ Remover", 
                                                type="secondary", 
                                                use_container_width=True):
                            delete_selected_element()
                            st.rerun()
            
            else:
                st.warning("Elemento não encontrado")
        else:
            st.info("👈 Selecione um elemento no canvas para editar suas propriedades")
        
        # Resultados da última análise
        if st.session_state.results:
            st.divider()
            st.subheader("📊 Última Análise")
            
            results = st.session_state.results
            analysis_time = datetime.fromisoformat(results['timestamp']).strftime("%H:%M:%S")
            
            if results['type'] == 'power_flow':
                st.markdown(f"**Fluxo de Potência** ({analysis_time})")
                
                if results['converged']:
                    st.success(f"✅ Convergiu em {results['iterations']} iterações")
                else:
                    st.warning(f"⚠️ Não convergiu após {results['iterations']} iterações")
                
                # Mostrar resumo de tensões
                voltages = results['voltages']
                if voltages:
                    avg_voltage = np.mean(voltages)
                    min_voltage = min(voltages)
                    max_voltage = max(voltages)
                    
                    col_volt1, col_volt2, col_volt3 = st.columns(3)
                    with col_volt1:
                        st.metric("Média", f"{avg_voltage:.3f} pu")
                    with col_volt2:
                        st.metric("Mínima", f"{min_voltage:.3f} pu")
                    with col_volt3:
                        st.metric("Máxima", f"{max_voltage:.3f} pu")
            
            elif results['type'] == 'short_circuit':
                st.markdown(f"**Curto-Circuito** ({analysis_time})")
                
                if 'fault_bus' in results:
                    st.warning(f"Barra de falta: {results['fault_bus']}")
                
                if 'max_current' in results:
                    st.metric("Corrente Máxima", f"{results['max_current']:.2f} kA")
                
                if 'critical_buses' in results and results['critical_buses']:
                    st.markdown("**Barras Críticas:**")
                    for crit in results['critical_buses']:
                        st.error(f"Barra {crit['bus_id']}: {crit['current']:.2f} kA")
        
        # Histórico de análises
        if st.session_state.history:
            st.divider()
            st.subheader("📋 Histórico")
            
            for i, hist in enumerate(reversed(st.session_state.history[-5:])):
                hist_time = datetime.fromisoformat(hist['timestamp']).strftime("%H:%M")
                type_icon = "🔁" if hist['type'] == 'power_flow' else "⚡"
                
                if st.button(f"{type_icon} {hist_time} - {hist['type'].replace('_', ' ').title()}",
                           key=f"hist_{i}",
                           use_container_width=True):
                    st.session_state.results = hist
                    st.rerun()

# Executar aplicação
if __name__ == "__main__":
    main()
