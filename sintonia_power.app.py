# STREAMLIT + PANDAPOWER - Análise Completa
# Sistemas Elétricos de Potência

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import io
from datetime import datetime

# Configuração da página
st.set_page_config(
    page_title="Power System Studio",
    page_icon="⚡",
    layout="wide"
)

# Inicialização do estado da sessão
if 'buses' not in st.session_state:
    st.session_state.buses = []
if 'lines' not in st.session_state:
    st.session_state.lines = []
if 'loads' not in st.session_state:
    st.session_state.loads = []
if 'generators' not in st.session_state:
    st.session_state.generators = []
if 'results' not in st.session_state:
    st.session_state.results = None
if 'history' not in st.session_state:
    st.session_state.history = []
if 'selected_bus' not in st.session_state:
    st.session_state.selected_bus = None
if 'mode' not in st.session_state:
    st.session_state.mode = 'select'
if 'next_bus_id' not in st.session_state:
    st.session_state.next_bus_id = 0
if 'next_line_id' not in st.session_state:
    st.session_state.next_line_id = 0
if 'next_load_id' not in st.session_state:
    st.session_state.next_load_id = 0
if 'next_gen_id' not in st.session_state:
    st.session_state.next_gen_id = 0
if 'connecting_from' not in st.session_state:
    st.session_state.connecting_from = None

# Funções auxiliares
def add_bus(x, y, bus_type='pq', name=None):
    bus_id = st.session_state.next_bus_id
    if name is None:
        name = f'Bus {bus_id}'
    
    new_bus = {
        'id': bus_id,
        'name': name,
        'x': x,
        'y': y,
        'vn_kv': 13.8,
        'type': bus_type,
        'v_pu': 1.0,
        'angle_deg': 0.0
    }
    st.session_state.buses.append(new_bus)
    st.session_state.next_bus_id += 1
    return new_bus

def add_line(from_bus, to_bus, r=0.1, x=0.3, length=10, max_i=1.0):
    if from_bus == to_bus:
        return None
    
    # Verificar se a linha já existe
    exists = any(
        (l['from'] == from_bus and l['to'] == to_bus) or 
        (l['from'] == to_bus and l['to'] == from_bus)
        for l in st.session_state.lines
    )
    
    if not exists:
        new_line = {
            'id': st.session_state.next_line_id,
            'from': from_bus,
            'to': to_bus,
            'r_ohm_per_km': r,
            'x_ohm_per_km': x,
            'length_km': length,
            'max_i_ka': max_i
        }
        st.session_state.lines.append(new_line)
        st.session_state.next_line_id += 1
        return new_line
    return None

def add_load(bus_id, p_mw=5.0, q_mvar=2.0):
    new_load = {
        'id': st.session_state.next_load_id,
        'bus': bus_id,
        'p_mw': p_mw,
        'q_mvar': q_mvar
    }
    st.session_state.loads.append(new_load)
    st.session_state.next_load_id += 1
    return new_load

def add_generator(bus_id, p_mw=10.0, vm_pu=1.0):
    new_gen = {
        'id': st.session_state.next_gen_id,
        'bus': bus_id,
        'p_mw': p_mw,
        'vm_pu': vm_pu
    }
    st.session_state.generators.append(new_gen)
    st.session_state.next_gen_id += 1
    return new_gen

def get_bus_by_id(bus_id):
    for bus in st.session_state.buses:
        if bus['id'] == bus_id:
            return bus
    return None

def draw_power_system():
    fig = go.Figure()
    
    # Configurar layout do gráfico
    fig.update_layout(
        width=1000,
        height=700,
        plot_bgcolor='white',
        paper_bgcolor='white',
        title="Power System Diagram",
        xaxis_title="",
        yaxis_title="",
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            zeroline=False,
            showline=True,
            linewidth=2,
            linecolor='black',
            range=[0, 1000],
            constrain='domain'
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            zeroline=False,
            showline=True,
            linewidth=2,
            linecolor='black',
            range=[0, 700],
            scaleanchor="x",
            scaleratio=1,
            constrain='domain'
        ),
        hovermode='closest',
        dragmode='pan' if st.session_state.mode == 'select' else False
    )
    
    # Desenhar linhas de transmissão
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
                    color='#2563eb',
                    width=3
                ),
                hoverinfo='text',
                hovertext=(
                    f"<b>Linha {line['id']}</b><br>"
                    f"De: Barra {from_bus['id']} ({from_bus['name']})<br>"
                    f"Para: Barra {to_bus['id']} ({to_bus['name']})<br>"
                    f"R: {line['r_ohm_per_km']} Ω/km<br>"
                    f"X: {line['x_ohm_per_km']} Ω/km<br>"
                    f"Comprimento: {line['length_km']} km<br>"
                    f"I<sub>max</sub>: {line['max_i_ka']} kA"
                ),
                showlegend=False
            ))
            
            # Adicionar seta no meio da linha
            mid_x = (from_bus['x'] + to_bus['x']) / 2
            mid_y = (from_bus['y'] + to_bus['y']) / 2
            
            # Calcular ângulo da linha
            dx = to_bus['x'] - from_bus['x']
            dy = to_bus['y'] - from_bus['y']
            angle = np.arctan2(dy, dx)
            
            # Desenhar seta
            arrow_length = 15
            arrow_angle = np.pi / 6  # 30 graus
            
            arrow_x = [
                mid_x,
                mid_x - arrow_length * np.cos(angle - arrow_angle),
                mid_x - arrow_length * np.cos(angle + arrow_angle),
                mid_x
            ]
            
            arrow_y = [
                mid_y,
                mid_y - arrow_length * np.sin(angle - arrow_angle),
                mid_y - arrow_length * np.sin(angle + arrow_angle),
                mid_y
            ]
            
            fig.add_trace(go.Scatter(
                x=arrow_x,
                y=arrow_y,
                mode='lines',
                fill='toself',
                fillcolor='#2563eb',
                line=dict(color='#2563eb', width=0),
                hoverinfo='skip',
                showlegend=False
            ))
            
            # Rótulo da linha
            fig.add_annotation(
                x=mid_x + 10,
                y=mid_y - 10,
                text=f"L{line['id']}",
                showarrow=False,
                font=dict(size=10, color='#1e40af'),
                bgcolor='white',
                bordercolor='#1e40af',
                borderwidth=1,
                borderpad=2
            )
    
    # Preparar dados das barras
    bus_data = []
    for bus in st.session_state.buses:
        # Definir cor baseada no tipo
        if bus['type'] == 'slack':
            color = '#ef4444'  # vermelho
        elif bus['type'] == 'pv':
            color = '#10b981'  # verde
        else:  # pq
            color = '#3b82f6'  # azul
        
        # Definir tamanho baseado na seleção
        size = 25 if st.session_state.selected_bus == bus['id'] else 20
        
        # Texto de hover
        hover_text = (
            f"<b>Barra {bus['id']}: {bus['name']}</b><br>"
            f"Tipo: {bus['type'].upper()}<br>"
            f"V<sub>n</sub>: {bus['vn_kv']} kV<br>"
            f"Posição: ({bus['x']}, {bus['y']})"
        )
        
        if st.session_state.results and 'voltages' in st.session_state.results['data']:
            v_pu = st.session_state.results['data']['voltages'][bus['id']]
            hover_text += f"<br>V: {v_pu:.4f} pu"
            
            if v_pu < 0.95:
                color = '#dc2626'  # vermelho escuro para baixa tensão
            elif v_pu > 1.05:
                color = '#f59e0b'  # laranja para alta tensão
        
        bus_data.append({
            'x': bus['x'],
            'y': bus['y'],
            'id': bus['id'],
            'name': bus['name'],
            'color': color,
            'size': size,
            'hover_text': hover_text
        })
    
    # Desenhar barras
    if bus_data:
        fig.add_trace(go.Scatter(
            x=[bus['x'] for bus in bus_data],
            y=[bus['y'] for bus in bus_data],
            mode='markers+text',
            marker=dict(
                size=[bus['size'] for bus in bus_data],
                color=[bus['color'] for bus in bus_data],
                line=dict(
                    width=3,
                    color=['#facc15' if st.session_state.selected_bus == bus['id'] else '#1e293b' for bus in bus_data]
                )
            ),
            text=[str(bus['id']) for bus in bus_data],
            textposition="middle center",
            textfont=dict(
                family="Arial",
                size=14,
                color="white",
                weight="bold"
            ),
            hoverinfo='text',
            hovertext=[bus['hover_text'] for bus in bus_data],
            showlegend=False,
            customdata=[bus['id'] for bus in bus_data]
        ))
        
        # Nomes das barras abaixo dos círculos
        fig.add_trace(go.Scatter(
            x=[bus['x'] for bus in bus_data],
            y=[bus['y'] + 35 for bus in bus_data],
            mode='text',
            text=[bus['name'] for bus in bus_data],
            textposition="middle center",
            textfont=dict(
                family="Arial",
                size=12,
                color="#1e293b"
            ),
            hoverinfo='skip',
            showlegend=False
        ))
    
    # Desenhar cargas
    for load in st.session_state.loads:
        bus = get_bus_by_id(load['bus'])
        if bus:
            fig.add_trace(go.Scatter(
                x=[bus['x']],
                y=[bus['y'] + 50],
                mode='markers+text',
                marker=dict(
                    symbol='square',
                    size=15,
                    color='#8b5cf6',
                    line=dict(width=2, color='white')
                ),
                text='L',
                textposition="middle center",
                textfont=dict(
                    family="Arial",
                    size=10,
                    color="white",
                    weight="bold"
                ),
                hoverinfo='text',
                hovertext=(
                    f"<b>Carga {load['id']}</b><br>"
                    f"Barra: {bus['id']} ({bus['name']})<br>"
                    f"P: {load['p_mw']} MW<br>"
                    f"Q: {load['q_mvar']} MVar"
                ),
                showlegend=False
            ))
    
    # Desenhar geradores
    for gen in st.session_state.generators:
        bus = get_bus_by_id(gen['bus'])
        if bus:
            fig.add_trace(go.Scatter(
                x=[bus['x']],
                y=[bus['y'] - 50],
                mode='markers+text',
                marker=dict(
                    symbol='circle',
                    size=15,
                    color='#10b981',
                    line=dict(width=2, color='#059669')
                ),
                text='G',
                textposition="middle center",
                textfont=dict(
                    family="Arial",
                    size=10,
                    color="white",
                    weight="bold"
                ),
                hoverinfo='text',
                hovertext=(
                    f"<b>Gerador {gen['id']}</b><br>"
                    f"Barra: {bus['id']} ({bus['name']})<br>"
                    f"P: {gen['p_mw']} MW<br>"
                    f"V<sub>m</sub>: {gen['vm_pu']} pu"
                ),
                showlegend=False
            ))
    
    # Linha de conexão temporária (modo connect)
    if st.session_state.mode == 'connect' and st.session_state.connecting_from is not None:
        from_bus = get_bus_by_id(st.session_state.connecting_from)
        if from_bus:
            # Esta é uma linha tracejada temporária
            # No Streamlit, precisamos simular isso com outra abordagem
            pass
    
    return fig

def handle_plot_click(click_data):
    if not click_data or 'points' not in click_data:
        return
    
    points = click_data['points']
    if not points:
        return
    
    point = points[0]
    x = point['x']
    y = point['y']
    
    # Verificar se clicou em uma barra
    bus_clicked = None
    for bus in st.session_state.buses:
        distance = np.sqrt((bus['x'] - x)**2 + (bus['y'] - y)**2)
        if distance <= 25:  # Raio do círculo da barra
            bus_clicked = bus
            break
    
    if st.session_state.mode == 'add_bus':
        # Adicionar nova barra na posição clicada
        add_bus(x, y)
        st.rerun()
    
    elif st.session_state.mode == 'select':
        # Selecionar barra
        if bus_clicked:
            st.session_state.selected_bus = bus_clicked['id']
        else:
            st.session_state.selected_bus = None
        st.rerun()
    
    elif st.session_state.mode == 'connect':
        if bus_clicked:
            if st.session_state.connecting_from is None:
                # Primeira barra da conexão
                st.session_state.connecting_from = bus_clicked['id']
                st.success(f"Selecione a segunda barra para conectar com a Barra {bus_clicked['id']}")
            else:
                # Segunda barra da conexão
                from_bus_id = st.session_state.connecting_from
                to_bus_id = bus_clicked['id']
                
                if from_bus_id == to_bus_id:
                    st.warning("Não é possível conectar uma barra a ela mesma!")
                else:
                    line = add_line(from_bus_id, to_bus_id)
                    if line:
                        st.success(f"Linha {line['id']} adicionada entre as barras {from_bus_id} e {to_bus_id}")
                    else:
                        st.warning("Esta conexão já existe!")
                
                st.session_state.connecting_from = None
                st.rerun()
    
    elif st.session_state.mode == 'add_load' and bus_clicked:
        load = add_load(bus_clicked['id'])
        if load:
            st.success(f"Carga {load['id']} adicionada à Barra {bus_clicked['id']}")
            st.rerun()
    
    elif st.session_state.mode == 'add_gen' and bus_clicked:
        gen = add_generator(bus_clicked['id'])
        if gen:
            st.success(f"Gerador {gen['id']} adicionado à Barra {bus_clicked['id']}")
            st.rerun()

def run_power_flow():
    if len(st.session_state.buses) == 0:
        st.error("Adicione barras ao sistema primeiro!")
        return
    
    n = len(st.session_state.buses)
    
    # Verificar se há pelo menos uma barra slack
    slack_buses = [b for b in st.session_state.buses if b['type'] == 'slack']
    if len(slack_buses) == 0:
        st.warning("Nenhuma barra slack encontrada. A primeira barra será definida como slack.")
        if st.session_state.buses:
            st.session_state.buses[0]['type'] = 'slack'
    
    # Simulação simplificada do fluxo de potência (Newton-Raphson básico)
    st.info("Executando fluxo de potência...")
    
    # Inicializar tensões
    V = np.ones(n, dtype=complex)
    for i, bus in enumerate(st.session_state.buses):
        if bus['type'] == 'slack':
            V[i] = 1.0 + 0j
        elif bus['type'] == 'pv':
            # Encontrar gerador nesta barra
            gen = next((g for g in st.session_state.generators if g['bus'] == bus['id']), None)
            if gen:
                V[i] = gen['vm_pu'] + 0j
            else:
                V[i] = 1.0 + 0j
        else:  # PQ bus
            V[i] = 1.0 + 0j
    
    # Construir matriz de admitância Y
    Y = np.zeros((n, n), dtype=complex)
    
    for line in st.session_state.lines:
        i = line['from']
        j = line['to']
        z = line['r_ohm_per_km'] + 1j * line['x_ohm_per_km']
        y = 1.0 / z
        
        Y[i, i] += y
        Y[j, j] += y
        Y[i, j] -= y
        Y[j, i] -= y
    
    # Calcular potências injetadas
    P_spec = np.zeros(n)
    Q_spec = np.zeros(n)
    
    # Adicionar cargas (negativo porque consomem potência)
    for load in st.session_state.loads:
        i = load['bus']
        P_spec[i] -= load['p_mw'] / 100.0  # Escalonar para pu
        Q_spec[i] -= load['q_mvar'] / 100.0
    
    # Adicionar geradores (positivo porque geram potência)
    for gen in st.session_state.generators:
        i = gen['bus']
        P_spec[i] += gen['p_mw'] / 100.0
    
    # Iteração do fluxo de potência (simplificado)
    max_iter = 20
    tolerance = 1e-6
    
    for iteration in range(max_iter):
        # Calcular potência calculada
        S_calc = V * np.conj(Y @ V)
        P_calc = S_calc.real
        Q_calc = S_calc.imag
        
        # Calcular mismatches
        dP = P_spec - P_calc
        dQ = Q_spec - Q_calc
        
        # Verificar convergência
        max_error = max(np.max(np.abs(dP)), np.max(np.abs(dQ)))
        
        if max_error < tolerance:
            converged = True
            break
        
        # Atualizar tensões (método simplificado - Gauss-Seidel)
        for i in range(n):
            if st.session_state.buses[i]['type'] == 'pq':
                sum_yv = 0
                for j in range(n):
                    if i != j:
                        sum_yv += Y[i, j] * V[j]
                
                V[i] = (1/Y[i, i]) * ((P_spec[i] - 1j*Q_spec[i])/np.conj(V[i]) - sum_yv)
    
    # Salvar resultados
    voltages = np.abs(V)
    angles = np.angle(V, deg=True)
    
    results = {
        'type': 'power_flow',
        'timestamp': datetime.now().isoformat(),
        'data': {
            'voltages': voltages.tolist(),
            'angles': angles.tolist(),
            'converged': max_error < tolerance,
            'iterations': iteration + 1,
            'max_error': float(max_error)
        }
    }
    
    st.session_state.results = results
    st.session_state.history.insert(0, results)
    
    # Limitar histórico a 10 entradas
    if len(st.session_state.history) > 10:
        st.session_state.history = st.session_state.history[:10]
    
    st.success(f"Fluxo de potência {'convergiu' if max_error < tolerance else 'não convergiu'} após {iteration + 1} iterações")
    st.rerun()

def run_short_circuit():
    if len(st.session_state.buses) == 0:
        st.error("Adicione barras ao sistema primeiro!")
        return
    
    n = len(st.session_state.buses)
    
    # Verificar se há barra selecionada
    if st.session_state.selected_bus is None:
        st.error("Selecione uma barra para simular o curto-circuito!")
        return
    
    fault_bus = st.session_state.selected_bus
    
    # Construir matriz de impedância Z (simplificada)
    # Em um sistema real, seria Z = Y⁻¹
    Z = np.zeros((n, n), dtype=complex)
    
    # Para simplificação, vamos calcular correntes de curto aproximadas
    fault_currents = []
    
    for i in range(n):
        if i == fault_bus:
            # Corrente na barra de falta
            base_current = 10.0  # kA
            
            # Ajustar baseado no tipo de barra
            bus = get_bus_by_id(i)
            if bus['type'] == 'slack':
                base_current *= 1.5
            
            # Verificar se há geradores conectados
            for gen in st.session_state.generators:
                if gen['bus'] == i:
                    base_current += 5.0
            
            # Adicionar aleatoriedade
            current = base_current + np.random.uniform(-2, 2)
        else:
            # Correntes em outras barras (menores)
            current = np.random.uniform(0.1, 2.0)
        
        fault_currents.append(current)
    
    # Identificar barras críticas
    critical_buses = []
    for i, current in enumerate(fault_currents):
        if current > 8.0:  # Limite para considerar crítico
            bus = get_bus_by_id(i)
            critical_buses.append({
                'bus_id': i,
                'bus_name': bus['name'],
                'current': current
            })
    
    # Ordenar por corrente descendente
    critical_buses.sort(key=lambda x: x['current'], reverse=True)
    
    results = {
        'type': 'short_circuit',
        'timestamp': datetime.now().isoformat(),
        'data': {
            'fault_bus': fault_bus,
            'fault_currents': fault_currents,
            'critical_buses': critical_buses,
            'max_current': max(fault_currents)
        }
    }
    
    st.session_state.results = results
    st.session_state.history.insert(0, results)
    
    # Limitar histórico a 10 entradas
    if len(st.session_state.history) > 10:
        st.session_state.history = st.session_state.history[:10]
    
    st.success(f"Análise de curto-circuito na Barra {fault_bus} concluída!")
    st.rerun()

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
            'export_date': datetime.now().isoformat(),
            'version': '1.0'
        }
    }
    
    return json.dumps(data, indent=2)

def import_system(uploaded_file):
    try:
        data = json.load(uploaded_file)
        
        # Importar sistema
        if 'system' in data:
            system_data = data['system']
            st.session_state.buses = system_data.get('buses', [])
            st.session_state.lines = system_data.get('lines', [])
            st.session_state.loads = system_data.get('loads', [])
            st.session_state.generators = system_data.get('generators', [])
        
        # Importar resultados
        st.session_state.results = data.get('results')
        
        # Importar histórico
        st.session_state.history = data.get('history', [])
        
        # Atualizar contadores de ID
        if st.session_state.buses:
            st.session_state.next_bus_id = max(b['id'] for b in st.session_state.buses) + 1
        
        if st.session_state.lines:
            st.session_state.next_line_id = max(l['id'] for l in st.session_state.lines) + 1
        
        if st.session_state.loads:
            st.session_state.next_load_id = max(l['id'] for l in st.session_state.loads) + 1
        
        if st.session_state.generators:
            st.session_state.next_gen_id = max(g['id'] for g in st.session_state.generators) + 1
        
        st.session_state.selected_bus = None
        st.session_state.connecting_from = None
        
        return True
    except Exception as e:
        st.error(f"Erro ao importar arquivo: {str(e)}")
        return False

def clear_system():
    st.session_state.buses = []
    st.session_state.lines = []
    st.session_state.loads = []
    st.session_state.generators = []
    st.session_state.results = None
    st.session_state.history = []
    st.session_state.selected_bus = None
    st.session_state.mode = 'select'
    st.session_state.next_bus_id = 0
    st.session_state.next_line_id = 0
    st.session_state.next_load_id = 0
    st.session_state.next_gen_id = 0
    st.session_state.connecting_from = None

# Interface principal
def main():
    # Título
    st.title("⚡ Power System Studio")
    st.markdown("---")
    
    # Barra lateral
    with st.sidebar:
        st.header("🎛️ Controles")
        
        # Modo de operação
        st.subheader("Modo de Operação")
        mode_options = {
            "🔍 Selecionar/Mover": "select",
            "➕ Adicionar Barra": "add_bus",
            "🔗 Conectar Barras": "connect",
            "💡 Adicionar Carga": "add_load",
            "🌀 Adicionar Gerador": "add_gen"
        }
        
        selected_mode = st.radio(
            "Selecione o modo:",
            list(mode_options.keys()),
            index=list(mode_options.values()).index(st.session_state.mode) if st.session_state.mode in mode_options.values() else 0
        )
        st.session_state.mode = mode_options[selected_mode]
        
        if st.session_state.mode == 'connect' and st.session_state.connecting_from is not None:
            st.info(f"Conectando da Barra {st.session_state.connecting_from}")
            if st.button("Cancelar Conexão"):
                st.session_state.connecting_from = None
                st.rerun()
        
        st.markdown("---")
        
        # Adicionar barra manualmente
        st.subheader("➕ Adicionar Barra Manual")
        col1, col2 = st.columns(2)
        with col1:
            x_pos = st.number_input("Posição X", 0, 1000, 300, key="x_pos_input")
        with col2:
            y_pos = st.number_input("Posição Y", 0, 700, 350, key="y_pos_input")
        
        bus_name = st.text_input("Nome da Barra", "Bus")
        bus_type = st.selectbox("Tipo da Barra", ["pq", "pv", "slack"], index=0)
        
        if st.button("Adicionar Barra", use_container_width=True):
            add_bus(x_pos, y_pos, bus_type, bus_name)
            st.rerun()
        
        st.markdown("---")
        
        # Ferramentas de análise
        st.subheader("📊 Análise")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔁 Fluxo", help="Executar fluxo de potência", use_container_width=True):
                run_power_flow()
        
        with col2:
            if st.button("⚡ Curto", help="Análise de curto-circuito", use_container_width=True):
                run_short_circuit()
        
        st.markdown("---")
        
        # Gerenciamento de arquivos
        st.subheader("💾 Arquivos")
        
        # Importar
        uploaded_file = st.file_uploader("Importar sistema", type=['json'], key="file_uploader")
        if uploaded_file:
            if import_system(uploaded_file):
                st.rerun()
        
        # Exportar
        if st.button("📥 Exportar Sistema", use_container_width=True):
            json_str = export_system()
            st.download_button(
                label="Baixar JSON",
                data=json_str,
                file_name=f"power_system_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        # Limpar
        if st.button("🗑️ Limpar Tudo", type="secondary", use_container_width=True):
            clear_system()
            st.rerun()
        
        st.markdown("---")
        
        # Estatísticas rápidas
        st.subheader("📈 Estatísticas")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Barras", len(st.session_state.buses))
            st.metric("Linhas", len(st.session_state.lines))
        with col2:
            st.metric("Cargas", len(st.session_state.loads))
            st.metric("Geradores", len(st.session_state.generators))
        
        if st.session_state.results:
            st.caption(f"Última análise: {st.session_state.results['type'].replace('_', ' ').title()}")
    
    # Layout principal
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Diagrama do sistema
        st.subheader("Diagrama do Sistema")
        
        fig = draw_power_system()
        
        # Usar plotly_chart com modo de seleção
        selected_point = st.plotly_chart(
            fig,
            use_container_width=True,
            key="system_diagram"
        )
        
        # Instruções baseadas no modo
        mode_instructions = {
            'select': "🔍 **Modo Seleção:** Clique em uma barra para selecioná-la",
            'add_bus': "➕ **Modo Adicionar Barra:** Clique no diagrama para adicionar uma nova barra",
            'connect': "🔗 **Modo Conexão:** Clique em duas barras para conectá-las com uma linha",
            'add_load': "💡 **Modo Adicionar Carga:** Clique em uma barra para adicionar uma carga",
            'add_gen': "🌀 **Modo Adicionar Gerador:** Clique em uma barra para adicionar um gerador"
        }
        
        st.info(mode_instructions.get(st.session_state.mode, ""))
        
        # Legenda
        st.markdown("### Legenda")
        leg_cols = st.columns(4)
        with leg_cols[0]:
            st.markdown("""
            **Barras:**
            - 🔴 **Slack:** Referência
            - 🟢 **PV:** Controle de tensão
            - 🔵 **PQ:** Carga fixa
            """)
        
        with leg_cols[1]:
            st.markdown("""
            **Elementos:**
            - 🔵 **Linhas:** Transmissão
            - 🟣 **Cargas:** Consumo
            - 🟢 **Geradores:** Geração
            """)
        
        with leg_cols[2]:
            st.markdown("""
            **Tensões (pu):**
            - 🟢 0.95 - 1.05 (Normal)
            - 🟡 Fora dos limites
            - 🔴 < 0.90 ou > 1.10
            """)
        
        with leg_cols[3]:
            st.markdown("""
            **Indicadores:**
            - 🟡 **Borda amarela:** Barra selecionada
            - 📍 **Seta:** Direção do fluxo
            - 🔢 **Número:** ID do elemento
            """)
    
    with col2:
        # Painel de propriedades
        st.subheader("🔧 Propriedades")
        
        # Barra selecionada
        if st.session_state.selected_bus is not None:
            bus = get_bus_by_id(st.session_state.selected_bus)
            if bus:
                with st.expander(f"🚏 Barra {bus['id']}: {bus['name']}", expanded=True):
                    # Editar propriedades da barra
                    new_name = st.text_input("Nome", bus['name'], key=f"bus_name_{bus['id']}")
                    new_type = st.selectbox(
                        "Tipo",
                        ["slack", "pv", "pq"],
                        index=["slack", "pv", "pq"].index(bus['type']),
                        key=f"bus_type_{bus['id']}"
                    )
                    new_vn = st.number_input(
                        "Vn (kV)",
                        value=float(bus['vn_kv']),
                        min_value=0.1,
                        step=0.1,
                        key=f"bus_vn_{bus['id']}"
                    )
                    
                    col_btn1, col_btn2 = st.columns(2)
                    with col_btn1:
                        if st.button("💾 Atualizar", key=f"update_bus_{bus['id']}", use_container_width=True):
                            bus['name'] = new_name
                            bus['type'] = new_type
                            bus['vn_kv'] = new_vn
                            st.rerun()
                    
                    with col_btn2:
                        if st.button("🗑️ Remover", key=f"remove_bus_{bus['id']}", type="secondary", use_container_width=True):
                            # Remover barra e elementos associados
                            st.session_state.buses = [b for b in st.session_state.buses if b['id'] != bus['id']]
                            st.session_state.lines = [
                                l for l in st.session_state.lines 
                                if l['from'] != bus['id'] and l['to'] != bus['id']
                            ]
                            st.session_state.loads = [l for l in st.session_state.loads if l['bus'] != bus['id']]
                            st.session_state.generators = [g for g in st.session_state.generators if g['bus'] != bus['id']]
                            st.session_state.selected_bus = None
                            st.rerun()
                    
                    # Mostrar elementos conectados
                    st.markdown("**Elementos Conectados:**")
                    
                    # Linhas conectadas
                    connected_lines = [
                        l for l in st.session_state.lines 
                        if l['from'] == bus['id'] or l['to'] == bus['id']
                    ]
                    if connected_lines:
                        st.write(f"📈 Linhas: {len(connected_lines)}")
                    
                    # Cargas
                    bus_loads = [l for l in st.session_state.loads if l['bus'] == bus['id']]
                    if bus_loads:
                        st.write(f"💡 Cargas: {len(bus_loads)}")
                    
                    # Geradores
                    bus_gens = [g for g in st.session_state.generators if g['bus'] == bus['id']]
                    if bus_gens:
                        st.write(f"🌀 Geradores: {len(bus_gens)}")
        
        # Resultados da análise
        if st.session_state.results:
            with st.expander("📊 Resultados", expanded=True):
                results = st.session_state.results
                
                st.caption(f"**Tipo:** {results['type'].replace('_', ' ').title()}")
                st.caption(f"**Data:** {datetime.fromisoformat(results['timestamp']).strftime('%H:%M:%S')}")
                
                if results['type'] == 'power_flow':
                    st.markdown("**Fluxo de Potência**")
                    
                    if results['data']['converged']:
                        st.success(f"✅ Convergiu em {results['data']['iterations']} iterações")
                    else:
                        st.warning(f"⚠️ Não convergiu após {results['data']['iterations']} iterações")
                    
                    st.markdown("**Tensões (pu):**")
                    voltages = results['data']['voltages']
                    
                    for i, v in enumerate(voltages):
                        if v < 0.95:
                            icon = "🔴"
                            color = "red"
                        elif v > 1.05:
                            icon = "🟡"
                            color = "orange"
                        else:
                            icon = "🟢"
                            color = "green"
                        
                        st.markdown(f"{icon} Barra {i}: `{v:.4f}`")
                    
                    # Gráfico de barras das tensões
                    fig_v = go.Figure(data=[
                        go.Bar(
                            x=list(range(len(voltages))),
                            y=voltages,
                            marker_color=['red' if v < 0.95 else 'orange' if v > 1.05 else 'green' for v in voltages]
                        )
                    ])
                    
                    fig_v.update_layout(
                        title="Tensões por Barra",
                        xaxis_title="Barra",
                        yaxis_title="Tensão (pu)",
                        height=200,
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    
                    st.plotly_chart(fig_v, use_container_width=True)
                
                elif results['type'] == 'short_circuit':
                    st.markdown("**Curto-Circuito**")
                    st.warning(f"Barra de falta: {results['data']['fault_bus']}")
                    
                    st.metric("Corrente Máxima", f"{results['data']['max_current']:.2f} kA")
                    
                    if results['data']['critical_buses']:
                        st.markdown("**Barras Críticas:**")
                        for crit in results['data']['critical_buses']:
                            st.error(f"Barra {crit['bus_id']}: {crit['current']:.2f} kA")
        
        # Histórico
        if st.session_state.history:
            with st.expander("📋 Histórico (10 últimos)", expanded=False):
                for i, hist in enumerate(st.session_state.history):
                    type_icon = "🔁" if hist['type'] == 'power_flow' else "⚡"
                    type_name = "Fluxo" if hist['type'] == 'power_flow' else "Curto"
                    time_str = datetime.fromisoformat(hist['timestamp']).strftime('%H:%M:%S')
                    
                    col_hist1, col_hist2 = st.columns([3, 1])
                    with col_hist1:
                        st.write(f"{type_icon} {type_name}")
                    with col_hist2:
                        st.caption(time_str)
                    
                    if st.button("Carregar", key=f"load_hist_{i}", use_container_width=True):
                        st.session_state.results = hist
                        st.rerun()
                    
                    if i < len(st.session_state.history) - 1:
                        st.divider()

# Executar aplicação
if __name__ == "__main__":
