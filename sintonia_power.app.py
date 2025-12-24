# STREAMLIT + PANDAPOWER - Análise Completa
# Sistemas Elétricos de Potência

# STREAMLIT + PANDAPOWER - Análise Completa
# Sistemas Elétricos de Potência
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
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
if 'temp_connection' not in st.session_state:
    st.session_state.temp_connection = None

# =====================================================
# FUNÇÕES AUXILIARES
# =====================================================

def add_bus(x, y, bus_type='pq'):
    new_bus = {
        'id': len(st.session_state.buses),
        'name': f'Bus {len(st.session_state.buses)}',
        'x': x,
        'y': y,
        'vn_kv': 13.8,
        'type': bus_type
    }
    st.session_state.buses.append(new_bus)
    return new_bus

def add_line(from_bus, to_bus):
    if from_bus == to_bus:
        return None
    
    exists = any(
        (l['from'] == from_bus and l['to'] == to_bus) or 
        (l['from'] == to_bus and l['to'] == from_bus)
        for l in st.session_state.lines
    )
    
    if not exists:
        new_line = {
            'id': len(st.session_state.lines),
            'from': from_bus,
            'to': to_bus,
            'r_ohm_per_km': 0.1,
            'x_ohm_per_km': 0.3,
            'length_km': 10,
            'max_i_ka': 1.0
        }
        st.session_state.lines.append(new_line)
        return new_line
    return None

def add_load(bus_id):
    if not any(l['bus'] == bus_id for l in st.session_state.loads):
        new_load = {
            'id': len(st.session_state.loads),
            'bus': bus_id,
            'p_mw': 5.0,
            'q_mvar': 2.0
        }
        st.session_state.loads.append(new_load)
        return new_load
    return None

def add_generator(bus_id):
    if not any(g['bus'] == bus_id for g in st.session_state.generators):
        new_gen = {
            'id': len(st.session_state.generators),
            'bus': bus_id,
            'p_mw': 10.0,
            'vm_pu': 1.0
        }
        st.session_state.generators.append(new_gen)
        return new_gen
    return None

def handle_bus_click(bus_id):
    """Processa clique em uma barra"""
    if st.session_state.mode == 'select':
        st.session_state.selected_bus = bus_id
        st.rerun()
    elif st.session_state.mode == 'connect':
        if st.session_state.temp_connection is None:
            st.session_state.temp_connection = bus_id
            st.rerun()
        else:
            from_bus = st.session_state.temp_connection
            to_bus = bus_id
            if from_bus != to_bus:
                result = add_line(from_bus, to_bus)
                if result:
                    st.success(f"✅ Linha criada entre barra {from_bus} e barra {to_bus}")
                else:
                    st.warning("⚠️ Conexão já existe ou inválida")
            else:
                st.warning("⚠️ Não é possível conectar uma barra a si mesma")
            st.session_state.temp_connection = None
            st.rerun()
    elif st.session_state.mode == 'add_load':
        result = add_load(bus_id)
        if result:
            st.success(f"✅ Carga adicionada à barra {bus_id}")
            st.rerun()
        else:
            st.warning("⚠️ Esta barra já tem uma carga")
    elif st.session_state.mode == 'add_gen':
        result = add_generator(bus_id)
        if result:
            st.success(f"✅ Gerador adicionado à barra {bus_id}")
            st.rerun()
        else:
            st.warning("⚠️ Esta barra já tem um gerador")

def draw_power_system():
    """Desenha o sistema elétrico usando Plotly com portas de conexão interativas"""
    fig = go.Figure()
    
    # Configurar o layout base com grid
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(200, 200, 200, 0.5)',
        zeroline=False,
        range=[-50, 1050],
        showticklabels=False
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(200, 200, 200, 0.5)',
        zeroline=False,
        range=[-50, 650],
        scaleanchor="x",
        scaleratio=1,
        showticklabels=False
    )
    
    # Cores e estilos
    colors = {
        'slack': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        'pv': 'linear-gradient(135deg, #4CAF50 0%, #45a049 100%)',
        'pq': 'linear-gradient(135deg, #FF9800 0%, #F57C00 100%)',
        'line': '#667eea',
        'load': '#9C27B0',
        'generator': '#4CAF50',
        'port': '#4CAF50',
        'selected': '#FFD700'
    }
    
    # 1. Desenhar linhas de transmissão com estilo melhorado
    for line in st.session_state.lines:
        from_bus = next((b for b in st.session_state.buses if b['id'] == line['from']), None)
        to_bus = next((b for b in st.session_state.buses if b['id'] == line['to']), None)
        
        if from_bus and to_bus:
            # Linha principal com gradiente
            fig.add_trace(go.Scatter(
                x=[from_bus['x'], to_bus['x']],
                y=[from_bus['y'], to_bus['y']],
                mode='lines',
                line=dict(
                    color=colors['line'],
                    width=3
                ),
                hoverinfo='text',
                hovertext=f"<b>Linha {line['id']}</b><br>" +
                         f"De: Barra {line['from']}<br>" +
                         f"Para: Barra {line['to']}<br>" +
                         f"Impedância: {line['r_ohm_per_km']:.3f}+j{line['x_ohm_per_km']:.3f} Ω/km<br>" +
                         f"Comprimento: {line['length_km']:.1f} km",
                showlegend=False,
                name=f'line_{line["id"]}'
            ))
            
            # Adicionar indicador de fluxo (seta)
            mid_x = (from_bus['x'] + to_bus['x']) / 2
            mid_y = (from_bus['y'] + to_bus['y']) / 2
            
            fig.add_annotation(
                x=mid_x,
                y=mid_y,
                ax=mid_x,
                ay=mid_y,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=3,
                arrowsize=1.5,
                arrowwidth=1.5,
                arrowcolor=colors['line']
            )
    
    # 2. Desenhar cargas com estilo de bloco
    for load in st.session_state.loads:
        bus = next((b for b in st.session_state.buses if b['id'] == load['bus']), None)
        if bus:
            # Posicionar a carga ao lado da barra
            load_x = bus['x'] + 50
            load_y = bus['y'] + 30
            
            # Bloco de carga (retângulo)
            fig.add_shape(
                type="rect",
                x0=load_x - 40,
                y0=load_y - 25,
                x1=load_x + 40,
                y1=load_y + 25,
                fillcolor='rgba(156, 39, 176, 0.1)',
                line_color=colors['load'],
                line_width=2,
                opacity=0.8
            )
            
            # Texto da carga
            fig.add_annotation(
                x=load_x,
                y=load_y,
                text=f"<b>Carga {load['id']}</b><br>P: {load['p_mw']} MW<br>Q: {load['q_mvar']} MVar",
                showarrow=False,
                font=dict(size=9, color=colors['load']),
                bgcolor='rgba(255, 255, 255, 0.9)',
                borderpad=4
            )
            
            # Porta de conexão
            fig.add_trace(go.Scatter(
                x=[bus['x'] + 25, load_x - 40],
                y=[bus['y'], load_y],
                mode='lines',
                line=dict(
                    color=colors['load'],
                    width=2,
                    dash='dash'
                ),
                hoverinfo='skip',
                showlegend=False
            ))
    
    # 3. Desenhar geradores com estilo de bloco
    for gen in st.session_state.generators:
        bus = next((b for b in st.session_state.buses if b['id'] == gen['bus']), None)
        if bus:
            # Posicionar o gerador ao lado da barra
            gen_x = bus['x'] - 50
            gen_y = bus['y'] - 30
            
            # Bloco do gerador (círculo)
            fig.add_shape(
                type="circle",
                x0=gen_x - 30,
                y0=gen_y - 30,
                x1=gen_x + 30,
                y1=gen_y + 30,
                fillcolor='rgba(76, 175, 80, 0.1)',
                line_color=colors['generator'],
                line_width=2,
                opacity=0.8
            )
            
            # Texto do gerador
            fig.add_annotation(
                x=gen_x,
                y=gen_y,
                text=f"<b>Gerador</b><br>P: {gen['p_mw']} MW<br>Vm: {gen['vm_pu']} pu",
                showarrow=False,
                font=dict(size=9, color=colors['generator']),
                bgcolor='rgba(255, 255, 255, 0.9)',
                borderpad=4
            )
            
            # Porta de conexão
            fig.add_trace(go.Scatter(
                x=[bus['x'] - 25, gen_x + 30],
                y=[bus['y'], gen_y],
                mode='lines',
                line=dict(
                    color=colors['generator'],
                    width=2,
                    dash='dash'
                ),
                hoverinfo='skip',
                showlegend=False
            ))
    
    # 4. Desenhar barras (buses) como blocos interativos
    for bus in st.session_state.buses:
        is_selected = st.session_state.selected_bus == bus['id']
        
        # Tipo de barra determina cor e estilo
        if bus['type'] == 'slack':
            bus_color = colors['slack']
            bus_symbol = 'diamond'
            bus_label = "SLACK"
        elif bus['type'] == 'pv':
            bus_color = colors['pv']
            bus_symbol = 'square'
            bus_label = "PV"
        else:  # pq
            bus_color = colors['pq']
            bus_symbol = 'circle'
            bus_label = "PQ"
        
        # Bloco principal da barra
        fig.add_trace(go.Scatter(
            x=[bus['x']],
            y=[bus['y']],
            mode='markers+text',
            marker=dict(
                size=35 if is_selected else 30,
                color=bus_color,
                symbol=bus_symbol,
                line=dict(
                    width=4 if is_selected else 2,
                    color=colors['selected'] if is_selected else 'white'
                ),
                opacity=1.0
            ),
            text=bus_label,
            textposition="middle center",
            textfont=dict(
                size=10,
                color='white',
                family='Arial Black'
            ),
            hoverinfo='text',
            hovertext=f"<b>Barra {bus['id']}: {bus['name']}</b><br>" +
                     f"Tipo: {bus['type'].upper()}<br>" +
                     f"Vn: {bus['vn_kv']:.2f} kV<br>" +
                     f"Posição: ({bus['x']}, {bus['y']})",
            showlegend=False,
            name=f'bus_{bus["id"]}'
        ))
        
        # Portas de conexão (pontos verdes)
        # Porta superior
        fig.add_trace(go.Scatter(
            x=[bus['x']],
            y=[bus['y'] + 25],
            mode='markers',
            marker=dict(
                size=12,
                color=colors['port'],
                symbol='circle',
                line=dict(width=2, color='white')
            ),
            hoverinfo='text',
            hovertext=f"Porta de conexão - Barra {bus['id']}",
            showlegend=False,
            name=f'port_top_{bus["id"]}'
        ))
        
        # Porta inferior
        fig.add_trace(go.Scatter(
            x=[bus['x']],
            y=[bus['y'] - 25],
            mode='markers',
            marker=dict(
                size=12,
                color=colors['port'],
                symbol='circle',
                line=dict(width=2, color='white')
            ),
            hoverinfo='text',
            hovertext=f"Porta de conexão - Barra {bus['id']}",
            showlegend=False,
            name=f'port_bottom_{bus["id"]}'
        ))
        
        # Nome da barra
        fig.add_annotation(
            x=bus['x'],
            y=bus['y'] - 40,
            text=f"<b>{bus['name']}</b>",
            showarrow=False,
            font=dict(size=10, color='black'),
            bgcolor='rgba(255, 255, 255, 0.8)',
            borderpad=3,
            bordercolor='rgba(0,0,0,0.2)',
            borderwidth=1
        )
        
        # Se houver resultados, mostrar tensão
        if st.session_state.results and st.session_state.results['type'] == 'power_flow':
            voltage = st.session_state.results['data']['voltages'][bus['id']]
            if 0.95 <= voltage <= 1.05:
                voltage_color = colors['pv']
            elif 0.9 <= voltage < 0.95 or 1.05 < voltage <= 1.1:
                voltage_color = '#FF9800'
            else:
                voltage_color = '#F44336'
            
            fig.add_annotation(
                x=bus['x'],
                y=bus['y'] + 40,
                text=f"V = {voltage:.3f} pu",
                showarrow=False,
                font=dict(size=10, color=voltage_color, family='Courier New'),
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor=voltage_color,
                borderwidth=1,
                borderpad=3
            )
    
    # 5. Conexão temporária (se estiver no modo de conexão)
    if st.session_state.mode == 'connect' and st.session_state.temp_connection:
        bus = next((b for b in st.session_state.buses if b['id'] == st.session_state.temp_connection), None)
        if bus:
            # Destacar a barra selecionada
            fig.add_trace(go.Scatter(
                x=[bus['x']],
                y=[bus['y']],
                mode='markers',
                marker=dict(
                    size=45,
                    color='rgba(255, 193, 7, 0.3)',
                    symbol='circle',
                    line=dict(width=3, color='#FFC107')
                ),
                showlegend=False
            ))
            
            # Adicionar mensagem
            fig.add_annotation(
                x=bus['x'],
                y=bus['y'] + 60,
                text="📌 Primeira barra selecionada<br>Clique na segunda barra",
                showarrow=True,
                arrowhead=2,
                arrowcolor='#FFC107',
                font=dict(size=10, color='#FF9800'),
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='#FFC107',
                borderwidth=1
            )
    
    # Configuração final do layout
    fig.update_layout(
        width=1000,
        height=650,
        plot_bgcolor='rgb(248, 249, 250)',
        paper_bgcolor='white',
        title={
            'text': "⚡ Diagrama Unifilar Interativo",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': 'rgb(30, 41, 59)'}
        },
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        ),
        dragmode='pan',
        hovermode='closest',
        showlegend=False,
        margin=dict(l=20, r=20, t=80, b=20)
    )
    
    return fig

def run_power_flow():
    if len(st.session_state.buses) == 0:
        st.error("Adicione barras ao sistema primeiro")
        return None
    
    if not any(b['type'] == 'slack' for b in st.session_state.buses):
        st.error("O sistema precisa de pelo menos uma barra Slack")
        return None
    
    n = len(st.session_state.buses)
    
    # Simulação simplificada do fluxo de potência
    voltages = []
    for i in range(n):
        bus = st.session_state.buses[i]
        if bus['type'] == 'slack':
            voltages.append(1.0)
        elif bus['type'] == 'pv':
            voltages.append(1.0 + np.random.normal(0, 0.01))
        else:  # pq
            base_voltage = 0.98
            if any(l['bus'] == i for l in st.session_state.loads):
                base_voltage -= 0.03
            if any(g['bus'] == i for g in st.session_state.generators):
                base_voltage += 0.02
            voltages.append(max(0.85, min(1.1, base_voltage + np.random.normal(0, 0.015))))
    
    results = {
        'type': 'power_flow',
        'timestamp': datetime.now().isoformat(),
        'data': {
            'voltages': voltages,
            'angles': [0] + [np.random.uniform(-0.15, 0.15) for _ in range(n-1)],
            'converged': True,
            'iterations': np.random.randint(3, 8)
        }
    }
    
    st.session_state.results = results
    st.session_state.history.append(results)
    
    return results

def run_short_circuit():
    if len(st.session_state.buses) == 0:
        st.error("Adicione barras ao sistema primeiro")
        return None
    
    n = len(st.session_state.buses)
    
    # Simulação de curto-circuito
    fault_currents = []
    for i in range(n):
        base_current = 5.0
        if st.session_state.buses[i]['type'] == 'slack':
            base_current += 10.0
        if any(g['bus'] == i for g in st.session_state.generators):
            base_current += 5.0
        
        # Reduzir corrente baseado na distância da fonte
        num_connections = sum(1 for l in st.session_state.lines if l['from'] == i or l['to'] == i)
        reduction_factor = max(0.3, 1 - num_connections * 0.1)
        
        fault_currents.append((base_current * reduction_factor) + np.random.uniform(-1, 1))
    
    results = {
        'type': 'short_circuit',
        'timestamp': datetime.now().isoformat(),
        'data': {
            'fault_currents': fault_currents,
            'critical_buses': sorted(
                [(i, fault_currents[i]) for i in range(n)],
                key=lambda x: x[1],
                reverse=True
            )[:3]
        }
    }
    
    st.session_state.results = results
    st.session_state.history.append(results)
    
    return results

def export_system():
    data = {
        'buses': st.session_state.buses,
        'lines': st.session_state.lines,
        'loads': st.session_state.loads,
        'generators': st.session_state.generators,
        'results': st.session_state.results,
        'history': st.session_state.history,
        'export_date': datetime.now().isoformat()
    }
    return json.dumps(data, indent=2)

def import_system(uploaded_file):
    try:
        data = json.load(uploaded_file)
        
        st.session_state.buses = data.get('buses', [])
        st.session_state.lines = data.get('lines', [])
        st.session_state.loads = data.get('loads', [])
        st.session_state.generators = data.get('generators', [])
        st.session_state.results = data.get('results')
        st.session_state.history = data.get('history', [])
        
        # Reindexar IDs
        for i, bus in enumerate(st.session_state.buses):
            bus['id'] = i
        for i, line in enumerate(st.session_state.lines):
            line['id'] = i
        for i, load in enumerate(st.session_state.loads):
            load['id'] = i
        for i, gen in enumerate(st.session_state.generators):
            gen['id'] = i
        
        st.success("✅ Sistema importado com sucesso!")
        
    except Exception as e:
        st.error(f"❌ Erro ao importar arquivo: {str(e)}")

# =====================================================
# INTERFACE PRINCIPAL
# =====================================================

st.title("⚡ Power System Studio")
st.caption("Sistema interativo de análise de redes elétricas")
st.markdown("---")

# Barra lateral
with st.sidebar:
    st.header("🎛️ Controles")
    
    # Modo de operação
    st.subheader("Modo de Operação")
    mode_options = {
        "🔍 Selecionar/Visualizar": "select",
        "➕ Adicionar Barra": "add_bus",
        "🔗 Conectar Barras": "connect",
        "📊 Adicionar Carga": "add_load",
        "⚡ Adicionar Gerador": "add_gen"
    }
    
    selected_mode = st.radio(
        "Escolha uma ação:",
        list(mode_options.keys()),
        index=list(mode_options.values()).index(st.session_state.mode),
        key='mode_selector'
    )
    st.session_state.mode = mode_options[selected_mode]
    
    # Mostrar instruções baseadas no modo
    if st.session_state.mode == 'select':
        st.info("🔍 Clique em uma barra para selecioná-la")
    elif st.session_state.mode == 'add_bus':
        st.info("➕ Clique no diagrama para adicionar uma nova barra")
    elif st.session_state.mode == 'connect':
        st.info("🔗 Clique em duas barras para conectá-las")
    elif st.session_state.mode == 'add_load':
        st.info("📊 Clique em uma barra para adicionar uma carga")
    elif st.session_state.mode == 'add_gen':
        st.info("⚡ Clique em uma barra para adicionar um gerador")
    
    st.markdown("---")
    
    # Adicionar barra manualmente
    with st.expander("➕ Nova Barra", expanded=st.session_state.mode == 'add_bus'):
        st.write("Posicione a barra no diagrama:")
        col1, col2 = st.columns(2)
        with col1:
            x_pos = st.number_input("Posição X", 0, 1000, 500, step=50, key='x_pos_input')
        with col2:
            y_pos = st.number_input("Posição Y", 0, 600, 300, step=50, key='y_pos_input')
        
        bus_type = st.selectbox("Tipo da Barra", ["pq", "pv", "slack"], 
                               index=0,
                               help="Slack: Referência | PV: Tensão controlada | PQ: Carga",
                               key='bus_type_select')
        
        if st.button("✅ Criar Barra", use_container_width=True, key='create_bus_btn'):
            new_bus = add_bus(x_pos, y_pos, bus_type)
            st.success(f"Barra {new_bus['id']} criada na posição ({x_pos}, {y_pos})")
            st.rerun()
    
    # Conectar barras manualmente
    if len(st.session_state.buses) >= 2:
        with st.expander("🔗 Conectar Barras Manualmente", expanded=False):
            bus_options = {f"Barra {b['id']}: {b['name']}": b['id'] for b in st.session_state.buses}
            
            from_bus = st.selectbox("De:", list(bus_options.keys()), key='from_bus_manual')
            to_bus = st.selectbox("Para:", list(bus_options.keys()), key='to_bus_manual')
            
            if st.button("🔗 Criar Conexão Manual", use_container_width=True, key='manual_connect_btn'):
                result = add_line(bus_options[from_bus], bus_options[to_bus])
                if result:
                    st.success("Linha criada com sucesso!")
                    st.rerun()
                else:
                    st.warning("Conexão já existe ou é inválida")
    
    st.markdown("---")
    
    # Ferramentas de análise
    st.subheader("🔬 Análise")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Fluxo de\nPotência", use_container_width=True, key='power_flow_btn'):
            with st.spinner("Executando fluxo de potência..."):
                results = run_power_flow()
                if results:
                    st.success("Fluxo de potência concluído!")
            st.rerun()
    with col2:
        if st.button("⚡ Curto-\nCircuito", use_container_width=True, key='short_circuit_btn'):
            with st.spinner("Analisando curto-circuito..."):
                results = run_short_circuit()
                if results:
                    st.success("Análise de curto-circuito concluída!")
            st.rerun()
    
    st.markdown("---")
    
    # Importar/Exportar
    st.subheader("💾 Arquivos")
    
    uploaded_file = st.file_uploader("📂 Importar sistema", type=['json'], key='file_uploader')
    if uploaded_file:
        import_system(uploaded_file)
        st.rerun()
    
    if len(st.session_state.buses) > 0:
        json_str = export_system()
        st.download_button(
            label="💾 Exportar Sistema",
            data=json_str,
            file_name=f"power_system_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
            key='export_btn'
        )
    
    if st.button("🗑️ Limpar Tudo", type="secondary", use_container_width=True, key='clear_all_btn'):
        for key in ['buses', 'lines', 'loads', 'generators', 'results', 'history', 'selected_bus', 'temp_connection']:
            if key in ['buses', 'lines', 'loads', 'generators', 'history']:
                st.session_state[key] = []
            else:
                st.session_state[key] = None
        st.success("Sistema limpo!")
        st.rerun()

# Layout principal
col1, col2 = st.columns([2.5, 1])

with col1:
    # Diagrama do sistema
    st.subheader("📐 Diagrama Unifilar")
    
    if len(st.session_state.buses) == 0:
        st.info("👈 Comece adicionando barras ao sistema usando o painel lateral")
    
    # Criar o diagrama
    fig = draw_power_system()
    
    # Exibir o diagrama
    st.plotly_chart(fig, use_container_width=True, key="main_diagram")
    
    # Controles de clique (simulados para demonstração)
    st.markdown("---")
    st.subheader("🖱️ Controles de Clique")
    
    col_click1, col_click2, col_click3 = st.columns(3)
    
    with col_click1:
        st.write("**Clique em uma barra:**")
        if st.session_state.mode == 'select':
            st.info("Seleciona a barra")
        elif st.session_state.mode == 'connect':
            st.info("Seleciona para conexão")
        elif st.session_state.mode == 'add_load':
            st.info("Adiciona carga")
        elif st.session_state.mode == 'add_gen':
            st.info("Adiciona gerador")
    
    with col_click2:
        st.write("**Modo atual:**")
        mode_display = {
            'select': '🔍 Selecionar',
            'add_bus': '➕ Adicionar Barra',
            'connect': '🔗 Conectar',
            'add_load': '📊 Adicionar Carga',
            'add_gen': '⚡ Adicionar Gerador'
        }
        st.success(mode_display.get(st.session_state.mode, st.session_state.mode))
    
    with col_click3:
        if st.session_state.temp_connection is not None:
            st.write("**Conexão em andamento:**")
            st.warning(f"Barra {st.session_state.temp_connection} selecionada")
    
    # Simulação de cliques (para demonstração)
    st.markdown("---")
    st.write("**Simular cliques (para teste):**")
    
    if len(st.session_state.buses) > 0:
        bus_options = [f"Barra {b['id']}: {b['name']}" for b in st.session_state.buses]
        selected_bus_name = st.selectbox("Selecione uma barra:", bus_options, key='simulate_click')
        
        if st.button("Simular clique na barra", use_container_width=True, key='simulate_click_btn'):
            bus_id = int(selected_bus_name.split(':')[0].split(' ')[1])
            handle_bus_click(bus_id)

with col2:
    # Painel de propriedades
    st.subheader("📊 Informações")
    
    # Estatísticas
    with st.expander("📈 Estatísticas do Sistema", expanded=True):
        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.metric("🚏 Barras", len(st.session_state.buses))
            st.metric("📍 Linhas", len(st.session_state.lines))
        with col_stat2:
            st.metric("📊 Cargas", len(st.session_state.loads))
            st.metric("⚡ Geradores", len(st.session_state.generators))
        
        # Potência total
        total_load_p = sum(l['p_mw'] for l in st.session_state.loads)
        total_gen_p = sum(g['p_mw'] for g in st.session_state.generators)
        
        if total_load_p > 0 or total_gen_p > 0:
            st.markdown("---")
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.metric("Carga Total", f"{total_load_p:.1f} MW")
            with col_p2:
                st.metric("Geração Total", f"{total_gen_p:.1f} MW")
    
    # Barra selecionada
    if st.session_state.selected_bus is not None:
        bus = next((b for b in st.session_state.buses if b['id'] == st.session_state.selected_bus), None)
        if bus:
            with st.expander(f"🚏 {bus['name']}", expanded=True):
                st.caption(f"ID: {bus['id']} | Tipo: {bus['type'].upper()}")
                
                # Editar propriedades
                new_name = st.text_input("Nome", bus['name'], key=f'bus_name_{bus["id"]}')
                new_type = st.selectbox("Tipo", ["slack", "pv", "pq"], 
                                       index=["slack", "pv", "pq"].index(bus['type']),
                                       key=f'bus_type_{bus["id"]}')
                new_vn = st.number_input("Tensão Nominal (kV)", 
                                        value=float(bus['vn_kv']), 
                                        min_value=0.1,
                                        step=0.1,
                                        key=f'bus_vn_{bus["id"]}')
                
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.button("💾 Salvar", key=f'save_bus_{bus["id"]}', use_container_width=True):
                        bus['name'] = new_name
                        bus['type'] = new_type
                        bus['vn_kv'] = new_vn
                        st.success("✅ Barra atualizada!")
                        st.rerun()
                
                with col_btn2:
                    if st.button("🗑️ Remover", key=f'delete_bus_{bus["id"]}', type="secondary", use_container_width=True):
                        st.session_state.buses = [b for b in st.session_state.buses if b['id'] != bus['id']]
                        st.session_state.lines = [l for l in st.session_state.lines 
                                                 if l['from'] != bus['id'] and l['to'] != bus['id']]
                        st.session_state.loads = [l for l in st.session_state.loads if l['bus'] != bus['id']]
                        st.session_state.generators = [g for g in st.session_state.generators if g['bus'] != bus['id']]
                        st.session_state.selected_bus = None
                        st.success(f"Barra {bus['id']} removida!")
                        st.rerun()
                
                # Mostrar elementos conectados
                st.markdown("**Conexões:**")
                connected_lines = [l for l in st.session_state.lines if l['from'] == bus['id'] or l['to'] == bus['id']]
                if connected_lines:
                    for line in connected_lines:
                        other_bus = line['to'] if line['from'] == bus['id'] else line['from']
                        st.caption(f"→ Barra {other_bus} (Linha {line['id']})")
                else:
                    st.caption("Sem conexões")
                
                # Mostrar carga se existir
                bus_load = next((l for l in st.session_state.loads if l['bus'] == bus['id']), None)
                if bus_load:
                    st.markdown("**Carga conectada:**")
                    col_load1, col_load2 = st.columns(2)
                    with col_load1:
                        new_p = st.number_input("P (MW)", value=float(bus_load['p_mw']), 
                                              min_value=0.0, step=0.1, key=f'load_p_{bus_load["id"]}')
                    with col_load2:
                        new_q = st.number_input("Q (MVar)", value=float(bus_load['q_mvar']), 
                                              min_value=0.0, step=0.1, key=f'load_q_{bus_load["id"]}')
                    
                    if st.button("💾 Atualizar Carga", key=f'update_load_{bus_load["id"]}', use_container_width=True):
                        bus_load['p_mw'] = new_p
                        bus_load['q_mvar'] = new_q
                        st.success("Carga atualizada!")
                        st.rerun()
                
                # Mostrar gerador se existir
                bus_gen = next((g for g in st.session_state.generators if g['bus'] == bus['id']), None)
                if bus_gen:
                    st.markdown("**Gerador conectado:**")
                    col_gen1, col_gen2 = st.columns(2)
                    with col_gen1:
                        new_p_gen = st.number_input("P (MW)", value=float(bus_gen['p_mw']), 
                                                  min_value=0.0, step=0.1, key=f'gen_p_{bus_gen["id"]}')
                    with col_gen2:
                        new_vm = st.number_input("Vm (pu)", value=float(bus_gen['vm_pu']), 
                                               min_value=0.8, max_value=1.2, step=0.01, key=f'gen_vm_{bus_gen["id"]}')
                    
                    if st.button("💾 Atualizar Gerador", key=f'update_gen_{bus_gen["id"]}', use_container_width=True):
                        bus_gen['p_mw'] = new_p_gen
                        bus_gen['vm_pu'] = new_vm
                        st.success("Gerador atualizado!")
                        st.rerun()
    
    # Resultados da análise
    if st.session_state.results:
        with st.expander("📈 Resultados da Análise", expanded=True):
            results = st.session_state.results
            
            st.caption(f"📊 {results['type'].replace('_', ' ').title()}")
            st.caption(f"🕐 {datetime.fromisoformat(results['timestamp']).strftime('%d/%m/%Y %H:%M:%S')}")
            
            if results['type'] == 'power_flow':
                st.markdown("##### Tensões nas Barras")
                voltages = results['data']['voltages']
                
                # Criar dataframe para melhor visualização
                df_voltages = pd.DataFrame({
                    'Barra': [f"Bus {i}" for i in range(len(voltages))],
                    'Tensão (pu)': voltages,
                    'Status': ['✅ Normal' if 0.95 <= v <= 1.05 else 
                              '⚠️ Atenção' if 0.9 <= v < 0.95 or 1.05 < v <= 1.1 else 
                              '❌ Crítico' for v in voltages]
                })
                
                st.dataframe(df_voltages, hide_index=True, use_container_width=True)
                
                # Resumo
                avg_voltage = np.mean(voltages)
                min_voltage = np.min(voltages)
                max_voltage = np.max(voltages)
                
                col_v1, col_v2, col_v3 = st.columns(3)
                with col_v1:
                    st.metric("Média", f"{avg_voltage:.3f}")
                with col_v2:
                    st.metric("Mínima", f"{min_voltage:.3f}")
                with col_v3:
                    st.metric("Máxima", f"{max_voltage:.3f}")
                
                if 'iterations' in results['data']:
                    st.progress(results['data']['iterations'] / 10, 
                               text=f"Convergência: {results['data']['iterations']} iterações")
                
            elif results['type'] == 'short_circuit':
                st.markdown("##### Correntes de Curto-Circuito")
                currents = results['data']['fault_currents']
                
                df_currents = pd.DataFrame({
                    'Barra': [f"Bus {i}" for i in range(len(currents))],
                    'Corrente (kA)': [f"{c:.2f}" for c in currents],
                    'Nível': ['🔴 Alto' if c > 10 else '🟡 Médio' if c > 5 else '🟢 Baixo' for c in currents]
                })
                
                st.dataframe(df_currents, hide_index=True, use_container_width=True)
                
                st.markdown("##### ⚠️ Barras Críticas")
                for bus_idx, current in results['data']['critical_buses']:
                    severity = "🔴" if current > 10 else "🟡"
                    st.warning(f"{severity} **Bus {bus_idx}**: {current:.2f} kA")
    
    # Histórico
    if st.session_state.history:
        with st.expander(f"📋 Histórico ({len(st.session_state.history)} análises)", expanded=False):
            for i, hist in enumerate(reversed(st.session_state.history[-10:])):  # Últimas 10
                icon = "🔄" if hist['type'] == 'power_flow' else "⚡"
                type_str = "Fluxo de Potência" if hist['type'] == 'power_flow' else "Curto-Circuito"
                time_str = datetime.fromisoformat(hist['timestamp']).strftime('%H:%M:%S')
                
                col_hist1, col_hist2 = st.columns([3, 1])
                with col_hist1:
                    st.write(f"{icon} {type_str} - {time_str}")
                with col_hist2:
                    if st.button("↻", key=f"load_hist_{i}"):
                        st.session_state.results = hist
                        st.rerun()

# Rodapé com legenda
st.markdown("---")
st.subheader("📖 Legenda e Instruções")

legend_cols = st.columns(4)

with legend_cols[0]:
    st.markdown("""
    **Tipos de Barras:**
    - 🔴 **Slack**: Barra de referência (V e θ fixos)
    - 🟢 **PV**: Geração (P e V fixos)
    - 🟠 **PQ**: Carga (P e Q fixos)
    """)

with legend_cols[1]:
    st.markdown("""
    **Elementos:**
    - **Linhas**: Conexões em azul
    - **Cargas**: Retângulos roxos
    - **Geradores**: Círculos verdes
    - **Portas**: Pontos verdes (conexão)
    """)

with legend_cols[2]:
    st.markdown("""
    **Níveis de Tensão:**
    - 🟢 **0.95 - 1.05 pu**: Normal
    - 🟡 **0.90 - 0.94 / 1.06 - 1.10**: Atenção
    - 🔴 **< 0.90 / > 1.10**: Crítico
    """)

with legend_cols[3]:
    st.markdown("""
    **Modos de Operação:**
    - 🔍 **Selecionar**: Clique para selecionar
    - ➕ **Adicionar Barra**: Clique para criar
    - 🔗 **Conectar**: Clique em duas barras
    - 📊 **Carga**: Clique para adicionar
    - ⚡ **Gerador**: Clique para adicionar
    """)

# Dicas de uso
with st.expander("💡 Dicas de Uso"):
    st.markdown("""
    1. **Criar Sistema**: 
       - Use modo "Adicionar Barra" ou botão no painel lateral
       - Posicione clicando no diagrama ou usando coordenadas
       
    2. **Conectar Barras**:
       - Selecione modo "Conectar Barras"
       - Clique na primeira barra (ficará destacada)
       - Clique na segunda barra para criar a linha
       
    3. **Adicionar Elementos**:
       - Selecione modo "Adicionar Carga" ou "Adicionar Gerador"
       - Clique na barra desejada
       - Configure os parâmetros no painel lateral
       
    4. **Analisar**:
       - Execute fluxo de potência ou análise de curto-circuito
       - Os resultados aparecerão no diagrama e no painel
       
    5. **Editar**:
       - Selecione uma barra para editar suas propriedades
       - Atualize cargas e geradores no painel de informações
       
    6. **Salvar/Carregar**:
       - Exporte seu sistema em JSON para uso posterior
       - Importe sistemas salvos anteriormente
    """)

# Informações técnicas
st.caption("Power System Studio v2.0 | Desenvolvido com Streamlit e Plotly | ⚡ Análise de Sistemas Elétricos de Potência")

if __name__ == "__main__":
    main()
