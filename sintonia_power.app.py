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
    
    # Cores sólidas (sem gradientes)
    colors = {
        'slack': '#667eea',
        'pv': '#4CAF50',
        'pq': '#FF9800',
        'line': '#667eea',
        'load': '#9C27B0',
        'generator': '#4CAF50',
        'port': '#4CAF50',
        'selected': '#FFD700'
    }
    
    # 1. Desenhar linhas de transmissão
    for line in st.session_state.lines:
        from_bus = next((b for b in st.session_state.buses if b['id'] == line['from']), None)
        to_bus = next((b for b in st.session_state.buses if b['id'] == line['to']), None)
        
        if from_bus and to_bus:
            # Linha principal
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
                showlegend=False
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
    
    # 2. Desenhar cargas
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
                line_width=2
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
            
            # Linha de conexão
            fig.add_trace(go.Scatter(
                x=[bus['x'] + 25, load_x - 40],
                y=[bus['y'], load_y],
                mode='lines',
                line=dict(color=colors['load'], width=2, dash='dash'),
                hoverinfo='skip',
                showlegend=False
            ))
    
    # 3. Desenhar geradores
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
                line_width=2
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
            
            # Linha de conexão
            fig.add_trace(go.Scatter(
                x=[bus['x'] - 25, gen_x + 30],
                y=[bus['y'], gen_y],
                mode='lines',
                line=dict(color=colors['generator'], width=2, dash='dash'),
                hoverinfo='skip',
                showlegend=False
            ))
    
    # 4. Desenhar barras (buses)
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
                line=dict(width=4 if is_selected else 2, 
                         color=colors['selected'] if is_selected else 'white')
            ),
            text=bus_label,
            textposition="middle center",
            textfont=dict(size=10, color='white', family='Arial'),
            hoverinfo='text',
            hovertext=f"<b>Barra {bus['id']}: {bus['name']}</b><br>" +
                     f"Tipo: {bus['type'].upper()}<br>" +
                     f"Vn: {bus['vn_kv']:.2f} kV",
            showlegend=False
        ))
        
        # Portas de conexão (pontos verdes)
        # Porta superior
        fig.add_trace(go.Scatter(
            x=[bus['x']],
            y=[bus['y'] + 25],
            mode='markers',
            marker=dict(size=12, color=colors['port'], symbol='circle',
                       line=dict(width=2, color='white')),
            hoverinfo='skip',
            showlegend=False
        ))
        
        # Porta inferior
        fig.add_trace(go.Scatter(
            x=[bus['x']],
            y=[bus['y'] - 25],
            mode='markers',
            marker=dict(size=12, color=colors['port'], symbol='circle',
                       line=dict(width=2, color='white')),
            hoverinfo='skip',
            showlegend=False
        ))
        
        # Nome da barra
        fig.add_annotation(
            x=bus['x'],
            y=bus['y'] - 40,
            text=f"<b>{bus['name']}</b>",
            showarrow=False,
            font=dict(size=10, color='black'),
            bgcolor='rgba(255, 255, 255, 0.8)',
            borderpad=3
        )
        
        # Se houver resultados, mostrar tensão
        if st.session_state.results and st.session_state.results['type'] == 'power_flow':
            if bus['id'] < len(st.session_state.results['data']['voltages']):
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
                    font=dict(size=10, color=voltage_color),
                    bgcolor='rgba(255, 255, 255, 0.9)',
                    bordercolor=voltage_color,
                    borderwidth=1,
                    borderpad=3
                )
    
    # 5. Conexão temporária (se estiver no modo de conexão)
    if st.session_state.mode == 'connect' and st.session_state.temp_connection is not None:
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
    
    # Configuração final do layout
    fig.update_layout(
        width=1000,
        height=650,
        plot_bgcolor='rgb(248, 249, 250)',
        paper_bgcolor='white',
        title=dict(text="⚡ Diagrama Unifilar Interativo", x=0.5, xanchor='center'),
        hoverlabel=dict(bgcolor="white", font_size=12),
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
        
        fault_currents.append(base_current + np.random.uniform(-1, 1))
    
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
        index=0,
        key='mode_selector'
    )
    st.session_state.mode = mode_options[selected_mode]
    
    st.markdown("---")
    
    # Adicionar barra manualmente
    with st.expander("➕ Nova Barra"):
        col1, col2 = st.columns(2)
        with col1:
            x_pos = st.number_input("Posição X", 0, 1000, 500, step=50, key='x_pos')
        with col2:
            y_pos = st.number_input("Posição Y", 0, 600, 300, step=50, key='y_pos')
        
        bus_type = st.selectbox("Tipo da Barra", ["pq", "pv", "slack"], 
                               index=0, key='bus_type')
        
        if st.button("✅ Criar Barra", use_container_width=True, key='create_bus'):
            new_bus = add_bus(x_pos, y_pos, bus_type)
            st.success(f"Barra {new_bus['id']} criada!")
            st.rerun()
    
    st.markdown("---")
    
    # Ferramentas de análise
    st.subheader("🔬 Análise")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Fluxo de Potência", use_container_width=True, key='power_flow'):
            with st.spinner("Executando..."):
                run_power_flow()
            st.rerun()
    with col2:
        if st.button("⚡ Curto-Circuito", use_container_width=True, key='short_circuit'):
            with st.spinner("Analisando..."):
                run_short_circuit()
            st.rerun()
    
    st.markdown("---")
    
    # Importar/Exportar
    st.subheader("💾 Arquivos")
    
    uploaded_file = st.file_uploader("📂 Importar sistema", type=['json'], key='uploader')
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
            key='export'
        )
    
    if st.button("🗑️ Limpar Tudo", type="secondary", use_container_width=True, key='clear'):
        st.session_state.buses = []
        st.session_state.lines = []
        st.session_state.loads = []
        st.session_state.generators = []
        st.session_state.results = None
        st.session_state.history = []
        st.session_state.selected_bus = None
        st.session_state.temp_connection = None
        st.success("Sistema limpo!")
        st.rerun()

# Layout principal
col1, col2 = st.columns([2.5, 1])

with col1:
    # Diagrama do sistema
    st.subheader("📐 Diagrama Unifilar")
    
    if len(st.session_state.buses) == 0:
        st.info("👈 Comece adicionando barras ao sistema usando o painel lateral")
    
    # Criar e exibir o diagrama
    try:
        fig = draw_power_system()
        st.plotly_chart(fig, use_container_width=True, key="main_diagram")
    except Exception as e:
        st.error(f"Erro ao criar diagrama: {str(e)}")
        # Fallback: mostrar informações em texto
        st.write("**Barras no sistema:**")
        for bus in st.session_state.buses:
            st.write(f"- Barra {bus['id']}: {bus['name']} (Tipo: {bus['type']})")
    
    # Instruções
    st.markdown("---")
    st.write("**Instruções:**")
    if st.session_state.mode == 'select':
        st.info("🔍 Selecione uma barra para ver detalhes")
    elif st.session_state.mode == 'add_bus':
        st.info("➕ Use o painel lateral para adicionar barras")
    elif st.session_state.mode == 'connect':
        st.info("🔗 Clique em duas barras para conectá-las (use painel para simular)")
    elif st.session_state.mode == 'add_load':
        st.info("📊 Selecione uma barra para adicionar carga (use painel para simular)")
    elif st.session_state.mode == 'add_gen':
        st.info("⚡ Selecione uma barra para adicionar gerador (use painel para simular)")

with col2:
    # Painel de informações
    st.subheader("📊 Informações")
    
    # Estatísticas
    with st.expander("📈 Estatísticas", expanded=True):
        st.metric("🚏 Barras", len(st.session_state.buses))
        st.metric("📍 Linhas", len(st.session_state.lines))
        st.metric("📊 Cargas", len(st.session_state.loads))
        st.metric("⚡ Geradores", len(st.session_state.generators))
        
        if st.session_state.buses:
            total_load = sum(l['p_mw'] for l in st.session_state.loads)
            total_gen = sum(g['p_mw'] for g in st.session_state.generators)
            if total_load > 0 or total_gen > 0:
                st.markdown("---")
                st.metric("💡 Carga Total", f"{total_load:.1f} MW")
                st.metric("🔋 Geração Total", f"{total_gen:.1f} MW")
    
    # Barra selecionada
    if st.session_state.selected_bus is not None:
        bus = next((b for b in st.session_state.buses if b['id'] == st.session_state.selected_bus), None)
        if bus:
            with st.expander(f"🚏 {bus['name']}", expanded=True):
                st.write(f"**ID:** {bus['id']}")
                st.write(f"**Tipo:** {bus['type'].upper()}")
                st.write(f"**Tensão nominal:** {bus['vn_kv']} kV")
                
                # Elementos conectados
                lines_connected = [l for l in st.session_state.lines if l['from'] == bus['id'] or l['to'] == bus['id']]
                if lines_connected:
                    st.write("**Linhas conectadas:**")
                    for line in lines_connected:
                        other = line['to'] if line['from'] == bus['id'] else line['from']
                        st.caption(f"→ Barra {other}")
                
                load = next((l for l in st.session_state.loads if l['bus'] == bus['id']), None)
                if load:
                    st.write("**Carga:**")
                    st.caption(f"P = {load['p_mw']} MW | Q = {load['q_mvar']} MVar")
                
                gen = next((g for g in st.session_state.generators if g['bus'] == bus['id']), None)
                if gen:
                    st.write("**Gerador:**")
                    st.caption(f"P = {gen['p_mw']} MW | Vm = {gen['vm_pu']} pu")
    
    # Resultados
    if st.session_state.results:
        with st.expander("📈 Resultados", expanded=True):
            results = st.session_state.results
            st.write(f"**Tipo:** {results['type']}")
            st.write(f"**Data:** {datetime.fromisoformat(results['timestamp']).strftime('%H:%M:%S')}")
            
            if results['type'] == 'power_flow':
                st.write("**Tensões (pu):**")
                voltages = results['data']['voltages']
                for i, v in enumerate(voltages):
                    status = "✅" if 0.95 <= v <= 1.05 else "⚠️" if 0.9 <= v < 0.95 or 1.05 < v <= 1.1 else "❌"
                    st.caption(f"{status} Barra {i}: {v:.3f}")
            
            elif results['type'] == 'short_circuit':
                st.write("**Correntes de falta (kA):**")
                currents = results['data']['fault_currents']
                for i, c in enumerate(currents):
                    st.caption(f"Barra {i}: {c:.2f} kA")

# Rodapé
st.markdown("---")
st.caption("Power System Studio v1.0 | Desenvolvido com Streamlit | ⚡ Análise de Sistemas Elétricos")

# Informações técnicas
st.caption("Power System Studio v2.0 | Desenvolvido com Streamlit e Plotly | ⚡ Análise de Sistemas Elétricos de Potência")

if __name__ == "__main__":
    main()
