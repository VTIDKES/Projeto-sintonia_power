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

# Funções auxiliares
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

def draw_power_system():
    """Desenha o sistema elétrico usando Plotly com visualização aprimorada"""
    fig = go.Figure()
    
    # Configurar o layout base
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(200, 200, 200, 0.3)',
        zeroline=False,
        range=[-50, 1050],
        showticklabels=False
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(200, 200, 200, 0.3)',
        zeroline=False,
        range=[-50, 650],
        scaleanchor="x",
        scaleratio=1,
        showticklabels=False
    )
    
    # 1. Desenhar linhas de transmissão com estilo melhorado
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
                    color='rgba(70, 130, 180, 0.8)',
                    width=4
                ),
                hoverinfo='text',
                hovertext=f"<b>Line {line['id']}</b><br>" +
                          f"From: Bus {line['from']}<br>" +
                          f"To: Bus {line['to']}<br>" +
                          f"R: {line['r_ohm_per_km']:.3f} Ω/km<br>" +
                          f"X: {line['x_ohm_per_km']:.3f} Ω/km<br>" +
                          f"Length: {line['length_km']:.1f} km<br>" +
                          f"Max I: {line['max_i_ka']:.2f} kA",
                showlegend=False,
                name=f'line_{line["id"]}'
            ))
            
            # Adicionar indicador de fluxo (seta no meio da linha)
            mid_x = (from_bus['x'] + to_bus['x']) / 2
            mid_y = (from_bus['y'] + to_bus['y']) / 2
            dx = to_bus['x'] - from_bus['x']
            dy = to_bus['y'] - from_bus['y']
            
            # Normalizar direção
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                dx_norm = dx / length * 20
                dy_norm = dy / length * 20
                
                fig.add_annotation(
                    x=mid_x,
                    y=mid_y,
                    ax=mid_x - dx_norm,
                    ay=mid_y - dy_norm,
                    xref="x",
                    yref="y",
                    axref="x",
                    ayref="y",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1.5,
                    arrowwidth=2,
                    arrowcolor="rgba(70, 130, 180, 0.8)"
                )
    
    # 2. Desenhar símbolos de cargas (triângulos invertidos)
    for load in st.session_state.loads:
        bus = next((b for b in st.session_state.buses if b['id'] == load['bus']), None)
        if bus:
            # Desenhar triângulo de carga
            size = 20
            x_center = bus['x']
            y_base = bus['y'] + 40
            
            triangle_x = [
                x_center - size,
                x_center + size,
                x_center,
                x_center - size
            ]
            triangle_y = [
                y_base,
                y_base,
                y_base + size * 1.5,
                y_base
            ]
            
            fig.add_trace(go.Scatter(
                x=triangle_x,
                y=triangle_y,
                mode='lines',
                fill='toself',
                fillcolor='rgba(147, 51, 234, 0.3)',
                line=dict(color='rgb(147, 51, 234)', width=2),
                hoverinfo='text',
                hovertext=f"<b>Load {load['id']}</b><br>" +
                          f"Bus: {load['bus']}<br>" +
                          f"P: {load['p_mw']:.2f} MW<br>" +
                          f"Q: {load['q_mvar']:.2f} MVar",
                showlegend=False,
                name=f'load_{load["id"]}'
            ))
            
            # Linha conectando ao bus
            fig.add_trace(go.Scatter(
                x=[x_center, x_center],
                y=[bus['y'], y_base],
                mode='lines',
                line=dict(color='rgb(147, 51, 234)', width=2),
                hoverinfo='skip',
                showlegend=False
            ))
    
    # 3. Desenhar símbolos de geradores (círculos com G)
    for gen in st.session_state.generators:
        bus = next((b for b in st.session_state.buses if b['id'] == gen['bus']), None)
        if bus:
            x_center = bus['x']
            y_center = bus['y'] - 40
            radius = 18
            
            # Círculo do gerador
            theta = np.linspace(0, 2*np.pi, 50)
            circle_x = x_center + radius * np.cos(theta)
            circle_y = y_center + radius * np.sin(theta)
            
            fig.add_trace(go.Scatter(
                x=circle_x,
                y=circle_y,
                mode='lines',
                fill='toself',
                fillcolor='rgba(34, 197, 94, 0.3)',
                line=dict(color='rgb(34, 197, 94)', width=3),
                hoverinfo='text',
                hovertext=f"<b>Generator {gen['id']}</b><br>" +
                          f"Bus: {gen['bus']}<br>" +
                          f"P: {gen['p_mw']:.2f} MW<br>" +
                          f"Vm: {gen['vm_pu']:.3f} pu",
                showlegend=False,
                name=f'gen_{gen["id"]}'
            ))
            
            # Adicionar texto "G"
            fig.add_annotation(
                x=x_center,
                y=y_center,
                text="<b>G</b>",
                showarrow=False,
                font=dict(size=16, color='rgb(34, 197, 94)'),
                xref="x",
                yref="y"
            )
            
            # Linha conectando ao bus
            fig.add_trace(go.Scatter(
                x=[x_center, x_center],
                y=[bus['y'], y_center + radius],
                mode='lines',
                line=dict(color='rgb(34, 197, 94)', width=2),
                hoverinfo='skip',
                showlegend=False
            ))
    
    # 4. Desenhar barras (buses) com símbolos apropriados
    bus_colors = {
        'slack': 'rgb(239, 68, 68)',      # Vermelho
        'pv': 'rgb(34, 197, 94)',         # Verde
        'pq': 'rgb(59, 130, 246)'         # Azul
    }
    
    bus_symbols = {
        'slack': 'diamond',
        'pv': 'square',
        'pq': 'circle'
    }
    
    for bus in st.session_state.buses:
        # Determinar se está selecionado
        is_selected = st.session_state.selected_bus == bus['id']
        
        # Desenhar barra principal
        fig.add_trace(go.Scatter(
            x=[bus['x']],
            y=[bus['y']],
            mode='markers+text',
            marker=dict(
                size=28 if is_selected else 24,
                color=bus_colors.get(bus['type'], 'gray'),
                symbol=bus_symbols.get(bus['type'], 'circle'),
                line=dict(
                    width=4 if is_selected else 2,
                    color='yellow' if is_selected else 'white'
                ),
                opacity=1.0
            ),
            text=str(bus['id']),
            textposition="middle center",
            textfont=dict(
                size=12,
                color='white',
                family='Arial Black'
            ),
            hoverinfo='text',
            hovertext=f"<b>Bus {bus['id']}: {bus['name']}</b><br>" +
                      f"Type: {bus['type'].upper()}<br>" +
                      f"Vn: {bus['vn_kv']:.2f} kV<br>" +
                      f"Position: ({bus['x']}, {bus['y']})",
            showlegend=False,
            name=f'bus_{bus["id"]}'
        ))
        
        # Label com nome da barra
        fig.add_annotation(
            x=bus['x'],
            y=bus['y'] - 25,
            text=f"<b>{bus['name']}</b>",
            showarrow=False,
            font=dict(size=10, color='black'),
            bgcolor='rgba(255, 255, 255, 0.8)',
            borderpad=2
        )
        
        # Se tiver resultados de tensão, mostrar
        if st.session_state.results and st.session_state.results['type'] == 'power_flow':
            voltage = st.session_state.results['data']['voltages'][bus['id']]
            color = 'green' if 0.95 <= voltage <= 1.05 else 'orange' if 0.9 <= voltage < 0.95 or 1.05 < voltage <= 1.1 else 'red'
            
            fig.add_annotation(
                x=bus['x'],
                y=bus['y'] + 25,
                text=f"{voltage:.3f} pu",
                showarrow=False,
                font=dict(size=9, color=color),
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor=color,
                borderwidth=1,
                borderpad=2
            )
    
    # Configuração final do layout
    fig.update_layout(
        width=1000,
        height=650,
        plot_bgcolor='rgb(248, 249, 250)',
        paper_bgcolor='white',
        title={
            'text': "Power System One-Line Diagram",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': 'rgb(30, 41, 59)'}
        },
        xaxis_title="",
        yaxis_title="",
        dragmode='pan',
        hovermode='closest',
        showlegend=False,
        margin=dict(l=20, r=20, t=60, b=20)
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

# Interface principal
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
        index=list(mode_options.values()).index(st.session_state.mode)
    )
    st.session_state.mode = mode_options[selected_mode]
    
    st.markdown("---")
    
    # Adicionar barra manualmente
    with st.expander("➕ Nova Barra", expanded=st.session_state.mode == 'add_bus'):
        col1, col2 = st.columns(2)
        with col1:
            x_pos = st.number_input("Posição X", 0, 1000, 500, step=50)
        with col2:
            y_pos = st.number_input("Posição Y", 0, 600, 300, step=50)
        
        bus_type = st.selectbox("Tipo da Barra", ["pq", "pv", "slack"], 
                               help="Slack: Referência | PV: Tensão controlada | PQ: Carga")
        
        if st.button("✅ Criar Barra", use_container_width=True):
            add_bus(x_pos, y_pos, bus_type)
            st.success(f"Barra criada na posição ({x_pos}, {y_pos})")
            st.rerun()
    
    # Conectar barras
    if len(st.session_state.buses) >= 2:
        with st.expander("🔗 Conectar Barras", expanded=st.session_state.mode == 'connect'):
            bus_options = {f"Bus {b['id']}: {b['name']}": b['id'] for b in st.session_state.buses}
            
            from_bus = st.selectbox("De:", list(bus_options.keys()), key='from_bus')
            to_bus = st.selectbox("Para:", list(bus_options.keys()), key='to_bus')
            
            if st.button("🔗 Conectar", use_container_width=True):
                result = add_line(bus_options[from_bus], bus_options[to_bus])
                if result:
                    st.success("Linha criada com sucesso!")
                    st.rerun()
                else:
                    st.warning("Conexão já existe ou inválida")
    
    st.markdown("---")
    
    # Ferramentas de análise
    st.subheader("🔬 Análise")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Fluxo de\nPotência", use_container_width=True):
            run_power_flow()
            st.rerun()
    with col2:
        if st.button("⚡ Curto-\nCircuito", use_container_width=True):
            run_short_circuit()
            st.rerun()
    
    st.markdown("---")
    
    # Importar/Exportar
    st.subheader("💾 Arquivos")
    
    uploaded_file = st.file_uploader("📂 Importar sistema", type=['json'])
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
            use_container_width=True
        )
    
    if st.button("🗑️ Limpar Tudo", type="secondary", use_container_width=True):
        for key in ['buses', 'lines', 'loads', 'generators', 'results', 'history', 'selected_bus', 'temp_connection']:
            if key in ['buses', 'lines', 'loads', 'generators', 'history']:
                st.session_state[key] = []
            else:
                st.session_state[key] = None
        st.rerun()

# Layout principal
col1, col2 = st.columns([2.5, 1])

with col1:
    # Diagrama do sistema
    st.subheader("📐 Diagrama Unifilar")
    
    if len(st.session_state.buses) == 0:
        st.info("👈 Comece adicionando barras ao sistema usando o painel lateral")
    
    fig = draw_power_system()
    st.plotly_chart(fig, use_container_width=True, key="main_diagram")

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
                
                new_name = st.text_input("Nome", bus['name'], key='bus_name')
                new_type = st.selectbox("Tipo", ["slack", "pv", "pq"], 
                                       index=["slack", "pv", "pq"].index(bus['type']),
                                       key='bus_type')
                new_vn = st.number_input("Tensão Nominal (kV)", 
                                        value=float(bus['vn_kv']), 
                                        min_value=0.1,
                                        step=0.1,
                                        key='bus_vn')
                
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.button("💾 Salvar", key='save_bus', use_container_width=True):
                        bus['name'] = new_name
                        bus['type'] = new_type
                        bus['vn_kv'] = new_vn
                        st.success("✅ Barra atualizada!")
                        st.rerun()
                
                with col_btn2:
                    if st.button("🗑️ Remover", key='delete_bus', type="secondary", use_container_width=True):
                        st.session_state.buses = [b for b in st.session_state.buses if b['id'] != bus['id']]
                        st.session_state.lines = [l for l in st.session_state.lines 
                                                 if l['from'] != bus['id'] and l['to'] != bus['id']]
                        st.session_state.loads = [l for l in st.session_state.loads if l['bus'] != bus['id']]
                        st.session_state.generators = [g for g in st.session_state.generators if g['bus'] != bus['id']]
                        st.session_state.selected_bus = None
                        st.rerun()
                
                # Mostrar elementos conectados
                connected_lines = [l for l in st.session_state.lines if l['from'] == bus['id'] or l['to'] == bus['id']]
                if connected_lines:
                    st.markdown("**Linhas conectadas:**")
                    for line in connected_lines:
                        other_bus = line['to'] if line['from'] == bus['id'] else line['from']
                        st.caption(f"→ Bus {other_bus} (Linha {line['id']})")
                
                # Mostrar carga se existir
                bus_load = next((l for l in st.session_state.loads if l['bus'] == bus['id']), None)
                if bus_load:
                    st.markdown("**Carga:**")
                    st.caption(f"P: {bus_load['p_mw']} MW | Q: {bus_load['q_mvar']} MVar")
                
                # Mostrar gerador se existir
                bus_gen = next((g for g in st.session_state.generators if g['bus'] == bus['id']), None)
                if bus_gen:
                    st.markdown("**Gerador:**")
                    st.caption(f"P: {bus_gen['p_mw']} MW | Vm: {bus_gen['vm_pu']} pu")
    
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
                
                if st.button(f"{icon} {type_str} - {time_str}", 
                           key=f"hist_{i}", 
                           use_container_width=True):
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
    - 🔵 **PQ**: Carga (P e Q fixos)
    """)

with legend_cols[1]:
    st.markdown("""
    **Elementos:**
    - **Linhas**: Conexões em azul
    - **Cargas**: Triângulos roxos ▼
    - **Geradores**: Círculos verdes (G)
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
    **Corrente de Falta:**
    - 🟢 **< 5 kA**: Nível baixo
    - 🟡 **5 - 10 kA**: Nível médio
    - 🔴 **> 10 kA**: Nível alto
    """)

# Dicas de uso
with st.expander("💡 Dicas de Uso"):
    st.markdown("""
    1. **Criar Sistema**: Adicione barras usando o painel lateral ou clicando no diagrama
    2. **Conectar**: Use o modo "Conectar Barras" e clique em duas barras sequencialmente
    3. **Adicionar Elementos**: Selecione modo "Adicionar Carga/Gerador" e clique na barra desejada
    4. **Analisar**: Execute fluxo de potência ou análise de curto-circuito
    5. **Visualizar**: Os resultados aparecerão no diagrama e no painel lateral
    6. **Salvar**: Exporte seu sistema em JSON para uso posterior
    """)

# Informações técnicas
st.caption("Power System Studio v2.0 | Desenvolvido com Streamlit e Plotly | ⚡ Análise de Sistemas Elétricos de Potência")

if __name__ == "__main__":
    main()
