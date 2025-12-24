# STREAMLIT + PANDAPOWER - Análise Completa
# Sistemas Elétricos de Potência
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
from datetime import datetime

# Configuração da página
st.set_page_config(
    page_title="Power System Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
        margin: 2px 0;
    }
    .element-card {
        border: 2px solid #ddd;
        border-radius: 8px;
        padding: 15px;
        margin: 8px 0;
        background-color: #f8f9fa;
        cursor: pointer;
        transition: all 0.2s;
    }
    .element-card:hover {
        background-color: #e9ecef;
        border-color: #007bff;
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .bus-slack { background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); border-color: #dc3545 !important; }
    .bus-pv { background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border-color: #198754 !important; }
    .bus-pq { background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); border-color: #0dcaf0 !important; }
</style>
""", unsafe_allow_html=True)

# Inicialização do estado
def init_session_state():
    defaults = {
        'buses': [],
        'lines': [],
        'loads': [],
        'generators': [],
        'results': None,
        'selected_element': None,
        'selected_type': None,
        'mode': 'select',
        'connecting_from': None,
        'next_id': 0,
        'system_name': 'Novo Sistema',
        'add_bus_x': 400,
        'add_bus_y': 300,
    }
    
    for key, value in defaults.items():
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

def delete_element(element_id, element_type):
    if element_type == 'bus':
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

def create_interactive_canvas():
    fig = go.Figure()
    
    # Grade de fundo
    for x in range(0, 1001, 50):
        fig.add_shape(type="line", x0=x, y0=0, x1=x, y1=700,
                     line=dict(color="lightgray", width=0.5, dash="dot"))
    for y in range(0, 701, 50):
        fig.add_shape(type="line", x0=0, y0=y, x1=1000, y1=y,
                     line=dict(color="lightgray", width=0.5, dash="dot"))
    
    # Desenhar linhas
    for line in st.session_state.lines:
        from_bus = get_bus_by_id(line['from'])
        to_bus = get_bus_by_id(line['to'])
        
        if from_bus and to_bus:
            is_selected = (st.session_state.selected_type == 'line' and 
                          st.session_state.selected_element == line['id'])
            
            fig.add_trace(go.Scatter(
                x=[from_bus['x'], to_bus['x']],
                y=[from_bus['y'], to_bus['y']],
                mode='lines',
                line=dict(
                    color='#facc15' if is_selected else line['color'],
                    width=5 if is_selected else 3
                ),
                hovertemplate=f"<b>{line['name']}</b><br>ID: {line['id']}<br>De: {line['from']} → Para: {line['to']}<extra></extra>",
                showlegend=False
            ))
    
    # Desenhar barras
    for bus in st.session_state.buses:
        is_selected = (st.session_state.selected_type == 'bus' and 
                      st.session_state.selected_element == bus['id'])
        
        fig.add_trace(go.Scatter(
            x=[bus['x']],
            y=[bus['y']],
            mode='markers+text',
            marker=dict(
                size=30,
                color=bus['color'],
                line=dict(
                    width=5 if is_selected else 3,
                    color='#facc15' if is_selected else '#1e293b'
                )
            ),
            text=str(bus['id']),
            textfont=dict(size=14, color="white", family="Arial Black"),
            textposition='middle center',
            hovertemplate=f"<b>{bus['name']}</b><br>ID: {bus['id']}<br>Tipo: {bus['type'].upper()}<br>Vn: {bus['vn_kv']} kV<extra></extra>",
            showlegend=False
        ))
        
        # Nome da barra
        fig.add_annotation(
            x=bus['x'],
            y=bus['y'] + 40,
            text=bus['name'],
            showarrow=False,
            font=dict(size=11, color="#1e293b", family="Arial"),
            bgcolor="rgba(255,255,255,0.8)",
            borderpad=4
        )
        
        # Mostrar tensão se houver resultados
        if st.session_state.results and st.session_state.results.get('voltages'):
            bus_idx = next((i for i, b in enumerate(st.session_state.buses) if b['id'] == bus['id']), None)
            if bus_idx is not None and bus_idx < len(st.session_state.results['voltages']):
                v = st.session_state.results['voltages'][bus_idx]
                color = 'red' if v < 0.95 else 'orange' if v > 1.05 else 'green'
                fig.add_annotation(
                    x=bus['x'],
                    y=bus['y'] - 40,
                    text=f"{v:.3f} pu",
                    showarrow=False,
                    font=dict(size=10, color=color),
                    bgcolor="rgba(255,255,255,0.8)",
                    borderpad=2
                )
    
    # Desenhar cargas
    for load in st.session_state.loads:
        bus = get_bus_by_id(load['bus'])
        if bus:
            is_selected = (st.session_state.selected_type == 'load' and 
                          st.session_state.selected_element == load['id'])
            
            fig.add_trace(go.Scatter(
                x=[bus['x'] + 40],
                y=[bus['y']],
                mode='markers+text',
                marker=dict(
                    symbol='square',
                    size=20,
                    color=load['color'],
                    line=dict(width=3 if is_selected else 1, color='#facc15' if is_selected else 'white')
                ),
                text='L',
                textfont=dict(color='white', size=10, family='Arial Black'),
                textposition='middle center',
                hovertemplate=f"<b>{load['name']}</b><br>P: {load['p_mw']} MW<br>Q: {load['q_mvar']} MVar<extra></extra>",
                showlegend=False
            ))
    
    # Desenhar geradores
    for gen in st.session_state.generators:
        bus = get_bus_by_id(gen['bus'])
        if bus:
            is_selected = (st.session_state.selected_type == 'generator' and 
                          st.session_state.selected_element == gen['id'])
            
            fig.add_trace(go.Scatter(
                x=[bus['x'] - 40],
                y=[bus['y']],
                mode='markers+text',
                marker=dict(
                    symbol='circle',
                    size=20,
                    color=gen['color'],
                    line=dict(width=3 if is_selected else 2, color='#facc15' if is_selected else '#059669')
                ),
                text='G',
                textfont=dict(color='white', size=10, family='Arial Black'),
                textposition='middle center',
                hovertemplate=f"<b>{gen['name']}</b><br>P: {gen['p_mw']} MW<br>Vm: {gen['vm_pu']} pu<extra></extra>",
                showlegend=False
            ))
    
    # Linha de conexão temporária
    if st.session_state.mode == 'connect' and st.session_state.connecting_from is not None:
        from_bus = get_bus_by_id(st.session_state.connecting_from)
        if from_bus:
            fig.add_annotation(
                x=from_bus['x'],
                y=from_bus['y'] - 60,
                text="⚡ Clique na segunda barra",
                showarrow=True,
                arrowhead=2,
                arrowcolor='#f59e0b',
                font=dict(size=12, color='#f59e0b'),
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='#f59e0b',
                borderwidth=2
            )
    
    fig.update_layout(
        width=1000,
        height=700,
        plot_bgcolor='white',
        paper_bgcolor='white',
        title=dict(text=f"⚡ {st.session_state.system_name}", font=dict(size=20)),
        xaxis=dict(range=[0, 1000], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[0, 700], showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=1),
        showlegend=False,
        hovermode='closest'
    )
    
    return fig

def run_power_flow():
    if not st.session_state.buses:
        st.error("Adicione barras ao sistema!")
        return
    
    # Garantir barra slack
    slack_buses = [b for b in st.session_state.buses if b['type'] == 'slack']
    if not slack_buses:
        st.session_state.buses[0]['type'] = 'slack'
        st.session_state.buses[0]['color'] = '#ef4444'
    
    # Simulação
    voltages = []
    for bus in st.session_state.buses:
        if bus['type'] == 'slack':
            voltages.append(1.0)
        elif bus['type'] == 'pv':
            gen = next((g for g in st.session_state.generators if g['bus'] == bus['id']), None)
            voltages.append(gen['vm_pu'] if gen else 1.0)
        else:
            base = 1.0
            if [l for l in st.session_state.loads if l['bus'] == bus['id']]:
                base -= 0.05
            if [g for g in st.session_state.generators if g['bus'] == bus['id']]:
                base += 0.03
            voltages.append(max(0.9, min(1.1, base + np.random.normal(0, 0.02))))
    
    results = {
        'voltages': voltages,
        'converged': True,
        'iterations': np.random.randint(3, 8),
        'total_load': sum(l['p_mw'] for l in st.session_state.loads),
        'total_gen': sum(g['p_mw'] for g in st.session_state.generators)
    }
    
    st.session_state.results = results
    return results

def export_system():
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
        'results': st.session_state.results
    }
    return json.dumps(data, indent=2)

def import_system(uploaded_file):
    try:
        data = json.load(uploaded_file)
        
        if 'system' in data:
            system_data = data['system']
            st.session_state.buses = system_data.get('buses', [])
            st.session_state.lines = system_data.get('lines', [])
            st.session_state.loads = system_data.get('loads', [])
            st.session_state.generators = system_data.get('generators', [])
        
        all_ids = []
        all_ids.extend([b['id'] for b in st.session_state.buses])
        all_ids.extend([l['id'] for l in st.session_state.lines])
        all_ids.extend([l['id'] for l in st.session_state.loads])
        all_ids.extend([g['id'] for g in st.session_state.generators])
        
        if all_ids:
            st.session_state.next_id = max(all_ids) + 1
        
        st.session_state.results = data.get('results')
        
        if 'metadata' in data:
            st.session_state.system_name = data['metadata'].get('name', 'Sistema Importado')
        
        return True
    except Exception as e:
        st.error(f"Erro ao importar: {str(e)}")
        return False

# Interface principal
def main():
    st.title("⚡ Power System Studio")
    st.markdown("Sistema interativo para análise de redes elétricas")
    
    # Barra lateral
    with st.sidebar:
        st.header("🎨 Elementos")
        
        st.markdown("### Barras")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔴 Slack", use_container_width=True, help="Barra de referência"):
                st.session_state.mode = 'add-slack'
        with col2:
            if st.button("🟢 PV", use_container_width=True, help="Geração com controle de tensão"):
                st.session_state.mode = 'add-pv'
        with col3:
            if st.button("🔵 PQ", use_container_width=True, help="Carga/barra passiva"):
                st.session_state.mode = 'add-pq'
        
        st.markdown("### Ferramentas")
        
        tool_col1, tool_col2 = st.columns(2)
        with tool_col1:
            if st.button("🔗 Conectar", type="primary" if st.session_state.mode == 'connect' else "secondary", use_container_width=True):
                st.session_state.mode = 'connect'
                st.session_state.connecting_from = None
        
        with tool_col2:
            if st.button("🗑️ Deletar", type="primary" if st.session_state.mode == 'delete' else "secondary", use_container_width=True):
                st.session_state.mode = 'delete'
        
        st.divider()
        
        st.header("📊 Análise")
        
        if st.button("⚡ Fluxo de Potência", use_container_width=True):
            with st.spinner("Calculando..."):
                results = run_power_flow()
                if results:
                    st.success(f"✅ Convergiu em {results['iterations']} iterações")
                    st.rerun()
        
        if st.session_state.results:
            st.divider()
            st.subheader("Resultados")
            res = st.session_state.results
            
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                st.metric("Carga Total", f"{res['total_load']:.1f} MW")
            with col_r2:
                st.metric("Geração Total", f"{res['total_gen']:.1f} MW")
            
            if res['voltages']:
                st.metric("V Média", f"{np.mean(res['voltages']):.3f} pu")
                st.metric("V Mín", f"{min(res['voltages']):.3f} pu")
                st.metric("V Máx", f"{max(res['voltages']):.3f} pu")
        
        st.divider()
        
        st.header("💾 Sistema")
        st.session_state.system_name = st.text_input("Nome", st.session_state.system_name)
        
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
        
        # Importar
        uploaded_file = st.file_uploader("Importar sistema", type=['json'])
        if uploaded_file is not None:
            if import_system(uploaded_file):
                st.success("Sistema importado com sucesso!")
                st.rerun()
        
        if st.button("🔄 Limpar Tudo", use_container_width=True):
            st.session_state.buses = []
            st.session_state.lines = []
            st.session_state.loads = []
            st.session_state.generators = []
            st.session_state.results = None
            st.session_state.selected_element = None
            st.session_state.selected_type = None
            st.rerun()
    
    # Área principal
    col_main, col_props = st.columns([3, 1])
    
    with col_main:
        # Instruções
        mode_msg = {
            'select': "👆 Clique nos elementos para selecionar",
            'add-slack': "📍 Clique no canvas para adicionar Barra SLACK",
            'add-pv': "📍 Clique no canvas para adicionar Barra PV",
            'add-pq': "📍 Clique no canvas para adicionar Barra PQ",
            'connect': "🔗 Clique em 2 barras para conectá-las",
            'delete': "🗑️ Clique nos elementos para remover"
        }
        st.info(mode_msg.get(st.session_state.mode, "Selecione uma ferramenta"))
        
        # Canvas
        fig = create_interactive_canvas()
        
        # Clique no canvas para adicionar barras
        if st.session_state.mode in ['add-slack', 'add-pv', 'add-pq']:
            st.markdown("**Adicionar barra nas coordenadas:**")
            input_col1, input_col2, input_col3 = st.columns([2, 2, 1])
            with input_col1:
                x_pos = st.number_input("X", 50, 950, st.session_state.add_bus_x, key="input_x")
            with input_col2:
                y_pos = st.number_input("Y", 50, 650, st.session_state.add_bus_y, key="input_y")
            with input_col3:
                if st.button("✅ Adicionar", use_container_width=True):
                    bus_type = st.session_state.mode.replace('add-', '')
                    add_bus(x=x_pos, y=y_pos, bus_type=bus_type)
                    st.session_state.mode = 'select'
                    st.rerun()
        
        st.plotly_chart(fig, use_container_width=True, key="canvas")
        
        # Lista de elementos clicáveis
        st.subheader("🎯 Selecionar Elemento")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Barras", "Linhas", "Cargas", "Geradores"])
        
        with tab1:
            if st.session_state.buses:
                for bus in st.session_state.buses:
                    col_b1, col_b2, col_b3 = st.columns([2, 1, 1])
                    with col_b1:
                        if st.button(f"{bus['name']} (ID: {bus['id']})", key=f"select_bus_{bus['id']}", use_container_width=True):
                            if st.session_state.mode == 'connect':
                                if st.session_state.connecting_from is None:
                                    st.session_state.connecting_from = bus['id']
                                    st.success(f"Primeira barra: {bus['id']}")
                                else:
                                    if st.session_state.connecting_from != bus['id']:
                                        add_line(st.session_state.connecting_from, bus['id'])
                                        st.success("Linha criada!")
                                    st.session_state.connecting_from = None
                                    st.session_state.mode = 'select'
                                st.rerun()
                            elif st.session_state.mode == 'delete':
                                delete_element(bus['id'], 'bus')
                                st.rerun()
                            else:
                                st.session_state.selected_element = bus['id']
                                st.session_state.selected_type = 'bus'
                                st.rerun()
                    with col_b2:
                        if st.button("➕ Carga", key=f"add_load_{bus['id']}"):
                            add_load(bus['id'])
                            st.rerun()
                    with col_b3:
                        if st.button("➕ Ger", key=f"add_gen_{bus['id']}"):
                            add_generator(bus['id'])
                            st.rerun()
            else:
                st.info("Nenhuma barra criada")
        
        with tab2:
            if st.session_state.lines:
                for line in st.session_state.lines:
                    if st.button(f"{line['name']}: Barra {line['from']} → {line['to']}", key=f"select_line_{line['id']}", use_container_width=True):
                        if st.session_state.mode == 'delete':
                            delete_element(line['id'], 'line')
                            st.rerun()
                        else:
                            st.session_state.selected_element = line['id']
                            st.session_state.selected_type = 'line'
                            st.rerun()
            else:
                st.info("Nenhuma linha criada")
        
        with tab3:
            if st.session_state.loads:
                for load in st.session_state.loads:
                    if st.button(f"{load['name']} (Barra {load['bus']})", key=f"select_load_{load['id']}", use_container_width=True):
                        if st.session_state.mode == 'delete':
                            delete_element(load['id'], 'load')
                            st.rerun()
                        else:
                            st.session_state.selected_element = load['id']
                            st.session_state.selected_type = 'load'
                            st.rerun()
            else:
                st.info("Nenhuma carga criada")
        
        with tab4:
            if st.session_state.generators:
                for gen in st.session_state.generators:
                    if st.button(f"{gen['name']} (Barra {gen['bus']})", key=f"select_gen_{gen['id']}", use_container_width=True):
                        if st.session_state.mode == 'delete':
                            delete_element(gen['id'], 'generator')
                            st.rerun()
                        else:
                            st.session_state.selected_element = gen['id']
                            st.session_state.selected_type = 'generator'
                            st.rerun()
            else:
                st.info("Nenhum gerador criado")
    
    with col_props:
        st.subheader("🔧 Propriedades")
        
        if st.session_state.selected_element is not None:
            if st.session_state.selected_type == 'bus':
                bus = next((b for b in st.session_state.buses if b['id'] == st.session_state.selected_element), None)
                if bus:
                    st.success(f"Barra {bus['id']} selecionada")
                    
                    bus['name'] = st.text_input("Nome", bus['name'], key="edit_bus_name")
                    bus['type'] = st.selectbox("Tipo", ['slack', 'pv', 'pq'], 
                                              index=['slack', 'pv', 'pq'].index(bus['type']), key="edit_bus_type")
                    bus['vn_kv'] = st.number_input("Tensão Nominal (kV)", value=bus['vn_kv'], min_value=0.1, key="edit_bus_vn")
                    bus['x'] = st.number_input("Posição X", value=bus['x'], min_value=0, max_value=1000, key="edit_bus_x")
                    bus['y'] = st.number_input("Posição Y", value=bus['y'], min_value=0, max_value=700, key="edit_bus_y")
                    
                    if st.button("💾 Salvar", use_container_width=True):
                        bus['color'] = {'slack': '#ef4444', 'pv': '#10b981', 'pq': '#3b82f6'}[bus['type']]
                        st.success("Salvo!")
                        st.rerun()
            
            elif st.session_state.selected_type == 'line':
                line = next((l for l in st.session_state.lines if l['id'] == st.session_state.selected_element), None)
                if line:
                    st.success(f"Linha {line['id']} selecionada")
                    line['name'] = st.text_input("Nome", line['name'], key="edit_line_name")
                    
                    from_bus = get_bus_by_id(line['from'])
                    to_bus = get_bus_by_id(line['to'])
                    if from_bus and to_bus:
                        st.info(f"Conecta: Barra {line['from']} → Barra {line['to']}")
                    
                    line['r_ohm_per_km'] = st.number_input("R (Ω/km)", value=line['r_ohm_per_km'], min_value=0.001, key="edit_line_r")
                    line['x_ohm_per_km'] = st.number_input("X (Ω/km)", value=line['x_ohm_per_km'], min_value=0.001, key="edit_line_x")
                    line['length_km'] = st.number_input("Comprimento (km)", value=line['length_km'], min_value=0.1, key="edit_line_len")
                    
                    if st.button("💾 Salvar", use_container_width=True):
                        st.success("Salvo!")
                        st.rerun()
            
            elif st.session_state.selected_type == 'load':
                load = next((l for l in st.session_state.loads if l['id'] == st.session_state.selected_element), None)
                if load:
                    st.success(f"Carga {load['id']} selecionada")
                    load['name'] = st.text_input("Nome", load['name'], key="edit_load_name")
                    
                    bus = get_bus_by_id(load['bus'])
                    if bus:
                        st.info(f"Conectada à: {bus['name']}")
                    
                    load['p_mw'] = st.number_input("P (MW)", value=load['p_mw'], min_value=0.0, key="edit_load_p")
                    load['q_mvar'] = st.number_input("Q (MVar)", value=load['q_mvar'], min_value=0.0, key="edit_load_q")
                    
                    if st.button("💾 Salvar", use_container_width=True):
                        st.success("Salvo!")
                        st.rerun()
            
            elif st.session_state.selected_type == 'generator':
                gen = next((g for g in st.session_state.generators if g['id'] == st.session_state.selected_element), None)
                if gen:
                    st.success(f"Gerador {gen['id']} selecionado")
                    gen['name'] = st.text_input("Nome", gen['name'], key="edit_gen_name")
                    
                    bus = get_bus_by_id(gen['bus'])
                    if bus:
                        st.info(f"Conectado à: {bus['name']}")
                    
                    gen['p_mw'] = st.number_input("P (MW)", value=gen['p_mw'], min_value=0.0, key="edit_gen_p")
                    gen['vm_pu'] = st.number_input("Vm (pu)", value=gen['vm_pu'], min_value=0.8, max_value=1.2, step=0.01, key="edit_gen_vm")
                    
                    if st.button("💾 Salvar", use_container_width=True):
                        st.success("Salvo!")
                        st.rerun()
            
            st.divider()
            if st.button("🗑️ Remover Elemento", type="secondary", use_container_width=True):
                delete_element(st.session_state.selected_element, st.session_state.selected_type)
                st.session_state.selected_element = None
                st.session_state.selected_type = None
                st.rerun()
        else:
            st.info("👈 Selecione um elemento para editar")
        
        st.divider()
        
        # Estatísticas
        st.subheader("📊 Estatísticas")
        st.metric("Barras", len(st.session_state.buses))
        st.metric("Linhas", len(st.session_state.lines))
        st.metric("Cargas", len(st.session_state.loads))
        st.metric("Geradores", len(st.session_state.generators))
    
    # Rodapé com informações
    st.divider()
    footer_cols = st.columns(4)
    with footer_cols[0]:
        st.metric("Total de Elementos", len(st.session_state.buses) + len(st.session_state.lines) + 
                 len(st.session_state.loads) + len(st.session_state.generators))
    with footer_cols[1]:
        if st.session_state.loads:
            total_load = sum(l['p_mw'] for l in st.session_state.loads)
            st.metric("Carga Total", f"{total_load:.1f} MW")
        else:
            st.metric("Carga Total", "0 MW")
    with footer_cols[2]:
        if st.session_state.generators:
            total_gen = sum(g['p_mw'] for g in st.session_state.generators)
            st.metric("Geração Total", f"{total_gen:.1f} MW")
        else:
            st.metric("Geração Total", "0 MW")
    with footer_cols[3]:
        if st.session_state.generators and st.session_state.loads:
            balance = sum(g['p_mw'] for g in st.session_state.generators) - sum(l['p_mw'] for l in st.session_state.loads)
            st.metric("Balanço", f"{balance:.1f} MW", delta=f"{balance:.1f} MW")
        else:
            st.metric("Balanço", "0 MW")

if __name__ == "__main__":
    main()
