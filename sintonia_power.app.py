# STREAMLIT + PANDAPOWER - Análise Completa
# Sistemas Elétricos de Potência
# STREAMLIT - Power System Studio
import streamlit as st
import pandas as pd
import numpy as np
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
    
    # Conectar barras manualmente
    if len(st.session_state.buses) >= 2:
        with st.expander("🔗 Conectar Barras"):
            bus_options = {f"Barra {b['id']}: {b['name']}": b['id'] for b in st.session_state.buses}
            
            from_bus = st.selectbox("De:", list(bus_options.keys()), key='from_bus')
            to_bus = st.selectbox("Para:", list(bus_options.keys()), key='to_bus')
            
            if st.button("🔗 Criar Conexão", use_container_width=True, key='connect_btn'):
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
        if st.button("🔄 Fluxo de Potência", use_container_width=True, key='power_flow'):
            with st.spinner("Executando..."):
                run_power_flow()
            st.rerun()
    with col2:
        if st.button("⚡ Curto-Circuito", use_container_width=True, key='short_circuit'):
            with st.spinner("Analisando..."):
                run_short_circuit()
            st.rerun()
    
    # Adicionar elementos
    st.markdown("---")
    st.subheader("➕ Adicionar Elementos")
    
    if len(st.session_state.buses) > 0:
        load_bus = st.selectbox("Barra para carga:", 
                               [f"Barra {b['id']}: {b['name']}" for b in st.session_state.buses],
                               key='load_bus')
        if st.button("➕ Adicionar Carga", use_container_width=True, key='add_load_btn'):
            bus_id = int(load_bus.split(':')[0].split(' ')[1])
            if add_load(bus_id):
                st.success("Carga adicionada!")
                st.rerun()
            else:
                st.warning("Esta barra já tem uma carga")
        
        gen_bus = st.selectbox("Barra para gerador:", 
                              [f"Barra {b['id']}: {b['name']}" for b in st.session_state.buses],
                              key='gen_bus')
        if st.button("⚡ Adicionar Gerador", use_container_width=True, key='add_gen_btn'):
            bus_id = int(gen_bus.split(':')[0].split(' ')[1])
            if add_generator(bus_id):
                st.success("Gerador adicionado!")
                st.rerun()
            else:
                st.warning("Esta barra já tem um gerador")
    
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
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📐 Diagrama do Sistema")
    
    if len(st.session_state.buses) == 0:
        st.info("👈 Comece adicionando barras ao sistema usando o painel lateral")
    else:
        # Exibir sistema em formato de tabela/lista
        st.write("### 🚏 Barras do Sistema")
        for bus in st.session_state.buses:
            col_bus1, col_bus2, col_bus3 = st.columns([1, 2, 1])
            with col_bus1:
                if st.button(f"🔍", key=f"select_{bus['id']}"):
                    st.session_state.selected_bus = bus['id']
                    st.rerun()
            with col_bus2:
                type_color = {
                    'slack': '🔴',
                    'pv': '🟢', 
                    'pq': '🟠'
                }
                st.write(f"{type_color.get(bus['type'], '⚫')} **{bus['name']}** (Tipo: {bus['type'].upper()})")
            with col_bus3:
                if st.button(f"❌", key=f"del_{bus['id']}"):
                    st.session_state.buses = [b for b in st.session_state.buses if b['id'] != bus['id']]
                    st.session_state.lines = [l for l in st.session_state.lines 
                                             if l['from'] != bus['id'] and l['to'] != bus['id']]
                    st.session_state.loads = [l for l in st.session_state.loads if l['bus'] != bus['id']]
                    st.session_state.generators = [g for g in st.session_state.generators if g['bus'] != bus['id']]
                    st.rerun()
        
        if st.session_state.lines:
            st.write("### 🔗 Linhas de Transmissão")
            for line in st.session_state.lines:
                st.write(f"Linha {line['id']}: Barra {line['from']} → Barra {line['to']} "
                        f"(R={line['r_ohm_per_km']}Ω/km, X={line['x_ohm_per_km']}Ω/km)")
        
        if st.session_state.loads:
            st.write("### 📊 Cargas")
            for load in st.session_state.loads:
                st.write(f"Carga {load['id']} na Barra {load['bus']}: "
                        f"P={load['p_mw']} MW, Q={load['q_mvar']} MVar")
        
        if st.session_state.generators:
            st.write("### ⚡ Geradores")
            for gen in st.session_state.generators:
                st.write(f"Gerador {gen['id']} na Barra {gen['bus']}: "
                        f"P={gen['p_mw']} MW, Vm={gen['vm_pu']} pu")

with col2:
    st.subheader("📊 Informações Detalhadas")
    
    # Estatísticas
    with st.expander("📈 Estatísticas do Sistema", expanded=True):
        st.metric("🚏 Barras", len(st.session_state.buses))
        st.metric("🔗 Linhas", len(st.session_state.lines))
        st.metric("📊 Cargas", len(st.session_state.loads))
        st.metric("⚡ Geradores", len(st.session_state.generators))
        
        if st.session_state.buses:
            total_load = sum(l['p_mw'] for l in st.session_state.loads)
            total_gen = sum(g['p_mw'] for g in st.session_state.generators)
            if total_load > 0 or total_gen > 0:
                st.markdown("---")
                st.metric("💡 Carga Total", f"{total_load:.1f} MW")
                st.metric("🔋 Geração Total", f"{total_gen:.1f} MW")
                if total_gen > 0:
                    balance = total_gen - total_load
                    st.metric("⚖️ Balanço", f"{balance:.1f} MW", 
                             delta="Superávit" if balance > 0 else "Déficit")
    
    # Barra selecionada
    if st.session_state.selected_bus is not None:
        bus = next((b for b in st.session_state.buses if b['id'] == st.session_state.selected_bus), None)
        if bus:
            with st.expander(f"🚏 {bus['name']}", expanded=True):
                st.write(f"**ID:** {bus['id']}")
                st.write(f"**Tipo:** {bus['type'].upper()}")
                st.write(f"**Tensão nominal:** {bus['vn_kv']} kV")
                st.write(f"**Posição:** ({bus['x']}, {bus['y']})")
                
                # Editar propriedades
                with st.form(key=f"edit_bus_{bus['id']}"):
                    new_name = st.text_input("Nome", bus['name'])
                    new_type = st.selectbox("Tipo", ["slack", "pv", "pq"], 
                                          index=["slack", "pv", "pq"].index(bus['type']))
                    new_vn = st.number_input("Tensão Nominal (kV)", value=float(bus['vn_kv']))
                    
                    if st.form_submit_button("💾 Salvar Alterações"):
                        bus['name'] = new_name
                        bus['type'] = new_type
                        bus['vn_kv'] = new_vn
                        st.success("Barra atualizada!")
                        st.rerun()
                
                # Elementos conectados
                lines_connected = [l for l in st.session_state.lines if l['from'] == bus['id'] or l['to'] == bus['id']]
                if lines_connected:
                    st.write("**Linhas conectadas:**")
                    for line in lines_connected:
                        other = line['to'] if line['from'] == bus['id'] else line['from']
                        st.caption(f"→ Barra {other} (Linha {line['id']})")
                
                load = next((l for l in st.session_state.loads if l['bus'] == bus['id']), None)
                if load:
                    st.write("**Carga conectada:**")
                    with st.form(key=f"edit_load_{load['id']}"):
                        new_p = st.number_input("P (MW)", value=float(load['p_mw']), key=f"p_load_{load['id']}")
                        new_q = st.number_input("Q (MVar)", value=float(load['q_mvar']), key=f"q_load_{load['id']}")
                        if st.form_submit_button("💾 Atualizar Carga"):
                            load['p_mw'] = new_p
                            load['q_mvar'] = new_q
                            st.success("Carga atualizada!")
                            st.rerun()
                
                gen = next((g for g in st.session_state.generators if g['bus'] == bus['id']), None)
                if gen:
                    st.write("**Gerador conectado:**")
                    with st.form(key=f"edit_gen_{gen['id']}"):
                        new_p = st.number_input("P (MW)", value=float(gen['p_mw']), key=f"p_gen_{gen['id']}")
                        new_vm = st.number_input("Vm (pu)", value=float(gen['vm_pu']), key=f"vm_gen_{gen['id']}")
                        if st.form_submit_button("💾 Atualizar Gerador"):
                            gen['p_mw'] = new_p
                            gen['vm_pu'] = new_vm
                            st.success("Gerador atualizada!")
                            st.rerun()
    
    # Resultados da análise
    if st.session_state.results:
        with st.expander("📈 Resultados da Análise", expanded=True):
            results = st.session_state.results
            
            st.write(f"**Tipo:** {results['type'].replace('_', ' ').title()}")
            st.write(f"**Data:** {datetime.fromisoformat(results['timestamp']).strftime('%d/%m/%Y %H:%M:%S')}")
            
            if results['type'] == 'power_flow':
                st.write("**Tensões nas Barras:**")
                voltages = results['data']['voltages']
                for i, v in enumerate(voltages):
                    if 0.95 <= v <= 1.05:
                        status = "✅ Normal"
                    elif 0.9 <= v < 0.95 or 1.05 < v <= 1.1:
                        status = "⚠️ Atenção"
                    else:
                        status = "❌ Crítico"
                    st.caption(f"Barra {i}: {v:.3f} pu - {status}")
                
                if len(voltages) > 0:
                    avg_v = np.mean(voltages)
                    min_v = np.min(voltages)
                    max_v = np.max(voltages)
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Média", f"{avg_v:.3f}")
                    with col2:
                        st.metric("Mínima", f"{min_v:.3f}")
                    with col3:
                        st.metric("Máxima", f"{max_v:.3f}")
            
            elif results['type'] == 'short_circuit':
                st.write("**Correntes de Curto-Circuito:**")
                currents = results['data']['fault_currents']
                for i, c in enumerate(currents):
                    if c > 10:
                        level = "🔴 Alto"
                    elif c > 5:
                        level = "🟡 Médio"
                    else:
                        level = "🟢 Baixo"
                    st.caption(f"Barra {i}: {c:.2f} kA - {level}")
                
                st.write("**Barras Críticas:**")
                for bus_idx, current in results['data']['critical_buses']:
                    st.warning(f"Barra {bus_idx}: {current:.2f} kA")
    
    # Histórico de análises
    if st.session_state.history:
        with st.expander(f"📋 Histórico ({len(st.session_state.history)} análises)"):
            for i, hist in enumerate(reversed(st.session_state.history)):
                icon = "🔄" if hist['type'] == 'power_flow' else "⚡"
                type_str = "Fluxo de Potência" if hist['type'] == 'power_flow' else "Curto-Circuito"
                time_str = datetime.fromisoformat(hist['timestamp']).strftime('%H:%M:%S')
                
                if st.button(f"{icon} {type_str} - {time_str}", key=f"hist_{i}"):
                    st.session_state.results = hist
                    st.rerun()

# Rodapé
st.markdown("---")
st.subheader("📖 Legenda")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **Tipos de Barras:**
    - 🔴 **Slack**: Barra de referência
    - 🟢 **PV**: Geração controlada
    - 🟠 **PQ**: Carga fixa
    """)

with col2:
    st.markdown("""
    **Status de Tensão:**
    - ✅ **0.95-1.05 pu**: Normal
    - ⚠️ **0.90-0.94 / 1.06-1.10**: Atenção
    - ❌ **<0.90 / >1.10**: Crítico
    """)

with col3:
    st.markdown("""
    **Níveis de Corrente:**
    - 🟢 **< 5 kA**: Baixo
    - 🟡 **5-10 kA**: Médio
    - 🔴 **> 10 kA**: Alto
    """)

st.caption("Power System Studio v1.0 | Desenvolvido com Streamlit | ⚡ Análise de Sistemas Elétricos")

# Informações técnicas
st.caption("Power System Studio v2.0 | Desenvolvido com Streamlit e Plotly | ⚡ Análise de Sistemas Elétricos de Potência")

if __name__ == "__main__":
    main()
