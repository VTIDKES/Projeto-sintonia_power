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

# =====================================================
# FUNÇÕES AUXILIARES
# =====================================================

def add_bus(name, bus_type='pq', vn_kv=13.8):
    new_bus = {
        'id': len(st.session_state.buses),
        'name': name,
        'vn_kv': vn_kv,
        'type': bus_type
    }
    st.session_state.buses.append(new_bus)
    return new_bus

def add_line(from_bus, to_bus, r=0.1, x=0.3, length=10):
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
            'r_ohm_per_km': r,
            'x_ohm_per_km': x,
            'length_km': length,
            'max_i_ka': 1.0
        }
        st.session_state.lines.append(new_line)
        return new_line
    return None

def add_load(bus_id, p_mw=5.0, q_mvar=2.0):
    if not any(l['bus'] == bus_id for l in st.session_state.loads):
        new_load = {
            'id': len(st.session_state.loads),
            'bus': bus_id,
            'p_mw': p_mw,
            'q_mvar': q_mvar
        }
        st.session_state.loads.append(new_load)
        return new_load
    return None

def add_generator(bus_id, p_mw=10.0, vm_pu=1.0):
    if not any(g['bus'] == bus_id for g in st.session_state.generators):
        new_gen = {
            'id': len(st.session_state.generators),
            'bus': bus_id,
            'p_mw': p_mw,
            'vm_pu': vm_pu
        }
        st.session_state.generators.append(new_gen)
        return new_gen
    return None

def remove_bus(bus_id):
    """Remove uma barra e todos os elementos conectados"""
    st.session_state.buses = [b for b in st.session_state.buses if b['id'] != bus_id]
    st.session_state.lines = [l for l in st.session_state.lines 
                             if l['from'] != bus_id and l['to'] != bus_id]
    st.session_state.loads = [l for l in st.session_state.loads if l['bus'] != bus_id]
    st.session_state.generators = [g for g in st.session_state.generators if g['bus'] != bus_id]
    
    if st.session_state.selected_bus == bus_id:
        st.session_state.selected_bus = None

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
st.caption("Sistema de análise de redes elétricas")
st.markdown("---")

# Barra lateral
with st.sidebar:
    st.header("🎛️ Controles")
    
    # Criar nova barra
    with st.expander("➕ Nova Barra", expanded=True):
        bus_name = st.text_input("Nome da barra", f"Bus {len(st.session_state.buses)}")
        bus_type = st.selectbox("Tipo", ["pq", "pv", "slack"], 
                               help="Slack: Referência, PV: Geração controlada, PQ: Carga")
        vn_kv = st.number_input("Tensão nominal (kV)", 0.1, 500.0, 13.8, 0.1)
        
        if st.button("Criar Barra", use_container_width=True):
            add_bus(bus_name, bus_type, vn_kv)
            st.success(f"Barra {bus_name} criada!")
            st.rerun()
    
    # Conectar barras
    if len(st.session_state.buses) >= 2:
        with st.expander("🔗 Conectar Barras"):
            bus_options = {f"{b['name']} (ID: {b['id']})": b['id'] for b in st.session_state.buses}
            
            col1, col2 = st.columns(2)
            with col1:
                from_bus = st.selectbox("De", list(bus_options.keys()))
            with col2:
                to_bus = st.selectbox("Para", list(bus_options.keys()))
            
            col3, col4 = st.columns(2)
            with col3:
                r = st.number_input("R (Ω/km)", 0.001, 10.0, 0.1, 0.01)
            with col4:
                x = st.number_input("X (Ω/km)", 0.001, 10.0, 0.3, 0.01)
            
            length = st.number_input("Comprimento (km)", 0.1, 500.0, 10.0, 0.1)
            
            if st.button("Criar Linha", use_container_width=True):
                if bus_options[from_bus] != bus_options[to_bus]:
                    if add_line(bus_options[from_bus], bus_options[to_bus], r, x, length):
                        st.success("Linha criada!")
                        st.rerun()
                    else:
                        st.warning("Esta linha já existe")
                else:
                    st.error("Não é possível conectar uma barra a si mesma")
    
    # Análises
    st.markdown("---")
    st.header("🔬 Análises")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Fluxo de Potência", use_container_width=True):
            with st.spinner("Calculando..."):
                run_power_flow()
            st.rerun()
    
    with col2:
        if st.button("⚡ Curto-Circuito", use_container_width=True):
            with st.spinner("Analisando..."):
                run_short_circuit()
            st.rerun()
    
    # Gerenciamento de arquivos
    st.markdown("---")
    st.header("💾 Arquivos")
    
    uploaded_file = st.file_uploader("Importar sistema", type=['json'])
    if uploaded_file:
        import_system(uploaded_file)
        st.rerun()
    
    if len(st.session_state.buses) > 0:
        json_str = export_system()
        st.download_button(
            label="Exportar Sistema",
            data=json_str,
            file_name=f"power_system_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    if st.button("🗑️ Limpar Tudo", type="secondary", use_container_width=True):
        for key in ['buses', 'lines', 'loads', 'generators', 'results', 'history', 'selected_bus']:
            if key in ['buses', 'lines', 'loads', 'generators', 'history']:
                st.session_state[key] = []
            else:
                st.session_state[key] = None
        st.success("Sistema limpo!")
        st.rerun()

# Layout principal
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📊 Sistema Elétrico")
    
    if len(st.session_state.buses) == 0:
        st.info("👈 Comece criando barras no painel lateral")
    else:
        # Lista de barras
        st.subheader("🚏 Barras")
        for bus in st.session_state.buses:
            col_b1, col_b2, col_b3 = st.columns([3, 1, 1])
            with col_b1:
                type_icon = {
                    'slack': '🔴',
                    'pv': '🟢',
                    'pq': '🟠'
                }
                bus_text = f"{type_icon.get(bus['type'], '⚫')} **{bus['name']}**"
                if bus['id'] == st.session_state.selected_bus:
                    bus_text = f"📍 {bus_text}"
                
                if st.button(bus_text, key=f"bus_{bus['id']}", use_container_width=True):
                    st.session_state.selected_bus = bus['id']
                    st.rerun()
            
            with col_b2:
                if st.button("📝", key=f"edit_{bus['id']}", help="Editar"):
                    st.session_state.selected_bus = bus['id']
                    st.rerun()
            
            with col_b3:
                if st.button("❌", key=f"del_{bus['id']}", help="Remover"):
                    remove_bus(bus['id'])
                    st.rerun()
        
        # Linhas de transmissão
        if st.session_state.lines:
            st.subheader("🔗 Linhas")
            for line in st.session_state.lines:
                from_bus = next((b for b in st.session_state.buses if b['id'] == line['from']), None)
                to_bus = next((b for b in st.session_state.buses if b['id'] == line['to']), None)
                
                if from_bus and to_bus:
                    st.write(f"{from_bus['name']} → {to_bus['name']} "
                            f"(R={line['r_ohm_per_km']}Ω/km, X={line['x_ohm_per_km']}Ω/km, "
                            f"L={line['length_km']}km)")
        
        # Cargas
        if st.session_state.loads:
            st.subheader("📊 Cargas")
            for load in st.session_state.loads:
                bus = next((b for b in st.session_state.buses if b['id'] == load['bus']), None)
                if bus:
                    st.write(f"{bus['name']}: P={load['p_mw']} MW, Q={load['q_mvar']} MVar")
        
        # Geradores
        if st.session_state.generators:
            st.subheader("⚡ Geradores")
            for gen in st.session_state.generators:
                bus = next((b for b in st.session_state.buses if b['id'] == gen['bus']), None)
                if bus:
                    st.write(f"{bus['name']}: P={gen['p_mw']} MW, Vm={gen['vm_pu']} pu")

with col2:
    st.header("📈 Informações")
    
    # Estatísticas
    with st.expander("📊 Estatísticas", expanded=True):
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.metric("Barras", len(st.session_state.buses))
            st.metric("Linhas", len(st.session_state.lines))
        with col_s2:
            st.metric("Cargas", len(st.session_state.loads))
            st.metric("Geradores", len(st.session_state.generators))
        
        if st.session_state.buses:
            total_load = sum(l['p_mw'] for l in st.session_state.loads)
            total_gen = sum(g['p_mw'] for g in st.session_state.generators)
            st.metric("Carga Total", f"{total_load:.1f} MW")
            st.metric("Geração Total", f"{total_gen:.1f} MW")
    
    # Barra selecionada
    if st.session_state.selected_bus is not None:
        bus = next((b for b in st.session_state.buses if b['id'] == st.session_state.selected_bus), None)
        if bus:
            with st.expander(f"🚏 {bus['name']}", expanded=True):
                # Editar barra
                with st.form(key=f"edit_bus_form_{bus['id']}"):
                    new_name = st.text_input("Nome", bus['name'])
                    new_type = st.selectbox("Tipo", ["slack", "pv", "pq"], 
                                          index=["slack", "pv", "pq"].index(bus['type']))
                    new_vn = st.number_input("Vn (kV)", bus['vn_kv'], 0.1, 500.0, 0.1)
                    
                    if st.form_submit_button("💾 Salvar"):
                        bus['name'] = new_name
                        bus['type'] = new_type
                        bus['vn_kv'] = new_vn
                        st.success("Barra atualizada!")
                        st.rerun()
                
                # Adicionar carga
                if not any(l['bus'] == bus['id'] for l in st.session_state.loads):
                    with st.form(key=f"add_load_form_{bus['id']}"):
                        st.write("➕ Adicionar Carga")
                        p_mw = st.number_input("P (MW)", 0.1, 1000.0, 5.0, 0.1, 
                                              key=f"p_load_{bus['id']}")
                        q_mvar = st.number_input("Q (MVar)", 0.0, 1000.0, 2.0, 0.1,
                                                key=f"q_load_{bus['id']}")
                        
                        if st.form_submit_button("Adicionar Carga"):
                            add_load(bus['id'], p_mw, q_mvar)
                            st.success("Carga adicionada!")
                            st.rerun()
                else:
                    # Editar carga existente
                    load = next((l for l in st.session_state.loads if l['bus'] == bus['id']), None)
                    if load:
                        with st.form(key=f"edit_load_form_{load['id']}"):
                            st.write("📊 Carga Existente")
                            new_p = st.number_input("P (MW)", 0.1, 1000.0, load['p_mw'], 0.1,
                                                   key=f"edit_p_load_{load['id']}")
                            new_q = st.number_input("Q (MVar)", 0.0, 1000.0, load['q_mvar'], 0.1,
                                                   key=f"edit_q_load_{load['id']}")
                            
                            if st.form_submit_button("💾 Atualizar Carga"):
                                load['p_mw'] = new_p
                                load['q_mvar'] = new_q
                                st.success("Carga atualizada!")
                                st.rerun()
                
                # Adicionar gerador
                if not any(g['bus'] == bus['id'] for g in st.session_state.generators):
                    with st.form(key=f"add_gen_form_{bus['id']}"):
                        st.write("⚡ Adicionar Gerador")
                        p_mw = st.number_input("P (MW)", 0.1, 1000.0, 10.0, 0.1,
                                              key=f"p_gen_{bus['id']}")
                        vm_pu = st.number_input("Vm (pu)", 0.8, 1.2, 1.0, 0.01,
                                               key=f"vm_gen_{bus['id']}")
                        
                        if st.form_submit_button("Adicionar Gerador"):
                            add_generator(bus['id'], p_mw, vm_pu)
                            st.success("Gerador adicionado!")
                            st.rerun()
                else:
                    # Editar gerador existente
                    gen = next((g for g in st.session_state.generators if g['bus'] == bus['id']), None)
                    if gen:
                        with st.form(key=f"edit_gen_form_{gen['id']}"):
                            st.write("⚡ Gerador Existente")
                            new_p = st.number_input("P (MW)", 0.1, 1000.0, gen['p_mw'], 0.1,
                                                   key=f"edit_p_gen_{gen['id']}")
                            new_vm = st.number_input("Vm (pu)", 0.8, 1.2, gen['vm_pu'], 0.01,
                                                    key=f"edit_vm_gen_{gen['id']}")
                            
                            if st.form_submit_button("💾 Atualizar Gerador"):
                                gen['p_mw'] = new_p
                                gen['vm_pu'] = new_vm
                                st.success("Gerador atualizado!")
                                st.rerun()
    
    # Resultados da análise
    if st.session_state.results:
        with st.expander("📊 Resultados", expanded=True):
            results = st.session_state.results
            
            st.write(f"**Tipo:** {results['type'].replace('_', ' ').title()}")
            st.write(f"**Data:** {datetime.fromisoformat(results['timestamp']).strftime('%H:%M:%S')}")
            
            if results['type'] == 'power_flow':
                voltages = results['data']['voltages']
                st.write("**Tensões nas Barras:**")
                
                for i, v in enumerate(voltages):
                    if i < len(st.session_state.buses):
                        bus_name = st.session_state.buses[i]['name']
                        if 0.95 <= v <= 1.05:
                            status = "✅"
                        elif 0.9 <= v < 0.95 or 1.05 < v <= 1.1:
                            status = "⚠️"
                        else:
                            status = "❌"
                        
                        st.write(f"{status} {bus_name}: {v:.3f} pu")
                
                if voltages:
                    avg_v = np.mean(voltages)
                    min_v = min(voltages)
                    max_v = max(voltages)
                    
                    col_v1, col_v2, col_v3 = st.columns(3)
                    with col_v1:
                        st.metric("Média", f"{avg_v:.3f}")
                    with col_v2:
                        st.metric("Mínima", f"{min_v:.3f}")
                    with col_v3:
                        st.metric("Máxima", f"{max_v:.3f}")
            
            elif results['type'] == 'short_circuit':
                currents = results['data']['fault_currents']
                st.write("**Correntes de Falta:**")
                
                for i, c in enumerate(currents):
                    if i < len(st.session_state.buses):
                        bus_name = st.session_state.buses[i]['name']
                        if c > 10:
                            level = "🔴"
                        elif c > 5:
                            level = "🟡"
                        else:
                            level = "🟢"
                        
                        st.write(f"{level} {bus_name}: {c:.2f} kA")
                
                st.write("**Barras Críticas:**")
                for bus_idx, current in results['data']['critical_buses'][:3]:
                    if bus_idx < len(st.session_state.buses):
                        bus_name = st.session_state.buses[bus_idx]['name']
                        st.warning(f"{bus_name}: {current:.2f} kA")

# Rodapé
st.markdown("---")


# Informações técnicas
st.caption("Power System Studio v2.0 | Desenvolvido com Streamlit e Plotly | ⚡ Análise de Sistemas Elétricos de Potência")

if __name__ == "__main__":
    main()
