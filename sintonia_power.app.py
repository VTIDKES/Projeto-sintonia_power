# STREAMLIT + PANDAPOWER - Análise Completa
# Sistemas Elétricos de Potência
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import pandapower as pp

st.set_page_config(page_title="Desenho Interativo de Rede Elétrica", layout="wide")
st.title("Desenho Interativo de Rede Elétrica com Pandapower")

# =========================
# Estado inicial
# =========================
if "barras" not in st.session_state:
    st.session_state.barras = pd.DataFrame(columns=["name", "x", "y"])
if "linhas" not in st.session_state:
    st.session_state.linhas = pd.DataFrame(columns=["from", "to"])

# =========================
# Adicionar barra
# =========================
with st.sidebar:
    st.subheader("Adicionar Barra")
    barra_name = st.text_input("Nome da Barra")
    x_coord = st.number_input("X", value=0)
    y_coord = st.number_input("Y", value=0)
    if st.button("Adicionar Barra"):
        if barra_name.strip() != "":
            st.session_state.barras.loc[len(st.session_state.barras)] = [barra_name, x_coord, y_coord]

# =========================
# Conectar barras
# =========================
with st.sidebar:
    st.subheader("Conectar Barras")
    if len(st.session_state.barras) >= 2:
        from_barra = st.selectbox("De", st.session_state.barras["name"])
        to_barra = st.selectbox("Para", st.session_state.barras["name"])
        if st.button("Adicionar Linha"):
            if from_barra != to_barra:
                st.session_state.linhas.loc[len(st.session_state.linhas)] = [from_barra, to_barra]

# =========================
# Plot da rede
# =========================
fig = go.Figure()

# Linhas
for idx, linha in st.session_state.linhas.iterrows():
    b_from = st.session_state.barras[st.session_state.barras["name"] == linha["from"]].iloc[0]
    b_to = st.session_state.barras[st.session_state.barras["name"] == linha["to"]].iloc[0]
    fig.add_trace(go.Scatter(
        x=[b_from.x, b_to.x],
        y=[b_from.y, b_to.y],
        mode="lines",
        line=dict(color="gray", width=3)
    ))

# Barras
fig.add_trace(go.Scatter(
    x=st.session_state.barras["x"],
    y=st.session_state.barras["y"],
    mode="markers+text",
    marker=dict(size=20, color="blue"),
    text=st.session_state.barras["name"],
    textposition="top center"
))

fig.update_layout(
    width=800,
    height=600,
    xaxis=dict(title="X"),
    yaxis=dict(title="Y"),
    showlegend=False
)

st.plotly_chart(fig)

# =========================
# Criar rede Pandapower
# =========================
if st.button("Gerar rede Pandapower e calcular fluxo"):
    net = pp.create_empty_network()
    barra_map = {}

    # Adicionar barras na rede
    for idx, row in st.session_state.barras.iterrows():
        # Arbitrariamente barras com y>0 em alta tensão, y<=0 baixa tensão
        vn_kv = 110 if row.y > 0 else 20
        barra_map[row.name] = pp.create_bus(net, vn_kv=vn_kv, name=row.name)

    # Ext_grid na primeira barra
    if len(barra_map) > 0:
        first_bus = list(barra_map.values())[0]
        pp.create_ext_grid(net, bus=first_bus, vm_pu=1.02, name="Subestação")

    # Conectar linhas
    for idx, row in st.session_state.linhas.iterrows():
        pp.create_line(net, barra_map[row["from"]], barra_map[row["to"]], length_km=5, std_type="NAYY 4x50 SE")

    # Executar fluxo de carga
    pp.runpp(net)
    st.success("Fluxo de carga calculado com sucesso!")

    # Mostrar resultados
    st.subheader("Tensões nas Barras (p.u.)")
    st.dataframe(net.res_bus[["vm_pu", "va_degree"]])

    st.subheader("Carregamento das Linhas (%)")
    st.dataframe(net.res_line[["loading_percent"]])




# Rodapé
st.markdown("---")


# Informações técnicas
st.caption("Power System Studio v2.0 | Desenvolvido com Streamlit e Plotly | ⚡ Análise de Sistemas Elétricos de Potência")

if __name__ == "__main__":
    main()
