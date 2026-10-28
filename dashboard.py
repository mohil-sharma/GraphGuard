"""Compliance dashboard:  streamlit run graphguard/dashboard.py"""
import networkx as nx
import pandas as pd
import streamlit as st

from graphguard.service import GraphGuard

st.set_page_config(page_title="GraphGuard", layout="wide")


@st.cache_resource
def svc():
    return GraphGuard()


s = svc()
st.title("GraphGuard - GNN anomaly detection with grounded explanations")
m = s.metrics()
if m:
    st.subheader("Detection quality (illicit class, temporal test split)")
    st.dataframe(pd.DataFrame(m["models"]).T[["precision", "recall", "f1"]].style.format("{:.3f}"))

c1, c2, c3 = st.columns(3)
pat = c1.selectbox("Pattern", [None, "fan_out", "fund_cycling", "community_bridging"])
step = c2.number_input("Time step (0 = all)", 0, 49, 0)
lim = c3.slider("Rows", 10, 200, 50)
rows = pd.DataFrame(s.flagged(lim, pat, int(step) or None))
st.dataframe(rows, use_container_width=True)
if len(rows):
    tx = st.selectbox("Inspect transaction", rows.tx_id)
    grounded = st.toggle("RAG-grounded explanation", True)
    w = s.wallet(int(tx), True, grounded)
    a, b = st.columns([2, 1])
    with a:
        st.markdown(f"**Risk {w['risk']:.2f}** - patterns: {', '.join(w['patterns']) or 'none'}")
        st.write(w["explanation"]["explanation"])
        st.caption("Citations: " + ", ".join(c["id"] for c in w["explanation"]["citations"]))
        st.json(w["evidence"])
    with b:
        G = nx.DiGraph(w["subgraph"]["edges"])
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4, 4))
        nx.draw(G, nx.spring_layout(G, seed=1), ax=ax, node_size=60, with_labels=False,
                node_color=["red" if n == int(tx) else "lightgray" for n in G.nodes])
        st.pyplot(fig)
