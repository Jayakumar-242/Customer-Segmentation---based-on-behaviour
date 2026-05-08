import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Customer Segmentation", page_icon="🎯", layout="wide")

PALETTE = ["#2E4057", "#048A81", "#EF946C", "#54C6EB", "#C4A35A", "#8338EC", "#FF006E", "#06D6A0"]

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
* { font-family: 'Inter', sans-serif !important; }
.main { background: #F7F9FC; }
[data-testid="stSidebar"] { background: #2E4057; }
[data-testid="stSidebar"] * { color: white !important; }
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stMultiSelect label { color: white !important; }
.kpi { background: white; border-radius: 14px; padding: 22px 20px;
       box-shadow: 0 2px 16px rgba(46,64,87,0.10); text-align: center; }
.kpi-val { font-size: 2rem; font-weight: 700; }
.kpi-lbl { font-size: 0.82rem; color: #6B7A8D; margin-top: 4px; font-weight: 500; }
.sec { font-size: 1.15rem; font-weight: 700; color: #2E4057;
       border-left: 4px solid #048A81; padding-left: 10px; margin: 18px 0 10px; }
</style>
""", unsafe_allow_html=True)

# ── Generate Data ─────────────────────────────────────────────────────────────
@st.cache_data
def generate_data():
    np.random.seed(42)
    n = 500
    return pd.DataFrame({
        "CustomerID": [f"CUST{str(i).zfill(4)}" for i in range(1, n+1)],
        "Age": np.random.randint(18, 70, n),
        "Annual_Income": np.random.randint(20000, 150000, n),
        "Spending_Score": np.random.randint(1, 100, n),
        "Purchase_Frequency": np.random.randint(1, 52, n),
        "Avg_Order_Value": np.round(np.random.uniform(20, 500, n), 2),
        "Gender": np.random.choice(["Male", "Female"], n),
        "Region": np.random.choice(["North", "South", "East", "West"], n),
    })

# ── Cluster ───────────────────────────────────────────────────────────────────
FEATURES = ["Age", "Annual_Income", "Spending_Score", "Purchase_Frequency", "Avg_Order_Value"]

@st.cache_data
def run_pipeline(n_clusters):
    df = generate_data()
    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURES])

    # Elbow + Silhouette
    inertias, silhouettes = [], []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbl = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, lbl))

    # Final model
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["Cluster"] = model.fit_predict(X)

    # PCA
    pca = PCA(n_components=2, random_state=42)
    comps = pca.fit_transform(X)
    df["PC1"], df["PC2"] = comps[:, 0], comps[:, 1]

    summary = df.groupby("Cluster")[FEATURES].mean().round(1)
    summary["Count"] = df.groupby("Cluster").size()

    LABELS = {0:"Budget Shoppers", 1:"High-Value Loyalists", 2:"Young Explorers",
              3:"Occasional Buyers", 4:"Premium Spenders", 5:"Seg 5", 6:"Seg 6", 7:"Seg 7"}
    df["Segment"] = df["Cluster"].map(lambda x: LABELS.get(x, f"Segment {x}"))

    return df, summary, {"k": list(range(2,11)), "inertia": inertias, "sil": silhouettes}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 Customer Segmentation")
    st.markdown("---")
    n_clusters = st.slider("Number of Segments", 2, 8, 4)
    st.markdown("---")
    regions = st.multiselect("Region", ["North","South","East","West"], default=["North","South","East","West"])
    genders = st.multiselect("Gender", ["Male","Female"], default=["Male","Female"])
    st.markdown("---")
    st.markdown("<small>K-Means · PCA · Streamlit</small>", unsafe_allow_html=True)

df, summary, elbow = run_pipeline(n_clusters)
dff = df[df["Region"].isin(regions) & df["Gender"].isin(genders)]

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🎯 Customer Segmentation Dashboard")
st.caption("K-Means clustering on 500 customers · demographic & behavioral features")
st.markdown("---")

# ── KPIs ──────────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
kpis = [
    (f"{len(dff):,}", "Total Customers", "#2E4057"),
    (str(n_clusters), "Segments", "#048A81"),
    (f"${dff['Annual_Income'].mean():,.0f}", "Avg Income", "#EF946C"),
    (f"{dff['Spending_Score'].mean():.1f}", "Avg Spending Score", "#54C6EB"),
    (f"${dff['Avg_Order_Value'].mean():.0f}", "Avg Order Value", "#C4A35A"),
]
for col, (val, lbl, color) in zip([k1,k2,k3,k4,k5], kpis):
    col.markdown(f'<div class="kpi"><div class="kpi-val" style="color:{color}">{val}</div><div class="kpi-lbl">{lbl}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Row 1: PCA Scatter + Pie ──────────────────────────────────────────────────
st.markdown('<div class="sec">Segment Overview</div>', unsafe_allow_html=True)
c1, c2 = st.columns([2, 1])

with c1:
    fig = px.scatter(dff, x="PC1", y="PC2", color=dff["Cluster"].astype(str),
                     color_discrete_sequence=PALETTE, template="plotly_white",
                     labels={"color":"Segment"}, opacity=0.82,
                     hover_data={"Age":True,"Annual_Income":True,"Spending_Score":True,"PC1":False,"PC2":False})
    fig.update_traces(marker=dict(size=7, line=dict(width=0.5, color="white")))
    fig.update_layout(title="PCA 2D Projection", font_family="Inter",
                      plot_bgcolor="#F7F9FC", paper_bgcolor="#F7F9FC",
                      margin=dict(l=0,r=0,t=40,b=0), legend_title_text="Segment")
    st.plotly_chart(fig, use_container_width=True)

with c2:
    seg_cnt = dff["Cluster"].value_counts().reset_index()
    seg_cnt.columns = ["Segment","Count"]
    fig2 = px.pie(seg_cnt, names="Segment", values="Count",
                  color_discrete_sequence=PALETTE, hole=0.48, template="plotly_white")
    fig2.update_layout(title="Segment Share", font_family="Inter",
                       plot_bgcolor="#F7F9FC", paper_bgcolor="#F7F9FC",
                       margin=dict(l=0,r=0,t=40,b=0))
    st.plotly_chart(fig2, use_container_width=True)

# ── Row 2: 3D Plot ────────────────────────────────────────────────────────────
st.markdown('<div class="sec">3D Segment Explorer</div>', unsafe_allow_html=True)
fig3d = px.scatter_3d(dff, x="Annual_Income", y="Spending_Score", z="Age",
                      color=dff["Cluster"].astype(str), color_discrete_sequence=PALETTE,
                      opacity=0.8, template="plotly_white",
                      labels={"color":"Segment","Annual_Income":"Income","Spending_Score":"Spending","Age":"Age"},
                      hover_data=["Gender","Region","Avg_Order_Value"])
fig3d.update_traces(marker=dict(size=4))
fig3d.update_layout(font_family="Inter", height=500, title="Income · Spending · Age",
                    paper_bgcolor="#F7F9FC", margin=dict(l=0,r=0,t=40,b=0))
st.plotly_chart(fig3d, use_container_width=True)

# ── Row 3: Elbow + Silhouette ─────────────────────────────────────────────────
st.markdown('<div class="sec">Optimal Cluster Analysis</div>', unsafe_allow_html=True)
e1, e2 = st.columns(2)

with e1:
    fig_e = px.line(x=elbow["k"], y=elbow["inertia"], markers=True, template="plotly_white",
                    labels={"x":"Number of Clusters (k)","y":"Inertia"}, title="Elbow Method")
    fig_e.update_traces(line=dict(color=PALETTE[0], width=3), marker=dict(size=8, color=PALETTE[0]))
    fig_e.update_layout(font_family="Inter", plot_bgcolor="#F7F9FC", paper_bgcolor="#F7F9FC",
                        margin=dict(l=0,r=0,t=40,b=0))
    st.plotly_chart(fig_e, use_container_width=True)

with e2:
    fig_s = px.line(x=elbow["k"], y=elbow["sil"], markers=True, template="plotly_white",
                    labels={"x":"Number of Clusters (k)","y":"Silhouette Score"}, title="Silhouette Score")
    fig_s.update_traces(line=dict(color=PALETTE[1], width=3), marker=dict(size=8, color=PALETTE[1]))
    fig_s.update_layout(font_family="Inter", plot_bgcolor="#F7F9FC", paper_bgcolor="#F7F9FC",
                        margin=dict(l=0,r=0,t=40,b=0))
    st.plotly_chart(fig_s, use_container_width=True)

# ── Row 4: Feature Profiles ───────────────────────────────────────────────────
st.markdown('<div class="sec">Segment Feature Profiles</div>', unsafe_allow_html=True)
fig_bar = go.Figure()
for i, feat in enumerate(FEATURES):
    fig_bar.add_trace(go.Bar(name=feat.replace("_"," "),
                             x=[f"Seg {c}" for c in summary.index],
                             y=summary[feat], marker_color=PALETTE[i]))
fig_bar.update_layout(barmode="group", template="plotly_white", font_family="Inter",
                      plot_bgcolor="#F7F9FC", paper_bgcolor="#F7F9FC", height=380,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02),
                      margin=dict(l=0,r=0,t=40,b=0))
st.plotly_chart(fig_bar, use_container_width=True)

# ── Row 5: Income vs Spending + Summary Table ─────────────────────────────────
st.markdown('<div class="sec">Deep Dive</div>', unsafe_allow_html=True)
d1, d2 = st.columns([2, 1])

with d1:
    fig_iv = px.scatter(dff, x="Annual_Income", y="Spending_Score",
                        color=dff["Cluster"].astype(str), size="Avg_Order_Value",
                        color_discrete_sequence=PALETTE, template="plotly_white",
                        labels={"color":"Segment","Annual_Income":"Annual Income ($)","Spending_Score":"Spending Score"},
                        hover_data=["Age","Gender","Region"], opacity=0.85,
                        title="Income vs Spending Score (bubble = order value)")
    fig_iv.update_layout(font_family="Inter", plot_bgcolor="#F7F9FC", paper_bgcolor="#F7F9FC",
                         margin=dict(l=0,r=0,t=40,b=0))
    st.plotly_chart(fig_iv, use_container_width=True)

with d2:
    st.markdown("**Cluster Summary**")
    st.dataframe(summary.style.background_gradient(cmap="Blues"), use_container_width=True, height=320)

# ── Distribution Histograms ───────────────────────────────────────────────────
st.markdown('<div class="sec">Feature Distributions by Segment</div>', unsafe_allow_html=True)
cols = st.columns(len(FEATURES))
for col, feat in zip(cols, FEATURES):
    fig_h = px.histogram(dff, x=feat, color=dff["Cluster"].astype(str),
                         color_discrete_sequence=PALETTE, barmode="overlay",
                         template="plotly_white", opacity=0.7,
                         labels={"color":"Seg"}, title=feat.replace("_"," "))
    fig_h.update_layout(font_family="Inter", plot_bgcolor="#F7F9FC", paper_bgcolor="#F7F9FC",
                        showlegend=False, margin=dict(l=0,r=0,t=40,b=0), height=260)
    col.plotly_chart(fig_h, use_container_width=True)

# ── Raw Data + Download ───────────────────────────────────────────────────────
with st.expander("📋 Raw Segmented Data"):
    st.dataframe(dff, use_container_width=True)
    st.download_button("⬇️ Download CSV", dff.to_csv(index=False).encode(), "segmented_customers.csv", "text/csv")
