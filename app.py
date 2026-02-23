"""
Banking Analytics Dashboard
Author : Mohammad Noorchenarboo
Stack  : Flask · Pandas · Scikit-learn · Plotly · Bootstrap 5
Port   : 7860 (Hugging Face Spaces / Docker)
"""

import json, warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from flask import Flask, render_template_string, jsonify
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score

warnings.filterwarnings("ignore")
app  = Flask(__name__)
SEED = 42
np.random.seed(SEED)

# ══════════════════════════════════════════════════════════
#  1. SYNTHETIC BANKING DATA  (1 200 customers)
# ══════════════════════════════════════════════════════════
def generate_data(n=1200):
    age      = np.random.randint(22, 70, n)
    income   = np.random.lognormal(10.8, 0.6, n).astype(int)
    balance  = np.random.lognormal(9.5,  0.9, n).astype(int)
    credit   = np.clip(np.random.normal(680, 80, n), 300, 850).astype(int)
    loan_amt = np.random.lognormal(10.2, 0.7, n).astype(int)
    tenure   = np.random.randint(1, 15, n)
    products = np.random.randint(1,  5, n)
    active   = np.random.choice([0, 1], n, p=[0.3, 0.7])
    gender   = np.random.choice(["Male", "Female"], n)
    region   = np.random.choice(
        ["Ontario", "BC", "Quebec", "Alberta", "Other"], n,
        p=[0.35, 0.25, 0.20, 0.12, 0.08])
    segment  = np.random.choice(
        ["Retail", "SME", "Premium", "Corporate"], n,
        p=[0.45, 0.25, 0.20, 0.10])

    # Logistic default probability (realistic relationships)
    p_def = 1 / (1 + np.exp(-(
        -3.5
        + 0.03  * (70  - age)
        - 8e-6  * income
        - 4e-6  * balance
        + 0.01  * (750 - credit)
        + 2e-6  * loan_amt
        - 0.05  * tenure
        - 0.15  * active
    )))
    default = (np.random.rand(n) < p_def).astype(int)

    return pd.DataFrame(dict(
        age=age, income=income, balance=balance, credit_score=credit,
        loan_amount=loan_amt, tenure=tenure, num_products=products,
        is_active=active, gender=gender, region=region,
        segment=segment, default=default
    ))

DF = generate_data()

# ══════════════════════════════════════════════════════════
#  2. ML PIPELINE
# ══════════════════════════════════════════════════════════
FEATURES = ["age", "income", "balance", "credit_score",
            "loan_amount", "tenure", "num_products", "is_active"]

def train_models(df):
    X = df[FEATURES];  y = df["default"]
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=SEED, stratify=y)

    # ── Random Forest ──────────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=120, max_depth=8,
        random_state=SEED, class_weight="balanced")
    rf.fit(Xtr, ytr)

    # ── Gradient Boosting ──────────────────────────────────
    gb = GradientBoostingClassifier(
        n_estimators=100, max_depth=4,
        learning_rate=0.1, random_state=SEED)
    gb.fit(Xtr, ytr)

    # ── Metrics ────────────────────────────────────────────
    rf_pred = rf.predict(Xte);  rf_prob = rf.predict_proba(Xte)[:, 1]
    gb_pred = gb.predict(Xte);  gb_prob = gb.predict_proba(Xte)[:, 1]

    fpr_rf, tpr_rf, _ = roc_curve(yte, rf_prob)
    fpr_gb, tpr_gb, _ = roc_curve(yte, gb_prob)

    # ── KMeans Segmentation ────────────────────────────────
    sc  = StandardScaler()
    Xs  = sc.fit_transform(df[["balance","credit_score","loan_amount","income"]])
    km  = KMeans(n_clusters=4, random_state=SEED, n_init=10)
    df2 = df.copy()
    df2["cluster"] = km.fit_predict(Xs)
    labels = {0: "Low Risk / Low Value",  1: "High Risk / High Value",
              2: "Low Risk / High Value", 3: "High Risk / Low Value"}
    df2["cluster_name"] = df2["cluster"].map(labels)

    return dict(
        rf_acc      = round(accuracy_score(yte, rf_pred), 4),
        gb_acc      = round(accuracy_score(yte, gb_pred), 4),
        fpr_rf      = fpr_rf.tolist(),  tpr_rf=tpr_rf.tolist(),
        fpr_gb      = fpr_gb.tolist(),  tpr_gb=tpr_gb.tolist(),
        auc_rf      = round(auc(fpr_rf, tpr_rf), 4),
        auc_gb      = round(auc(fpr_gb, tpr_gb), 4),
        cm_rf       = confusion_matrix(yte, rf_pred).tolist(),
        cm_gb       = confusion_matrix(yte, gb_pred).tolist(),
        importances = dict(zip(FEATURES, rf.feature_importances_.round(4).tolist())),
        df_cluster  = df2
    )

ML     = train_models(DF)
MONTHS = pd.date_range("2023-01-01", periods=12, freq="MS")
COLORS = px.colors.qualitative.Bold

# ══════════════════════════════════════════════════════════
#  3. CHART HELPERS
# ══════════════════════════════════════════════════════════
def fig_json(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#e2e8f0"),
        margin=dict(l=40, r=20, t=44, b=36)
    )
    return json.loads(fig.to_json())

def chart_trend():
    rates = np.clip(np.random.default_rng(SEED).normal(0.12, 0.03, 12), 0.04, 0.25)
    fig = go.Figure(go.Scatter(
        x=[m.strftime("%b %Y") for m in MONTHS],
        y=(rates * 100).round(2),
        mode="lines+markers",
        line=dict(color="#6366f1", width=3),
        fill="tozeroy", fillcolor="rgba(99,102,241,0.15)",
        marker=dict(size=7, color="#818cf8")
    ))
    fig.update_layout(title="Monthly Default Rate (%)",
                      yaxis_title="%", xaxis_title="")
    return fig_json(fig)

def chart_credit():
    fig = px.histogram(
        DF, x="credit_score", nbins=30, color="default",
        color_discrete_map={0: "#22d3ee", 1: "#f43f5e"},
        labels={"default": "Defaulted", "credit_score": "Credit Score"},
        barmode="overlay",
        title="Credit Score Distribution by Default Status")
    fig.update_traces(opacity=0.75)
    return fig_json(fig)

def chart_scatter():
    s = DF.sample(350, random_state=SEED)
    fig = px.scatter(
        s, x="income", y="balance",
        color="segment", size="loan_amount",
        hover_data=["credit_score", "age"],
        color_discrete_sequence=COLORS,
        title="Income vs Balance by Segment",
        labels={"income": "Annual Income", "balance": "Account Balance"})
    return fig_json(fig)

def chart_region():
    g = (DF.groupby("region")["default"]
           .agg(["sum", "count"])
           .reset_index()
           .rename(columns={"sum": "defaults", "count": "total"}))
    g["rate"] = (g["defaults"] / g["total"] * 100).round(2)
    fig = go.Figure([
        go.Bar(name="Total Customers", x=g["region"], y=g["total"],
               marker_color="#6366f1"),
        go.Scatter(name="Default Rate (%)", x=g["region"], y=g["rate"],
                   mode="lines+markers",
                   marker=dict(size=9, color="#f59e0b"),
                   line=dict(width=2.5, color="#f59e0b"),
                   yaxis="y2")
    ])
    fig.update_layout(
        title="Customers & Default Rate by Region",
        yaxis=dict(title="# Customers"),
        yaxis2=dict(title="Default Rate (%)", overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.14)
    )
    return fig_json(fig)

def chart_roc():
    fig = go.Figure([
        go.Scatter(x=ML["fpr_rf"], y=ML["tpr_rf"],
                   name=f"Random Forest  (AUC = {ML['auc_rf']})",
                   line=dict(color="#6366f1", width=2.5)),
        go.Scatter(x=ML["fpr_gb"], y=ML["tpr_gb"],
                   name=f"Gradient Boost (AUC = {ML['auc_gb']})",
                   line=dict(color="#22d3ee", width=2.5)),
        go.Scatter(x=[0, 1], y=[0, 1], name="Random Baseline",
                   line=dict(dash="dash", color="#475569", width=1.5))
    ])
    fig.update_layout(title="ROC Curve — Loan Default Prediction",
                      xaxis_title="False Positive Rate",
                      yaxis_title="True Positive Rate")
    return fig_json(fig)

def chart_confusion(model="rf"):
    cm   = ML[f"cm_{model}"]
    lbls = ["No Default", "Default"]
    name = "Random Forest" if model == "rf" else "Gradient Boosting"
    fig  = px.imshow(cm, text_auto=True, x=lbls, y=lbls,
                     color_continuous_scale="Purples",
                     title=f"Confusion Matrix — {name}")
    fig.update_layout(xaxis_title="Predicted", yaxis_title="Actual",
                      coloraxis_showscale=False)
    return fig_json(fig)

def chart_importance():
    imp  = ML["importances"]
    feat = sorted(imp, key=imp.get)
    vals = [imp[f] for f in feat]
    fig  = go.Figure(go.Bar(
        x=vals, y=feat, orientation="h",
        marker_color="#6366f1",
        text=[f"{v:.3f}" for v in vals],
        textposition="outside"
    ))
    fig.update_layout(title="Feature Importance — Random Forest",
                      xaxis_title="Importance")
    return fig_json(fig)

def chart_cluster():
    df = ML["df_cluster"].sample(400, random_state=SEED)
    fig = px.scatter(
        df, x="balance", y="credit_score",
        color="cluster_name", size="loan_amount",
        hover_data=["income", "age"],
        color_discrete_sequence=COLORS,
        title="Customer Segments — KMeans (k=4)",
        labels={"balance": "Account Balance", "credit_score": "Credit Score"})
    return fig_json(fig)

def chart_pie():
    vc  = DF["segment"].value_counts()
    fig = px.pie(values=vc.values, names=vc.index,
                 color_discrete_sequence=COLORS,
                 title="Portfolio by Customer Segment",
                 hole=0.45)
    fig.update_traces(textinfo="percent+label")
    return fig_json(fig)

def chart_box():
    fig = px.box(
        DF, x="segment", y="loan_amount", color="default",
        color_discrete_map={0: "#22d3ee", 1: "#f43f5e"},
        labels={"default": "Defaulted", "loan_amount": "Loan Amount ($)"},
        title="Loan Amount Distribution by Segment & Default Status")
    return fig_json(fig)

# ══════════════════════════════════════════════════════════
#  4. KPI SUMMARY
# ══════════════════════════════════════════════════════════
def kpis():
    return dict(
        total_customers = f"{len(DF):,}",
        default_rate    = f"{DF['default'].mean()*100:.1f}%",
        avg_balance     = f"${DF['balance'].mean():,.0f}",
        avg_credit      = f"{DF['credit_score'].mean():.0f}",
        total_loans     = f"${DF['loan_amount'].sum()/1e6:.1f}M",
        active_rate     = f"{DF['is_active'].mean()*100:.1f}%",
        rf_accuracy     = f"{ML['rf_acc']*100:.1f}%",
        auc_rf          = f"{ML['auc_rf']:.3f}"
    )

# ══════════════════════════════════════════════════════════
#  5. HTML TEMPLATE  (Bootstrap 5 + Plotly 2.30)
# ══════════════════════════════════════════════════════════
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>🏦 Banking Analytics Dashboard</title>
<link rel="stylesheet"
  href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css"/>
<link rel="stylesheet"
  href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css"/>
<script src="https://cdn.plot.ly/plotly-2.30.1.min.js"></script>
<style>
:root{
  --bg:#0f172a; --sf:#1e293b; --sf2:#273349;
  --acc:#6366f1; --acc2:#22d3ee; --warn:#f59e0b; --err:#f43f5e;
  --tx:#e2e8f0; --mt:#94a3b8;
}
*{box-sizing:border-box;margin:0;padding:0;}
body{background:var(--bg);color:var(--tx);font-family:'Inter',sans-serif;min-height:100vh;}

/* ── Sidebar ───────────────────────────────── */
#sidebar{
  width:240px;min-height:100vh;background:var(--sf);
  position:fixed;top:0;left:0;z-index:100;
  display:flex;flex-direction:column;
  box-shadow:2px 0 16px rgba(0,0,0,.45);
}
.sb-brand{
  padding:1.35rem 1.2rem;border-bottom:1px solid var(--sf2);
  font-size:1.05rem;font-weight:700;color:#fff;
  display:flex;align-items:center;gap:.55rem;
}
.nav-section{
  font-size:.68rem;text-transform:uppercase;letter-spacing:.09em;
  color:var(--mt);padding:.8rem 1.2rem .2rem;
}
.nav-link{
  color:var(--mt)!important;padding:.52rem 1.2rem;
  border-radius:8px;margin:.1rem .55rem;
  display:flex;align-items:center;gap:.55rem;font-size:.875rem;
  transition:all .18s;
}
.nav-link:hover,.nav-link.active{
  color:#fff!important;background:rgba(99,102,241,.22)!important;
}
.sb-footer{
  margin-top:auto;padding:.9rem 1.2rem;
  border-top:1px solid var(--sf2);font-size:.72rem;color:var(--mt);
}
.sb-footer strong{color:var(--tx);display:block;margin-bottom:.15rem;}

/* ── Main ──────────────────────────────────── */
#main{margin-left:240px;padding:1.6rem 1.8rem;min-height:100vh;}
.page{display:none;} .page.active{display:block;}

/* ── Top bar ───────────────────────────────── */
.topbar{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:1.4rem;}
.topbar h4{font-weight:700;font-size:1.18rem;margin-bottom:.15rem;}
.topbar small{color:var(--mt);}
.badge-live{
  background:rgba(34,211,238,.13);color:var(--acc2);
  border:1px solid var(--acc2);border-radius:20px;
  padding:.22rem .7rem;font-size:.74rem;white-space:nowrap;
}

/* ── KPI grid ──────────────────────────────── */
.kpi-grid{
  display:grid;
  grid-template-columns:repeat(auto-fill,minmax(175px,1fr));
  gap:.9rem;margin-bottom:1.3rem;
}
.kpi-card{
  background:var(--sf);border-radius:14px;padding:1.1rem 1.15rem;
  border:1px solid var(--sf2);
  transition:transform .2s,box-shadow .2s;
}
.kpi-card:hover{transform:translateY(-3px);box-shadow:0 8px 26px rgba(0,0,0,.35);}
.kpi-icon{font-size:1.45rem;float:right;opacity:.55;}
.kpi-label{font-size:.72rem;color:var(--mt);text-transform:uppercase;
           letter-spacing:.06em;margin-bottom:.3rem;}
.kpi-value{font-size:1.65rem;font-weight:800;line-height:1.1;}

/* ── Chart cards ───────────────────────────── */
.chart-card{
  background:var(--sf);border-radius:14px;padding:1rem 1rem .6rem;
  border:1px solid var(--sf2);margin-bottom:1.1rem;height:380px;
}
.chart-card.tall{height:445px;}
.chart-label{font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;
             color:var(--mt);margin-bottom:.35rem;}
.chart-body{height:calc(100% - 22px);}

/* ── Data table ────────────────────────────── */
.data-table{width:100%;border-collapse:collapse;font-size:.84rem;}
.data-table th{
  background:var(--sf2);color:var(--mt);font-size:.72rem;
  text-transform:uppercase;padding:.6rem 1rem;text-align:left;
  position:sticky;top:0;
}
.data-table td{padding:.52rem 1rem;border-bottom:1px solid var(--sf2);}
.data-table tr:hover td{background:rgba(99,102,241,.07);}
.badge-default{background:rgba(244,63,94,.14);color:#f43f5e;
               border-radius:20px;padding:.12rem .52rem;font-size:.74rem;}
.badge-ok{background:rgba(34,211,238,.12);color:var(--acc2);
          border-radius:20px;padding:.12rem .52rem;font-size:.74rem;}

/* ── Scrollbar ─────────────────────────────── */
::-webkit-scrollbar{width:5px;}
::-webkit-scrollbar-track{background:var(--bg);}
::-webkit-scrollbar-thumb{background:var(--sf2);border-radius:3px;}
</style>
</head>
<body>

<!-- ═══════ SIDEBAR ═══════ -->
<nav id="sidebar">
  <div class="sb-brand">
    <i class="bi bi-bank2" style="color:var(--acc)"></i> BankAnalytics
  </div>

  <div class="nav-section">Overview</div>
  <a class="nav-link active" href="#" onclick="return nav('overview',this)">
    <i class="bi bi-grid-1x2"></i> Dashboard
  </a>

  <div class="nav-section">Analytics</div>
  <a class="nav-link" href="#" onclick="return nav('portfolio',this)">
    <i class="bi bi-pie-chart"></i> Portfolio
  </a>
  <a class="nav-link" href="#" onclick="return nav('risk',this)">
    <i class="bi bi-exclamation-triangle"></i> Risk Analysis
  </a>
  <a class="nav-link" href="#" onclick="return nav('geography',this)">
    <i class="bi bi-geo-alt"></i> Geography
  </a>

  <div class="nav-section">Machine Learning</div>
  <a class="nav-link" href="#" onclick="return nav('models',this)">
    <i class="bi bi-cpu"></i> Prediction Models
  </a>
  <a class="nav-link" href="#" onclick="return nav('segments',this)">
    <i class="bi bi-people"></i> Segmentation
  </a>

  <div class="nav-section">Data</div>
  <a class="nav-link" href="#" onclick="return nav('explorer',this)">
    <i class="bi bi-table"></i> Data Explorer
  </a>

  <div class="sb-footer">
    <strong>Mohammad Noorchenarboo</strong>
    Data Scientist · Ontario, CA
  </div>
</nav>

<!-- ═══════ MAIN ═══════ -->
<main id="main">

  <!-- ── OVERVIEW ── -->
  <div id="page-overview" class="page active">
    <div class="topbar">
      <div>
        <h4><i class="bi bi-grid-1x2 me-2" style="color:var(--acc)"></i>Dashboard Overview</h4>
        <small>Synthetic banking portfolio · 1,200 customers · Feb 2026</small>
      </div>
      <span class="badge-live">
        <i class="bi bi-circle-fill me-1" style="font-size:.58rem"></i>Live Demo
      </span>
    </div>

    <div class="kpi-grid">
      <div class="kpi-card">
        <i class="bi bi-people-fill kpi-icon" style="color:var(--acc)"></i>
        <div class="kpi-label">Total Customers</div>
        <div class="kpi-value" style="color:var(--acc)">{{ kpi.total_customers }}</div>
      </div>
      <div class="kpi-card">
        <i class="bi bi-exclamation-octagon kpi-icon" style="color:var(--err)"></i>
        <div class="kpi-label">Default Rate</div>
        <div class="kpi-value" style="color:var(--err)">{{ kpi.default_rate }}</div>
      </div>
      <div class="kpi-card">
        <i class="bi bi-currency-dollar kpi-icon" style="color:var(--acc2)"></i>
        <div class="kpi-label">Avg Balance</div>
        <div class="kpi-value" style="color:var(--acc2)">{{ kpi.avg_balance }}</div>
      </div>
      <div class="kpi-card">
        <i class="bi bi-graph-up kpi-icon" style="color:var(--warn)"></i>
        <div class="kpi-label">Avg Credit Score</div>
        <div class="kpi-value" style="color:var(--warn)">{{ kpi.avg_credit }}</div>
      </div>
      <div class="kpi-card">
        <i class="bi bi-bank kpi-icon" style="color:#a78bfa"></i>
        <div class="kpi-label">Total Loans</div>
        <div class="kpi-value" style="color:#a78bfa">{{ kpi.total_loans }}</div>
      </div>
      <div class="kpi-card">
        <i class="bi bi-person-check kpi-icon" style="color:#34d399"></i>
        <div class="kpi-label">Active Customers</div>
        <div class="kpi-value" style="color:#34d399">{{ kpi.active_rate }}</div>
      </div>
      <div class="kpi-card">
        <i class="bi bi-cpu kpi-icon" style="color:var(--acc)"></i>
        <div class="kpi-label">RF Accuracy</div>
        <div class="kpi-value" style="color:var(--acc)">{{ kpi.rf_accuracy }}</div>
      </div>
      <div class="kpi-card">
        <i class="bi bi-bar-chart kpi-icon" style="color:var(--acc2)"></i>
        <div class="kpi-label">ROC-AUC (RF)</div>
        <div class="kpi-value" style="color:var(--acc2)">{{ kpi.auc_rf }}</div>
      </div>
    </div>

    <div class="row g-3">
      <div class="col-lg-8"><div class="chart-card">
        <div class="chart-label"><i class="bi bi-graph-up me-1"></i>Monthly Default Rate Trend</div>
        <div class="chart-body" id="ov-trend"></div>
      </div></div>
      <div class="col-lg-4"><div class="chart-card">
        <div class="chart-label"><i class="bi bi-pie-chart me-1"></i>Portfolio by Segment</div>
        <div class="chart-body" id="ov-pie"></div>
      </div></div>
      <div class="col-lg-6"><div class="chart-card">
        <div class="chart-label"><i class="bi bi-bar-chart me-1"></i>Credit Score Distribution</div>
        <div class="chart-body" id="ov-credit"></div>
      </div></div>
      <div class="col-lg-6"><div class="chart-card">
        <div class="chart-label"><i class="bi bi-scatter-chart me-1"></i>Income vs Balance by Segment</div>
        <div class="chart-body" id="ov-scatter"></div>
      </div></div>
    </div>
  </div>

  <!-- ── PORTFOLIO ── -->
  <div id="page-portfolio" class="page">
    <div class="topbar">
      <div><h4><i class="bi bi-pie-chart me-2" style="color:var(--acc)"></i>Portfolio Analysis</h4></div>
    </div>
    <div class="row g-3">
      <div class="col-lg-6"><div class="chart-card tall">
        <div class="chart-label">Loan Amount by Segment &amp; Default Status</div>
        <div class="chart-body" id="pf-box"></div>
      </div></div>
      <div class="col-lg-6"><div class="chart-card tall">
        <div class="chart-label">Segment Share (Donut)</div>
        <div class="chart-body" id="pf-pie"></div>
      </div></div>
      <div class="col-12"><div class="chart-card">
        <div class="chart-label">Income vs Balance</div>
        <div class="chart-body" id="pf-scatter"></div>
      </div></div>
    </div>
  </div>

  <!-- ── RISK ANALYSIS ── -->
  <div id="page-risk" class="page">
    <div class="topbar">
      <div><h4><i class="bi bi-exclamation-triangle me-2" style="color:var(--err)"></i>Risk Analysis</h4></div>
    </div>
    <div class="row g-3">
      <div class="col-lg-8"><div class="chart-card tall">
        <div class="chart-label">Credit Score Distribution by Default Status</div>
        <div class="chart-body" id="ri-credit"></div>
      </div></div>
      <div class="col-lg-4"><div class="chart-card tall">
        <div class="chart-label">Monthly Default Rate Trend</div>
        <div class="chart-body" id="ri-trend"></div>
      </div></div>
      <div class="col-12"><div class="chart-card">
        <div class="chart-label">Loan Distribution by Segment &amp; Default</div>
        <div class="chart-body" id="ri-box"></div>
      </div></div>
    </div>
  </div>

  <!-- ── GEOGRAPHY ── -->
  <div id="page-geography" class="page">
    <div class="topbar">
      <div><h4><i class="bi bi-geo-alt me-2" style="color:var(--acc)"></i>Geographic Analysis</h4></div>
    </div>
    <div class="row g-3">
      <div class="col-12"><div class="chart-card tall">
        <div class="chart-label">Customers &amp; Default Rate by Province</div>
        <div class="chart-body" id="ge-region"></div>
      </div></div>
      <div class="col-lg-6"><div class="chart-card">
        <div class="chart-label">Income vs Balance</div>
        <div class="chart-body" id="ge-scatter"></div>
      </div></div>
      <div class="col-lg-6"><div class="chart-card">
        <div class="chart-label">Credit Score Distribution</div>
        <div class="chart-body" id="ge-credit"></div>
      </div></div>
    </div>
  </div>

  <!-- ── ML MODELS ── -->
  <div id="page-models" class="page">
    <div class="topbar">
      <div>
        <h4><i class="bi bi-cpu me-2" style="color:var(--acc)"></i>Prediction Models</h4>
        <small>Random Forest · Gradient Boosting · Evaluated on 25% hold-out</small>
      </div>
      <div class="d-flex gap-3 align-items-center">
        <span style="color:var(--acc2);font-size:.84rem;">
          <i class="bi bi-check-circle me-1"></i>RF Acc: {{ kpi.rf_accuracy }}
        </span>
        <span style="color:#a78bfa;font-size:.84rem;">
          <i class="bi bi-check-circle me-1"></i>AUC: {{ kpi.auc_rf }}
        </span>
      </div>
    </div>
    <div class="row g-3">
      <div class="col-lg-7"><div class="chart-card tall">
        <div class="chart-label">ROC Curve — RF vs Gradient Boosting</div>
        <div class="chart-body" id="ml-roc"></div>
      </div></div>
      <div class="col-lg-5"><div class="chart-card tall">
        <div class="chart-label">Feature Importance — Random Forest</div>
        <div class="chart-body" id="ml-imp"></div>
      </div></div>
      <div class="col-lg-6"><div class="chart-card">
        <div class="chart-label">Confusion Matrix — Random Forest</div>
        <div class="chart-body" id="ml-cm-rf"></div>
      </div></div>
      <div class="col-lg-6"><div class="chart-card">
        <div class="chart-label">Confusion Matrix — Gradient Boosting</div>
        <div class="chart-body" id="ml-cm-gb"></div>
      </div></div>
    </div>
  </div>

  <!-- ── SEGMENTATION ── -->
  <div id="page-segments" class="page">
    <div class="topbar">
      <div><h4><i class="bi bi-people me-2" style="color:var(--acc)"></i>Customer Segmentation</h4></div>
    </div>
    <div class="row g-3">
      <div class="col-12"><div class="chart-card tall">
        <div class="chart-label">KMeans Clusters (k=4) — Balance × Credit Score</div>
        <div class="chart-body" id="sg-cluster"></div>
      </div></div>
      <div class="col-lg-5"><div class="chart-card">
        <div class="chart-label">Segment Share</div>
        <div class="chart-body" id="sg-pie"></div>
      </div></div>
      <div class="col-lg-7"><div class="chart-card">
        <div class="chart-label">Income vs Balance</div>
        <div class="chart-body" id="sg-scatter"></div>
      </div></div>
    </div>
  </div>

  <!-- ── DATA EXPLORER ── -->
  <div id="page-explorer" class="page">
    <div class="topbar">
      <div>
        <h4><i class="bi bi-table me-2" style="color:var(--acc)"></i>Data Explorer</h4>
        <small>First 50 records — synthetically generated, no real customer data</small>
      </div>
    </div>
    <div style="background:var(--sf);border-radius:14px;padding:1rem;
                border:1px solid var(--sf2);overflow:auto;max-height:73vh;">
      <table class="data-table">
        <thead>
          <tr>{% for c in table_cols %}<th>{{ c }}</th>{% endfor %}</tr>
        </thead>
        <tbody>
          {% for row in table_rows %}
          <tr>
            {% for val in row %}
            <td>
              {% if loop.index == table_cols|length %}
                {% if val == 1 %}<span class="badge-default">Default</span>
                {% else %}<span class="badge-ok">OK</span>{% endif %}
              {% else %}{{ val }}{% endif %}
            </td>
            {% endfor %}
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>
  </div>

</main><!-- /main -->

<script>
const C   = {{ charts|safe }};   // all chart data injected by Flask
const cfg = {responsive:true, displayModeBar:false};
const done = new Set();

function plot(id, key) {
  const el = document.getElementById(id);
  if (!el || !C[key]) return;
  Plotly.newPlot(id, C[key].data, C[key].layout, cfg);
}

// Page render map — each page lists its (divId, chartKey) pairs
const PAGES = {
  overview  : [['ov-trend','trend'],['ov-pie','pie'],['ov-credit','credit'],['ov-scatter','scatter']],
  portfolio : [['pf-box','box'],['pf-pie','pie'],['pf-scatter','scatter']],
  risk      : [['ri-credit','credit'],['ri-trend','trend'],['ri-box','box']],
  geography : [['ge-region','region'],['ge-scatter','scatter'],['ge-credit','credit']],
  models    : [['ml-roc','roc'],['ml-imp','importance'],['ml-cm-rf','cm_rf'],['ml-cm-gb','cm_gb']],
  segments  : [['sg-cluster','cluster'],['sg-pie','pie'],['sg-scatter','scatter']],
  explorer  : []
};

function renderPage(name) {
  if (done.has(name)) return;
  done.add(name);
  (PAGES[name] || []).forEach(([id, key]) => plot(id, key));
}

function nav(name, el) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-link').forEach(a => a.classList.remove('active'));
  document.getElementById('page-' + name).classList.add('active');
  el.classList.add('active');
  setTimeout(() => renderPage(name), 30);
  return false;
}

// Render initial page
renderPage('overview');

// Relay on resize
window.addEventListener('resize', () => {
  document.querySelectorAll('[id]').forEach(el => {
    if (el._fullLayout) Plotly.relayout(el.id, {autosize: true});
  });
});
</script>
</body>
</html>
"""

# ══════════════════════════════════════════════════════════
#  6. ROUTES
# ══════════════════════════════════════════════════════════
@app.route("/")
def index():
    # Build all charts once per request
    charts = dict(
        trend      = chart_trend(),
        credit     = chart_credit(),
        scatter    = chart_scatter(),
        region     = chart_region(),
        roc        = chart_roc(),
        cm_rf      = chart_confusion("rf"),
        cm_gb      = chart_confusion("gb"),
        importance = chart_importance(),
        cluster    = chart_cluster(),
        pie        = chart_pie(),
        box        = chart_box()
    )

    # Data Explorer — first 50 rows, formatted
    cols = ["age","income","balance","credit_score",
            "loan_amount","tenure","num_products",
            "is_active","segment","region","default"]
    disp = DF[cols].head(50).copy()
    disp["income"]      = disp["income"].apply(lambda x: f"${x:,}")
    disp["balance"]     = disp["balance"].apply(lambda x: f"${x:,}")
    disp["loan_amount"] = disp["loan_amount"].apply(lambda x: f"${x:,}")

    return render_template_string(
        HTML,
        kpi        = kpis(),
        charts     = json.dumps(charts),
        table_cols = [c.replace("_", " ").title() for c in cols],
        table_rows = disp.values.tolist()
    )


@app.route("/api/kpis")
def api_kpis():
    """JSON endpoint — KPI summary."""
    return jsonify(kpis())


@app.route("/api/charts/<name>")                   # ← FIXED: was <n>
def api_chart(name):
    """JSON endpoint — individual chart by name."""
    dispatch = {
        "trend"      : chart_trend,
        "credit"     : chart_credit,
        "scatter"    : chart_scatter,
        "region"     : chart_region,
        "roc"        : chart_roc,
        "importance" : chart_importance,
        "cluster"    : chart_cluster,
        "pie"        : chart_pie,
        "box"        : chart_box,
        "cm_rf"      : lambda: chart_confusion("rf"),
        "cm_gb"      : lambda: chart_confusion("gb"),
    }
    fn = dispatch.get(name)
    return jsonify(fn()) if fn else (jsonify({"error": "chart not found"}), 404)


# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)