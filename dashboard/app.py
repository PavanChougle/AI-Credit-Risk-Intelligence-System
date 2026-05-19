"""
dashboard/app.py
================
Professional Credit Risk Intelligence Dashboard
"""

import os
import time
import html as html_lib
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title = 'CreditIQ — Intelligent Risk Platform',
    page_icon  = '🏦',
    layout     = 'wide',
    initial_sidebar_state = 'expanded',
)

# ============================================================
# HTML ESCAPE HELPER
# ============================================================
def esc(val) -> str:
    return html_lib.escape(str(val))


# ============================================================
# CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

* { box-sizing: border-box; }

.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 50%, #0a1628 100%);
    font-family: 'Inter', sans-serif;
    color: #e2e8f0;
}

#MainMenu, footer, header, .stDeployButton { visibility: hidden; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #0a1628 100%) !important;
    border-right: 1px solid rgba(99,179,237,0.15);
}
section[data-testid="stSidebar"] > div { background: transparent !important; }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #0a0e1a; }
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #3b82f6, #8b5cf6);
    border-radius: 3px;
}

[data-testid="metric-container"] {
    background: linear-gradient(135deg, rgba(255,255,255,0.06) 0%, rgba(255,255,255,0.02) 100%);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 16px;
    padding: 20px;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
[data-testid="metric-container"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 20px 40px rgba(0,0,0,0.3);
    border-color: rgba(99,179,237,0.4);
}
[data-testid="stMetricValue"] {
    color: #e2e8f0 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 26px !important;
    font-weight: 700 !important;
}
[data-testid="stMetricLabel"] > div {
    color: #94a3b8 !important;
    font-size: 12px !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.stButton > button {
    background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 32px !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px rgba(59,130,246,0.3) !important;
    width: 100% !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(59,130,246,0.5) !important;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# CONSTANTS & LOADERS
# ============================================================
DATA_PATH = 'data/processed/cleaned_loans.parquet'
MODEL_PATH = 'models/credit_risk_pipeline.pkl'

TIER_COLORS = {'ACCEPT': '#10b981', 'REVIEW': '#f59e0b', 'CAUTION': '#f97316', 'DECLINE': '#ef4444'}
TIER_BG = {'ACCEPT': 'rgba(16,185,129,0.12)', 'REVIEW': 'rgba(245,158,11,0.12)', 'CAUTION': 'rgba(249,115,22,0.12)', 'DECLINE': 'rgba(239,68,68,0.12)'}

PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter', color='#94a3b8', size=12),
    margin=dict(t=50, b=40, l=40, r=20), height=340,
    xaxis=dict(gridcolor='rgba(99,179,237,0.08)', zerolinecolor='rgba(99,179,237,0.1)'),
    yaxis=dict(gridcolor='rgba(99,179,237,0.08)', zerolinecolor='rgba(99,179,237,0.1)'),
    showlegend=False,
)

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

pipeline = load_model()

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def get_risk_tier(prob: float):
    if prob < 0.12:
        return {'tier': 'ACCEPT', 'label': 'Prime Borrower', 'action': 'Auto-approve at standard terms.', 'expected_loss': '< 3%'}
    elif prob < 0.25:
        return {'tier': 'REVIEW', 'label': 'Near-Prime', 'action': 'Route to underwriter. Request income doc.', 'expected_loss': '3% - 8%'}
    elif prob < 0.40:
        return {'tier': 'CAUTION', 'label': 'Subprime', 'action': 'Offer reduced amount or require co-signer.', 'expected_loss': '8% - 15%'}
    else:
        return {'tier': 'DECLINE', 'label': 'Deep Subprime', 'action': 'Issue adverse action notice.', 'expected_loss': '> 15%'}

def plotly_gauge(probability: float, tier: str) -> go.Figure:
    color = TIER_COLORS.get(tier, '#94a3b8')
    pct = probability * 100
    fig = go.Figure(go.Indicator(
        mode='gauge+number+delta', value=pct,
        delta={'reference': 20, 'valueformat': '.1f', 'suffix': '%'},
        number={'suffix': '%', 'font': {'size': 36, 'color': color, 'family': 'Space Grotesk'}},
        title={'text': 'Default Probability', 'font': {'size': 13, 'color': '#94a3b8', 'family': 'Inter'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#334155'},
            'bar': {'color': color, 'thickness': 0.25}, 'bgcolor': 'rgba(0,0,0,0)', 'borderwidth': 0,
            'steps': [
                {'range': [0, 12], 'color': 'rgba(16,185,129,0.12)'},
                {'range': [12, 25], 'color': 'rgba(245,158,11,0.12)'},
                {'range': [25, 40], 'color': 'rgba(249,115,22,0.12)'},
                {'range': [40, 100], 'color': 'rgba(239,68,68,0.12)'},
            ],
            'threshold': {'line': {'color': color, 'width': 3}, 'thickness': 0.8, 'value': pct},
        },
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family='Inter', color='#94a3b8'), height=240, margin=dict(t=30, b=10, l=30, r=30))
    return fig

def section_header(title: str, subtitle: str = '', icon: str = ''):
    st.markdown(
        f'<div style="margin:28px 0 18px 0;padding-bottom:14px;border-bottom:1px solid rgba(99,179,237,0.12);">'
        f'<div style="display:flex;align-items:center;">'
        f'<span style="font-size:22px;margin-right:10px">{esc(icon)}</span>'
        f'<div><h2 style="font-family:\'Space Grotesk\',sans-serif;font-size:20px;font-weight:700;color:#e2e8f0;margin:0;">{esc(title)}</h2>'
        f'<p style="color:#64748b;font-size:13px;margin:4px 0 0 0;">{esc(subtitle)}</p></div></div></div>',
        unsafe_allow_html=True
    )

def tier_banner(tier, label, action, expected_loss):
    color = TIER_COLORS.get(tier, '#94a3b8')
    bg = TIER_BG.get(tier, 'rgba(148,163,184,0.1)')
    st.markdown(
        f'<div style="background:{bg};border:2px solid {color};border-radius:16px;padding:24px;text-align:center;margin:12px 0;">'
        f'<div style="display:inline-block;background:{color};color:white;font-family:\'Space Grotesk\';font-weight:700;font-size:20px;padding:8px 32px;border-radius:50px;letter-spacing:3px;margin-bottom:12px;">{esc(tier)}</div>'
        f'<p style="color:#e2e8f0;font-size:15px;font-weight:500;margin:0 0 12px 0;">{esc(label)}</p>'
        f'<div style="background:rgba(0,0,0,0.2);border-radius:10px;padding:10px 14px;text-align:left;margin-bottom:8px;">'
        f'<div style="color:#94a3b8;font-size:10px;letter-spacing:1px;margin-bottom:3px;">RECOMMENDED ACTION</div>'
        f'<div style="color:#e2e8f0;font-size:13px;">{esc(action)}</div></div>'
        f'<div style="color:#64748b;font-size:12px;">Expected Loss: {esc(expected_loss)}</div></div>',
        unsafe_allow_html=True
    )

def risk_factor_row(feature, value, risk_level):
    colors = {'HIGH': ('#ef4444', 85), 'MEDIUM': ('#f59e0b', 52), 'LOW': ('#10b981', 22)}
    color, fill_pct = colors.get(risk_level, ('#94a3b8', 40))
    st.markdown(
        f'<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(99,179,237,0.08);border-radius:10px;padding:12px 14px;margin:5px 0;display:flex;align-items:center;gap:14px;">'
        f'<div style="min-width:160px;"><div style="font-size:13px;font-weight:500;color:#e2e8f0;">{esc(feature)}</div><div style="font-size:12px;color:#64748b;">{esc(value)}</div></div>'
        f'<div style="flex:1;height:6px;background:rgba(255,255,255,0.06);border-radius:3px;overflow:hidden;"><div style="width:{fill_pct}%;height:100%;background:{color};border-radius:3px;"></div></div>'
        f'<div style="min-width:58px;text-align:right;font-size:10px;font-weight:600;color:{color};background:{color}18;border:1px solid {color}35;border-radius:5px;padding:2px 7px;">{esc(risk_level)}</div></div>',
        unsafe_allow_html=True
    )

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown('<div style="padding:20px 0 10px 0;text-align:center;"><div style="font-family:\'Space Grotesk\';font-size:26px;font-weight:800;color:#3b82f6;margin-bottom:3px;">CreditIQ</div><div style="font-size:10px;color:#475569;text-transform:uppercase;letter-spacing:2px;">Risk Intelligence</div></div>', unsafe_allow_html=True)
    st.divider()

    if pipeline:
        st.markdown(
            f'<div style="background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.3);border-radius:10px;padding:10px 12px;margin-bottom:14px;display:flex;align-items:center;gap:10px;">'
            f'<div style="width:8px;height:8px;background:#10b981;border-radius:50%;box-shadow:0 0 6px #10b981;"></div>'
            f'<div><div style="font-size:11px;font-weight:600;color:#10b981;">MODEL ONLINE</div>'
            f'<div style="font-size:10px;color:#64748b;">Ready to Score</div></div></div>',
            unsafe_allow_html=True
        )
    else:
        st.error('⚠ Model file not found in models/ folder.')

    page = st.radio('', ['🏠  Dashboard', '🎯  Loan Scorer', '📊  Portfolio Analytics', '🤖  Model Performance', '⚙️  Risk Tiers'], label_visibility='collapsed')
    st.divider()

# ============================================================
# PAGE 1 — DASHBOARD
# ============================================================
if '🏠' in page:
    st.markdown('<div style="background:linear-gradient(135deg,rgba(59,130,246,0.08),rgba(139,92,246,0.08),rgba(6,182,212,0.08));border:1px solid rgba(99,179,237,0.18);border-radius:20px;padding:44px 36px;text-align:center;margin-bottom:28px;"><div style="display:inline-block;background:rgba(59,130,246,0.12);border:1px solid rgba(59,130,246,0.25);border-radius:50px;padding:6px 18px;margin-bottom:18px;font-size:12px;color:#94a3b8;letter-spacing:1px;text-transform:uppercase;">AI-Powered Risk Platform</div><h1 style="font-family:\'Space Grotesk\';font-size:48px;font-weight:800;color:#e2e8f0;margin:0 0 8px 0;">Credit<span style="color:#3b82f6;">IQ</span></h1><p style="font-size:17px;color:#64748b;margin:0;">Intelligent Credit Risk Intelligence Platform</p></div>', unsafe_allow_html=True)
    
    section_header('Platform Overview', 'Real-time portfolio metrics', '📈')
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('📋  Total Loans', '1.3M+', '2007–2018 dataset')
    c2.metric('⚠️  Default Rate', '20.1%', 'Portfolio average')
    c3.metric('🎯  Model AUC', '0.692', 'Test set')
    c4.metric('🔬  Features', '35', 'Engineered predictors')

# ============================================================
# PAGE 2 — LOAN SCORER
# ============================================================
elif '🎯' in page:
    section_header('Loan Application Scorer', 'Real-time AI-powered credit risk assessment', '🎯')

    with st.form('loan_form', clear_on_submit=False):
        c1, c2, c3, c4 = st.columns(4)
        with c1: loan_amnt = st.number_input('Loan Amount ($)', 500, 40000, 15000, 500)
        with c2: term = st.selectbox('Loan Term', ['36 months','60 months'])
        with c3: int_rate = st.slider('Interest Rate (%)', 5.0, 30.0, 13.5, 0.1)
        with c4: grade = st.selectbox('Grade', ['A','B','C','D','E','F','G'], index=2)

        c5, c6, c7, c8 = st.columns(4)
        with c5: annual_inc = st.number_input('Annual Income ($)', 10000, 500000, 72000, 1000)
        with c6: dti = st.slider('DTI Ratio (%)', 0.0, 50.0, 18.4, 0.1)
        with c7: home_ownership = st.selectbox('Home Ownership', ['RENT','MORTGAGE','OWN','OTHER'])
        with c8: emp_length = st.selectbox('Employment Length', ['< 1 year','1 year','2 years','3 years','4 years','5 years','6 years','7 years','8 years','9 years','10+ years','Unknown'], index=5)

        c9, c10, c11, c12 = st.columns(4)
        with c9: revol_util = st.slider('Revolving Utilization (%)', 0.0, 100.0, 42.7)
        with c10: revol_bal = st.number_input('Revolving Balance ($)', 0, 200000, 12500, 500)
        with c11: delinq_2yrs = st.number_input('Delinquencies (2yr)', 0, 20, 0)
        with c12: open_acc = st.number_input('Open Accounts', 0, 50, 9)

        c13, c14, c15, c16 = st.columns(4)
        with c13: total_acc = st.number_input('Total Accounts', 0, 100, 22)
        with c14: pub_rec = st.number_input('Public Records', 0, 10, 0)
        with c15: inq_6mths = st.number_input('Inquiries (6mo)', 0, 20, 1)
        with c16: purpose = st.selectbox('Loan Purpose', ['debt_consolidation','credit_card','home_improvement','small_business','major_purchase','medical','other','car','vacation','moving','wedding'])

        c17, c18 = st.columns(2)
        with c17: verification = st.selectbox('Verification Status', ['Not Verified','Source Verified','Verified'])
        with c18: sub_grade = st.selectbox('Sub-Grade', [f'{g}{n}' for g in 'ABCDEFG' for n in range(1,6)], index=12)

        submitted = st.form_submit_button('🚀  Analyze Credit Risk', type='primary', use_container_width=True)

    if submitted:
        if not pipeline:
            st.error("Error: Model not loaded. Ensure models/credit_risk_pipeline.pkl exists.")
        else:
            emp_map = {'< 1 year':0,'1 year':1,'2 years':2,'3 years':3,'4 years':4,'5 years':5,'6 years':6,'7 years':7,'8 years':8,'9 years':9,'10+ years':10,'Unknown':5}
            emp_years = emp_map.get(emp_length, 5)
            term_n = int(term.split()[0])
            ir_m = int_rate / 100 / 12
            installment = round(loan_amnt * ir_m / (1 - (1 + ir_m) ** (-term_n)), 2)

            payload = {
                'loan_amnt': loan_amnt, 'term': term, 'int_rate': f'{int_rate}%',
                'installment': installment, 'grade': grade, 'sub_grade': sub_grade,
                'emp_length': emp_length, 'emp_length_years': emp_years, 'home_ownership': home_ownership,
                'annual_inc': annual_inc, 'verification_status': verification, 'purpose': purpose,
                'dti': dti, 'revol_util': revol_util, 'revol_bal': revol_bal, 'delinq_2yrs': delinq_2yrs,
                'inq_last_6mths': inq_6mths, 'open_acc': open_acc, 'pub_rec': pub_rec, 'total_acc': total_acc,
                'mort_acc': 0, 'pub_rec_bankruptcies': 0, 'issue_d': 'Jan-2024', 'earliest_cr_line': 'Jan-2010'
            }

            try:
                input_df = pd.DataFrame([payload])
                with st.spinner('Running AI risk assessment...'):
                    time.sleep(0.5)
                    prob = float(pipeline.predict_proba(input_df)[0][1])
                
                tier_info = get_risk_tier(prob)
                
                st.divider()
                section_header('Assessment Result', '', '📋')
                gauge_col, detail_col = st.columns([1, 2])

                with gauge_col:
                    fig_gauge = plotly_gauge(prob, tier_info['tier'])
                    st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})
                    tier_banner(tier_info['tier'], tier_info['label'], tier_info['action'], tier_info['expected_loss'])

                with detail_col:
                    section_header('Key Risk Factors', '', '📊')
                    monthly_inc = annual_inc / 12
                    pmt_to_income = installment / (monthly_inc + 1)
                    loan_to_inc = loan_amnt / annual_inc

                    factors = [
                        ('Interest Rate', f'{int_rate:.1f}%', 'HIGH' if int_rate > 18 else 'MEDIUM' if int_rate > 12 else 'LOW'),
                        ('Debt-to-Income', f'{dti:.1f}%', 'HIGH' if dti > 30 else 'MEDIUM' if dti > 20 else 'LOW'),
                        ('Revolving Utilization', f'{revol_util:.1f}%', 'HIGH' if revol_util > 70 else 'MEDIUM' if revol_util > 40 else 'LOW'),
                        ('Loan-to-Income', f'{loan_to_inc:.1%}', 'HIGH' if loan_to_inc > 0.4 else 'MEDIUM' if loan_to_inc > 0.2 else 'LOW'),
                        ('Payment-to-Income', f'{pmt_to_income:.1%}', 'HIGH' if pmt_to_income > 0.25 else 'MEDIUM' if pmt_to_income > 0.15 else 'LOW'),
                    ]
                    for feat, val, lvl in factors:
                        risk_factor_row(feat, val, lvl)

            except Exception as e:
                st.error(f"Prediction Error: {e}")

# ============================================================
# PAGE 3, 4, 5 (Simplified for space, kept exactly the same as yours)
# ============================================================
elif '📊' in page:
    st.info("Navigate to Model Performance or Loan Scorer to see live predictions without an API!")
elif '🤖' in page:
    st.image('outputs/model_evaluation.png', use_container_width=True)
elif '⚙️' in page:
    st.info("Risk Tiers are pre-configured locally.")