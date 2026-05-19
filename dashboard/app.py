"""
dashboard/app.py
================
Professional Credit Risk Intelligence Dashboard
Run: streamlit run dashboard/app.py
"""

import os
import sys
import time
import html as html_lib
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import requests
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
# HTML ESCAPE HELPER  (fixes raw-tag rendering bug)
# ============================================================
def esc(val) -> str:
    """Escape a value for safe HTML interpolation."""
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

# MainMenu, footer, header, .stDeployButton { visibility: hidden; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #0a1628 100%) !important;
    border-right: 1px solid rgba(99,179,237,0.15);
}

section[data-testid="stSidebar"] > div {
    background: transparent !important;
}

[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #0a0e1a; }
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #3b82f6, #8b5cf6);
    border-radius: 3px;
}

[data-testid="metric-container"] {
    background: linear-gradient(135deg,
        rgba(255,255,255,0.06) 0%,
        rgba(255,255,255,0.02) 100%);
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

div[data-baseweb="input"] input,
div[data-baseweb="select"] div {
    background: rgba(255,255,255,0.05) !important;
    border-color: rgba(99,179,237,0.2) !important;
    color: #e2e8f0 !important;
    border-radius: 10px !important;
}

.stSlider [data-testid="stSlider"] > div > div {
    background: linear-gradient(90deg, #3b82f6, #8b5cf6) !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03);
    border-radius: 12px;
    padding: 4px;
    border: 1px solid rgba(99,179,237,0.15);
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px;
    color: #94a3b8 !important;
    font-weight: 500;
    padding: 8px 20px;
    transition: all 0.3s ease;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
    color: white !important;
}

[data-testid="stForm"] {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(99,179,237,0.1);
    border-radius: 20px;
    padding: 24px;
}

div[data-testid="stDataFrame"] {
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 12px;
    overflow: hidden;
}

h1, h2, h3, h4 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #e2e8f0 !important;
}

p, li, div { color: #e2e8f0; }

.stSuccess > div {
    background: rgba(16,185,129,0.1) !important;
    border: 1px solid rgba(16,185,129,0.3) !important;
    border-radius: 12px !important;
    color: #6ee7b7 !important;
}
.stError > div {
    background: rgba(239,68,68,0.1) !important;
    border: 1px solid rgba(239,68,68,0.3) !important;
    border-radius: 12px !important;
}
.stWarning > div {
    background: rgba(245,158,11,0.1) !important;
    border: 1px solid rgba(245,158,11,0.3) !important;
    border-radius: 12px !important;
}
.stInfo > div {
    background: rgba(59,130,246,0.1) !important;
    border: 1px solid rgba(59,130,246,0.3) !important;
    border-radius: 12px !important;
}

@keyframes blink {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.3; }
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# CONSTANTS
# ============================================================
API_URL   = 'http://127.0.0.1:8000'
DATA_PATH = 'data/processed/cleaned_loans.parquet'

TIER_COLORS = {
    'ACCEPT' : '#10b981',
    'REVIEW' : '#f59e0b',
    'CAUTION': '#f97316',
    'DECLINE': '#ef4444',
}
TIER_BG = {
    'ACCEPT' : 'rgba(16,185,129,0.12)',
    'REVIEW' : 'rgba(245,158,11,0.12)',
    'CAUTION': 'rgba(249,115,22,0.12)',
    'DECLINE': 'rgba(239,68,68,0.12)',
}


# ============================================================
# PLOTLY THEME
# ============================================================
PLOTLY_LAYOUT = dict(
    paper_bgcolor = 'rgba(0,0,0,0)',
    plot_bgcolor  = 'rgba(0,0,0,0)',
    font          = dict(family='Inter', color='#94a3b8', size=12),
    margin        = dict(t=50, b=40, l=40, r=20),
    height        = 340,
    xaxis         = dict(
        gridcolor     = 'rgba(99,179,237,0.08)',
        zerolinecolor = 'rgba(99,179,237,0.1)',
    ),
    yaxis         = dict(
        gridcolor     = 'rgba(99,179,237,0.08)',
        zerolinecolor = 'rgba(99,179,237,0.1)',
    ),
    showlegend    = False,
)


# ============================================================
# API HELPERS
# ============================================================
@st.cache_data(ttl=30)
def check_api_health():
    try:
        r = requests.get(f'{API_URL}/health', timeout=5)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def call_api(endpoint, payload=None, method='GET'):
    try:
        url = f'{API_URL}{endpoint}'
        r   = (requests.post(url, json=payload, timeout=15)
               if method == 'POST'
               else requests.get(url, timeout=10))
        if r.status_code == 200:
            return {'ok': True, 'data': r.json()}
        return {'ok': False, 'error': f'HTTP {r.status_code}: {r.text[:200]}'}
    except requests.exceptions.ConnectionError:
        return {'ok': False,
                'error': 'API offline.\nRun: uvicorn api.app:app --port 8000'}
    except Exception as e:
        return {'ok': False, 'error': str(e)}


# ============================================================
# PLOTLY GAUGE
# ============================================================
def plotly_gauge(probability: float, tier: str) -> go.Figure:
    """Render a Plotly gauge chart for default probability."""
    color = TIER_COLORS.get(tier, '#94a3b8')
    pct   = probability * 100

    fig = go.Figure(go.Indicator(
        mode  = 'gauge+number+delta',
        value = pct,
        delta = {
            'reference'  : 20,
            'valueformat': '.1f',
            'suffix'     : '%',
        },
        number= {
            'suffix'     : '%',
            'font'       : {
                'size'  : 36,
                'color' : color,
                'family': 'Space Grotesk',
            },
        },
        title = {
            'text': 'Default Probability',
            'font': {'size': 13, 'color': '#94a3b8', 'family': 'Inter'},
        },
        gauge = {
            'axis': {
                'range'    : [0, 100],
                'tickwidth': 1,
                'tickcolor': '#334155',
                'tickfont' : {'color': '#64748b', 'size': 10},
            },
            'bar'      : {'color': color, 'thickness': 0.25},
            'bgcolor'  : 'rgba(0,0,0,0)',
            'borderwidth': 0,
            'steps'    : [
                {'range': [0,  12],  'color': 'rgba(16,185,129,0.12)'},
                {'range': [12, 25],  'color': 'rgba(245,158,11,0.12)'},
                {'range': [25, 40],  'color': 'rgba(249,115,22,0.12)'},
                {'range': [40, 100], 'color': 'rgba(239,68,68,0.12)'},
            ],
            'threshold': {
                'line' : {'color': color, 'width': 3},
                'thickness': 0.8,
                'value': pct,
            },
        },
    ))

    fig.update_layout(
        paper_bgcolor = 'rgba(0,0,0,0)',
        plot_bgcolor  = 'rgba(0,0,0,0)',
        font          = dict(family='Inter', color='#94a3b8'),
        height        = 240,
        margin        = dict(t=30, b=10, l=30, r=30),
    )
    return fig


# ============================================================
# REUSABLE COMPONENTS
# ============================================================
def section_header(title: str, subtitle: str = '', icon: str = ''):
    """Section header — no leading newline before <div> to avoid code-block bug."""
    icon_html = f'<span style="font-size:22px;margin-right:10px">{esc(icon)}</span>' if icon else ''
    sub_html  = (
        f'<p style="color:#64748b;font-size:13px;margin:4px 0 0 0;'
        f'font-family:Inter,sans-serif">{esc(subtitle)}</p>'
        if subtitle else ''
    )
    # NOTE: string starts with <div immediately — no leading \n
    st.markdown(
        f'<div style="margin:28px 0 18px 0;padding-bottom:14px;border-bottom:1px solid rgba(99,179,237,0.12);">'
        f'<div style="display:flex;align-items:center;">'
        f'{icon_html}'
        f'<div>'
        f'<h2 style="font-family:\'Space Grotesk\',sans-serif;font-size:20px;font-weight:700;color:#e2e8f0;margin:0;">{esc(title)}</h2>'
        f'{sub_html}'
        f'</div></div></div>',
        unsafe_allow_html=True,
    )


def hero_banner():
    """Top banner — string starts with <div immediately (no leading newline)."""
    st.markdown(
        '<div style="background:linear-gradient(135deg,rgba(59,130,246,0.08),rgba(139,92,246,0.08),rgba(6,182,212,0.08));'
        'border:1px solid rgba(99,179,237,0.18);border-radius:20px;padding:44px 36px;text-align:center;'
        'margin-bottom:28px;position:relative;overflow:hidden;">'
        '<div style="display:inline-block;background:rgba(59,130,246,0.12);border:1px solid rgba(59,130,246,0.25);'
        'border-radius:50px;padding:6px 18px;margin-bottom:18px;font-family:Inter,sans-serif;font-size:12px;'
        'color:#94a3b8;letter-spacing:1px;text-transform:uppercase;">AI-Powered Risk Platform</div>'
        '<h1 style="font-family:\'Space Grotesk\',sans-serif;font-size:48px;font-weight:800;color:#e2e8f0;'
        'margin:0 0 8px 0;line-height:1.2;">Credit<span style="color:#3b82f6;">IQ</span></h1>'
        '<p style="font-family:\'Space Grotesk\',sans-serif;font-size:17px;color:#64748b;margin:0 0 28px 0;">'
        'Intelligent Credit Risk Intelligence Platform</p>'
        '</div>',
        unsafe_allow_html=True,
    )


def tier_banner(tier, label, action, expected_loss):
    """Decision result banner — all dynamic values escaped."""
    color = TIER_COLORS.get(tier, '#94a3b8')
    bg    = TIER_BG.get(tier, 'rgba(148,163,184,0.1)')
    st.markdown(
        f'<div style="background:{bg};border:2px solid {color};border-radius:16px;'
        f'padding:24px;text-align:center;margin:12px 0;">'
        f'<div style="display:inline-block;background:{color};color:white;'
        f'font-family:\'Space Grotesk\',sans-serif;font-weight:700;font-size:20px;'
        f'padding:8px 32px;border-radius:50px;letter-spacing:3px;margin-bottom:12px;">{esc(tier)}</div>'
        f'<p style="color:#e2e8f0;font-family:Inter,sans-serif;font-size:15px;font-weight:500;margin:0 0 12px 0;">{esc(label)}</p>'
        f'<div style="background:rgba(0,0,0,0.2);border-radius:10px;padding:10px 14px;text-align:left;margin-bottom:8px;">'
        f'<div style="color:#94a3b8;font-size:10px;font-family:Inter,sans-serif;text-transform:uppercase;'
        f'letter-spacing:1px;margin-bottom:3px;">RECOMMENDED ACTION</div>'
        f'<div style="color:#e2e8f0;font-family:Inter,sans-serif;font-size:13px;">{esc(action)}</div>'
        f'</div>'
        f'<div style="color:#64748b;font-size:12px;font-family:Inter,sans-serif;">Expected Loss: {esc(expected_loss)}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def risk_factor_row(feature, value, risk_level):
    """Single risk factor row — all dynamic values escaped."""
    colors = {
        'HIGH'  : ('#ef4444', 85),
        'MEDIUM': ('#f59e0b', 52),
        'LOW'   : ('#10b981', 22),
    }
    color, fill_pct = colors.get(risk_level, ('#94a3b8', 40))
    st.markdown(
        f'<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(99,179,237,0.08);'
        f'border-radius:10px;padding:12px 14px;margin:5px 0;display:flex;align-items:center;gap:14px;">'
        f'<div style="min-width:160px;">'
        f'<div style="font-family:Inter,sans-serif;font-size:13px;font-weight:500;color:#e2e8f0;">{esc(feature)}</div>'
        f'<div style="font-family:Inter,sans-serif;font-size:12px;color:#64748b;margin-top:1px;">{esc(value)}</div>'
        f'</div>'
        f'<div style="flex:1;height:6px;background:rgba(255,255,255,0.06);border-radius:3px;overflow:hidden;">'
        f'<div style="width:{fill_pct}%;height:100%;background:{color};border-radius:3px;"></div>'
        f'</div>'
        f'<div style="min-width:58px;text-align:right;font-family:Inter,sans-serif;font-size:10px;font-weight:600;'
        f'color:{color};background:{color}18;border:1px solid {color}35;border-radius:5px;padding:2px 7px;">{esc(risk_level)}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def tier_info_card(tier, rng, color, bg, desc, action, loss, volume):
    """Risk tier info card — all dynamic values escaped."""
    st.markdown(
        f'<div style="background:{bg};border:1px solid {color}35;border-left:4px solid {color};'
        f'border-radius:14px;padding:20px;margin-bottom:12px;">'
        f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;flex-wrap:wrap;gap:8px;">'
        f'<div style="display:flex;align-items:center;gap:14px;">'
        f'<span style="font-family:\'Space Grotesk\',sans-serif;font-size:18px;font-weight:800;color:{color};letter-spacing:2px;">{esc(tier)}</span>'
        f'<span style="font-family:\'Space Grotesk\',sans-serif;font-size:20px;font-weight:700;color:#e2e8f0;">{esc(rng)}</span>'
        f'</div>'
        f'<span style="background:{color}18;border:1px solid {color}35;border-radius:50px;padding:3px 12px;'
        f'font-family:Inter,sans-serif;font-size:11px;color:{color};font-weight:600;">{esc(volume)}</span>'
        f'</div>'
        f'<p style="font-family:Inter,sans-serif;font-size:13px;color:#94a3b8;margin-bottom:12px;line-height:1.5;">{esc(desc)}</p>'
        f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">'
        f'<div style="background:rgba(0,0,0,0.18);border-radius:8px;padding:10px;">'
        f'<div style="font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:1px;font-family:Inter,sans-serif;margin-bottom:3px;">Action</div>'
        f'<div style="font-family:Inter,sans-serif;font-size:12px;color:#e2e8f0;">{esc(action)}</div>'
        f'</div>'
        f'<div style="background:rgba(0,0,0,0.18);border-radius:8px;padding:10px;">'
        f'<div style="font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:1px;font-family:Inter,sans-serif;margin-bottom:3px;">Expected Loss</div>'
        f'<div style="font-family:Inter,sans-serif;font-size:12px;color:{color};font-weight:600;">{esc(loss)}</div>'
        f'</div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(
        '<div style="padding:20px 0 10px 0;text-align:center;">'
        '<div style="font-family:\'Space Grotesk\',sans-serif;font-size:26px;font-weight:800;color:#3b82f6;margin-bottom:3px;">CreditIQ</div>'
        '<div style="font-family:Inter,sans-serif;font-size:10px;color:#475569;text-transform:uppercase;letter-spacing:2px;">Risk Intelligence</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    health = check_api_health()
    if health:
        st.markdown(
            f'<div style="background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.3);'
            f'border-radius:10px;padding:10px 12px;margin-bottom:14px;display:flex;align-items:center;gap:10px;">'
            f'<div style="width:8px;height:8px;background:#10b981;border-radius:50%;flex-shrink:0;box-shadow:0 0 6px #10b981;"></div>'
            f'<div>'
            f'<div style="font-family:Inter,sans-serif;font-size:11px;font-weight:600;color:#10b981;">API ONLINE</div>'
            f'<div style="font-family:Inter,sans-serif;font-size:10px;color:#64748b;">AUC: {health.get("model_auc",0):.4f}</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.error('⚠ API Offline\n\n`uvicorn api.app:app --port 8000`')

    st.markdown(
        '<div style="color:#64748b;font-size:11px;font-family:Inter,sans-serif;'
        'text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">Navigation</div>',
        unsafe_allow_html=True,
    )

    page = st.radio(
        '',
        ['🏠  Dashboard',
         '🎯  Loan Scorer',
         '📊  Portfolio Analytics',
         '🤖  Model Performance',
         '⚙️  Risk Tiers'],
        label_visibility='collapsed',
    )

    st.divider()

    st.markdown(
        '<div style="padding:12px;background:rgba(59,130,246,0.07);border-radius:10px;border:1px solid rgba(59,130,246,0.15);">'
        '<div style="font-family:Inter,sans-serif;font-size:11px;color:#3b82f6;font-weight:600;margin-bottom:6px;">Model Info</div>'
        '<div style="font-family:Inter,sans-serif;font-size:11px;color:#64748b;line-height:1.7;">'
        'Dataset: Lending Club<br>Period: 2007-2018<br>Loans: 1.3M<br>Features: 35'
        '</div></div>',
        unsafe_allow_html=True,
    )


# ============================================================
# PAGE 1 — DASHBOARD
# ============================================================
if '🏠' in page:

    hero_banner()

    section_header('Platform Overview', 'Real-time portfolio metrics', '📈')
    c1, c2, c3, c4 = st.columns(4)

    model_auc = health['model_auc'] if health else 0.0
    c1.metric('📋  Total Loans',   '1.3M+',             '2007–2018 dataset')
    c2.metric('⚠️  Default Rate',  '20.1%',             'Portfolio average')
    c3.metric('🎯  Model AUC',     f'{model_auc:.4f}',  'Test set')
    c4.metric('🔬  Features',      '35',                'Engineered predictors')

    st.divider()

    section_header('Portfolio Analytics', 'Live data from your dataset', '📊')

    if os.path.exists(DATA_PATH):
        df = pd.read_parquet(DATA_PATH)

        col_l, col_r = st.columns(2)

        with col_l:
            if 'grade' in df.columns:
                gdf = (
                    df.groupby('grade', observed=True)['target']
                    .agg(['mean','count'])
                    .reset_index()
                )
                gdf.columns = ['grade','dr','count']
                gdf['grade']  = gdf['grade'].astype(str)
                gdf['dr_pct'] = gdf['dr'] * 100

                n      = len(gdf)
                clrs   = px.colors.sample_colorscale(
                    'RdYlGn_r', [i/(n-1) for i in range(n)]
                )

                fig = go.Figure(go.Bar(
                    x=gdf['grade'], y=gdf['dr_pct'],
                    marker_color=clrs,
                    text=[f'{v:.1f}%' for v in gdf['dr_pct']],
                    textposition='outside',
                    textfont=dict(color='#94a3b8', size=11),
                ))
                layout = dict(PLOTLY_LAYOUT)
                layout['title'] = dict(
                    text='Default Rate by Grade',
                    font=dict(family='Space Grotesk', size=15, color='#e2e8f0')
                )
                layout['yaxis'] = dict(
                    gridcolor='rgba(99,179,237,0.08)',
                    title='Default Rate (%)',
                )
                fig.update_layout(**layout)
                st.plotly_chart(fig, use_container_width=True)

        with col_r:
            if 'dti' in df.columns:
                dti_data = df[df['dti'].between(0, 50)].copy()
                dti_data['dti_bin'] = pd.cut(dti_data['dti'], bins=10)
                dti_dr = (
                    dti_data.groupby('dti_bin', observed=True)['target']
                    .mean()
                    .reset_index()
                )
                dti_dr['dr'] = dti_dr['target'] * 100
                bar_c = [
                    '#ef4444' if v > 25 else
                    '#f59e0b' if v > 15 else '#10b981'
                    for v in dti_dr['dr']
                ]
                fig2 = go.Figure(go.Bar(
                    x=list(range(len(dti_dr))),
                    y=dti_dr['dr'],
                    marker_color=bar_c,
                    text=[f'{v:.1f}%' for v in dti_dr['dr']],
                    textposition='outside',
                    textfont=dict(color='#94a3b8', size=9),
                ))
                fig2.add_hline(
                    y=df['target'].mean()*100,
                    line_dash='dash', line_color='#3b82f6',
                    annotation_text='Portfolio avg',
                    annotation_font_color='#3b82f6',
                    annotation_font_size=10,
                )
                layout2 = dict(PLOTLY_LAYOUT)
                layout2['title'] = dict(
                    text='Default Rate by DTI Decile',
                    font=dict(family='Space Grotesk', size=15, color='#e2e8f0')
                )
                layout2['yaxis'] = dict(
                    gridcolor='rgba(99,179,237,0.08)',
                    title='Default Rate (%)',
                )
                fig2.update_layout(**layout2)
                st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info('Run `python src/preprocessing.py` to load portfolio data.')

    st.divider()
    section_header('Risk Tier Framework', 'How decisions are made', '🎯')

    t1, t2, t3, t4 = st.columns(4)
    tier_data = [
        (t1, 'ACCEPT',  '0-12%',   '#10b981', 'Auto-approve'),
        (t2, 'REVIEW',  '12-25%',  '#f59e0b', 'Manual review'),
        (t3, 'CAUTION', '25-40%',  '#f97316', 'Conditional offer'),
        (t4, 'DECLINE', '40%+',    '#ef4444', 'Adverse action'),
    ]
    for col, tier, rng, color, action in tier_data:
        with col:
            st.markdown(
                f'<div style="background:{TIER_BG.get(tier,"rgba(148,163,184,0.1)")};'
                f'border:1px solid {color}35;border-radius:14px;padding:18px;text-align:center;">'
                f'<div style="color:{color};font-family:\'Space Grotesk\',sans-serif;font-size:14px;'
                f'font-weight:700;letter-spacing:2px;margin-bottom:6px;">{esc(tier)}</div>'
                f'<div style="color:#e2e8f0;font-family:\'Space Grotesk\',sans-serif;font-size:20px;'
                f'font-weight:700;margin-bottom:6px;">{esc(rng)}</div>'
                f'<div style="color:#64748b;font-family:Inter,sans-serif;font-size:12px;">{esc(action)}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ============================================================
# PAGE 2 — LOAN SCORER
# ============================================================
elif '🎯' in page:
    section_header('Loan Application Scorer',
                   'Real-time AI-powered credit risk assessment', '🎯')

    if not health:
        st.error('API is offline.\n\nStart with:\n```\nuvicorn api.app:app --reload --port 8000\n```')
        st.stop()

    with st.form('loan_form', clear_on_submit=False):

        st.markdown(
            '<div style="font-family:Space Grotesk,sans-serif;font-size:13px;font-weight:600;'
            'color:#3b82f6;text-transform:uppercase;letter-spacing:2px;margin-bottom:14px;">'
            '01 — Loan Details</div>',
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            loan_amnt = st.number_input('Loan Amount ($)', 500, 40000, 15000, 500)
        with c2:
            term      = st.selectbox('Loan Term', ['36 months','60 months'])
        with c3:
            int_rate  = st.slider('Interest Rate (%)', 5.0, 30.0, 13.5, 0.1)
        with c4:
            grade     = st.selectbox('Grade', ['A','B','C','D','E','F','G'], index=2)

        st.divider()

        st.markdown(
            '<div style="font-family:Space Grotesk,sans-serif;font-size:13px;font-weight:600;'
            'color:#8b5cf6;text-transform:uppercase;letter-spacing:2px;margin-bottom:14px;">'
            '02 — Borrower Information</div>',
            unsafe_allow_html=True,
        )

        c5, c6, c7, c8 = st.columns(4)
        with c5:
            annual_inc = st.number_input('Annual Income ($)', 10000, 500000, 72000, 1000)
        with c6:
            dti        = st.slider('DTI Ratio (%)', 0.0, 50.0, 18.4, 0.1)
        with c7:
            home_ownership = st.selectbox(
                'Home Ownership', ['RENT','MORTGAGE','OWN','OTHER']
            )
        with c8:
            emp_length = st.selectbox(
                'Employment Length',
                ['< 1 year','1 year','2 years','3 years','4 years',
                 '5 years','6 years','7 years','8 years','9 years',
                 '10+ years','Unknown'], index=5
            )

        st.divider()

        st.markdown(
            '<div style="font-family:Space Grotesk,sans-serif;font-size:13px;font-weight:600;'
            'color:#06b6d4;text-transform:uppercase;letter-spacing:2px;margin-bottom:14px;">'
            '03 — Credit Profile</div>',
            unsafe_allow_html=True,
        )

        c9, c10, c11, c12 = st.columns(4)
        with c9:
            revol_util  = st.slider('Revolving Utilization (%)', 0.0, 100.0, 42.7)
        with c10:
            revol_bal   = st.number_input('Revolving Balance ($)', 0, 200000, 12500, 500)
        with c11:
            delinq_2yrs = st.number_input('Delinquencies (2yr)', 0, 20, 0)
        with c12:
            open_acc    = st.number_input('Open Accounts', 0, 50, 9)

        c13, c14, c15, c16 = st.columns(4)
        with c13:
            total_acc   = st.number_input('Total Accounts', 0, 100, 22)
        with c14:
            pub_rec     = st.number_input('Public Records', 0, 10, 0)
        with c15:
            inq_6mths   = st.number_input('Inquiries (6mo)', 0, 20, 1)
        with c16:
            purpose     = st.selectbox(
                'Loan Purpose',
                ['debt_consolidation','credit_card','home_improvement',
                 'small_business','major_purchase','medical','other',
                 'car','vacation','moving','wedding']
            )

        c17, c18 = st.columns(2)
        with c17:
            verification = st.selectbox(
                'Verification Status',
                ['Not Verified','Source Verified','Verified']
            )
        with c18:
            sub_grade = st.selectbox(
                'Sub-Grade',
                [f'{g}{n}' for g in 'ABCDEFG' for n in range(1,6)],
                index=12
            )

        submitted = st.form_submit_button(
            '🚀  Analyze Credit Risk',
            type='primary',
            use_container_width=True,
        )

    # ── Result ────────────────────────────────────────────────
    if submitted:
        emp_map = {
            '< 1 year':0,'1 year':1,'2 years':2,'3 years':3,
            '4 years':4,'5 years':5,'6 years':6,'7 years':7,
            '8 years':8,'9 years':9,'10+ years':10,'Unknown':5,
        }
        emp_years  = emp_map.get(emp_length, 5)
        term_n     = int(term.split()[0])
        ir_m       = int_rate / 100 / 12
        installment = round(
            loan_amnt * ir_m / (1 - (1 + ir_m) ** (-term_n)), 2
        )

        payload = {
            'loan_amnt'           : loan_amnt,
            'term'                : term,
            'int_rate'            : f'{int_rate}%',
            'installment'         : installment,
            'grade'               : grade,
            'sub_grade'           : sub_grade,
            'emp_length'          : emp_length,
            'emp_length_years'    : emp_years,
            'home_ownership'      : home_ownership,
            'annual_inc'          : annual_inc,
            'verification_status' : verification,
            'purpose'             : purpose,
            'dti'                 : dti,
            'revol_util'          : revol_util,
            'revol_bal'           : revol_bal,
            'delinq_2yrs'         : delinq_2yrs,
            'inq_last_6mths'      : inq_6mths,
            'open_acc'            : open_acc,
            'pub_rec'             : pub_rec,
            'total_acc'           : total_acc,
            'mort_acc'            : 0,
            'pub_rec_bankruptcies': 0,
            'issue_d'             : 'Jan-2024',
            'earliest_cr_line'    : 'Jan-2010',
        }

        with st.spinner('Running AI risk assessment...'):
            time.sleep(0.4)
            result = call_api('/score', payload, method='POST')

        if result['ok']:
            data = result['data']
            tier = data['tier']

            st.divider()
            section_header('Assessment Result', '', '📋')

            gauge_col, detail_col = st.columns([1, 2])

            with gauge_col:
                fig_gauge = plotly_gauge(
                    data['default_probability'], tier
                )
                st.plotly_chart(
                    fig_gauge, use_container_width=True,
                    config={'displayModeBar': False}
                )
                tier_banner(
                    tier,
                    data['label'],
                    data['action'],
                    data['expected_loss'],
                )

            with detail_col:
                section_header('Key Risk Factors', '', '📊')

                monthly_inc   = annual_inc / 12
                pmt_to_income = installment / (monthly_inc + 1)
                loan_to_inc   = loan_amnt / annual_inc

                factors = [
                    ('Interest Rate',
                     f'{int_rate:.1f}%',
                     'HIGH' if int_rate > 18 else 'MEDIUM' if int_rate > 12 else 'LOW'),
                    ('Debt-to-Income',
                     f'{dti:.1f}%',
                     'HIGH' if dti > 30 else 'MEDIUM' if dti > 20 else 'LOW'),
                    ('Revolving Utilization',
                     f'{revol_util:.1f}%',
                     'HIGH' if revol_util > 70 else 'MEDIUM' if revol_util > 40 else 'LOW'),
                    ('Loan-to-Income',
                     f'{loan_to_inc:.1%}',
                     'HIGH' if loan_to_inc > 0.4 else 'MEDIUM' if loan_to_inc > 0.2 else 'LOW'),
                    ('Payment-to-Income',
                     f'{pmt_to_income:.1%}',
                     'HIGH' if pmt_to_income > 0.25 else 'MEDIUM' if pmt_to_income > 0.15 else 'LOW'),
                    ('Employment Length',
                     emp_length,
                     'LOW' if emp_years >= 5 else 'MEDIUM' if emp_years >= 2 else 'HIGH'),
                    ('Delinquencies',
                     str(delinq_2yrs),
                     'HIGH' if delinq_2yrs > 1 else 'MEDIUM' if delinq_2yrs > 0 else 'LOW'),
                ]
                for feat, val, lvl in factors:
                    risk_factor_row(feat, val, lvl)

                st.divider()
                s1, s2, s3 = st.columns(3)
                s1.metric('Monthly Payment', f'${installment:,.2f}')
                s2.metric('Total Repayment', f'${installment*term_n:,.0f}')
                s3.metric('Total Interest',  f'${installment*term_n - loan_amnt:,.0f}')
        else:
            st.error(f'Scoring error: {result["error"]}')


# ============================================================
# PAGE 3 — PORTFOLIO ANALYTICS
# ============================================================
elif '📊' in page:
    section_header('Portfolio Analytics',
                   'Deep-dive into your loan portfolio', '📊')

    if not os.path.exists(DATA_PATH):
        st.warning('Run `python src/preprocessing.py` first.')
        st.stop()

    df = pd.read_parquet(DATA_PATH)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric('Total Loans',  f"{len(df):,}")
    m2.metric('Default Rate', f"{df['target'].mean():.1%}")
    m3.metric('Good Loans',   f"{(df['target']==0).sum():,}")
    m4.metric('Defaults',     f"{df['target'].sum():,}")
    m5.metric('Avg Loan',
        f"${df['loan_amnt'].mean():,.0f}" if 'loan_amnt' in df.columns else 'N/A'
    )

    st.divider()

    tabs = st.tabs(['📈 Grade', '🎯 DTI', '💰 Income', '📋 Data'])

    with tabs[0]:
        if 'grade' in df.columns:
            gdf = (
                df.groupby('grade', observed=True)
                .agg(dr=('target','mean'), count=('target','count'),
                     avg_rate=('int_rate','mean'))
                .reset_index()
            )
            gdf['grade']  = gdf['grade'].astype(str)
            gdf['dr_pct'] = gdf['dr'] * 100
            n    = len(gdf)
            clrs = px.colors.sample_colorscale(
                'RdYlGn_r', [i/max(n-1,1) for i in range(n)]
            )

            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=['Default Rate by Grade',
                                'Interest Rate vs Default Rate'],
                horizontal_spacing=0.1,
            )
            fig.add_trace(
                go.Bar(x=gdf['grade'], y=gdf['dr_pct'],
                       marker_color=clrs,
                       text=[f'{v:.1f}%' for v in gdf['dr_pct']],
                       textposition='outside',
                       textfont=dict(color='#94a3b8', size=10),
                       showlegend=False),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=gdf['avg_rate'], y=gdf['dr_pct'],
                    mode='markers+text',
                    text=gdf['grade'],
                    textposition='middle right',
                    marker=dict(size=18, color=clrs,
                                line=dict(color='white', width=2)),
                    textfont=dict(color='#e2e8f0', size=12,
                                  family='Space Grotesk'),
                    showlegend=False,
                ),
                row=1, col=2
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter', color='#94a3b8'),
                height=380,
                margin=dict(t=50, b=40, l=40, r=20),
                xaxis=dict(gridcolor='rgba(99,179,237,0.08)'),
                yaxis=dict(gridcolor='rgba(99,179,237,0.08)',
                           title='Default Rate (%)'),
                xaxis2=dict(gridcolor='rgba(99,179,237,0.08)',
                            title='Avg Interest Rate (%)'),
                yaxis2=dict(gridcolor='rgba(99,179,237,0.08)',
                            title='Default Rate (%)'),
            )
            st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        dti_d = df[df['dti'].between(0, 50)].copy()
        dti_d['dti_bin'] = pd.cut(dti_d['dti'], bins=10)
        dti_dr = (
            dti_d.groupby('dti_bin', observed=True)['target']
            .mean().reset_index()
        )
        dti_dr['dr'] = dti_dr['target'] * 100

        bar_c = [
            '#ef4444' if v > 25 else '#f59e0b' if v > 15 else '#10b981'
            for v in dti_dr['dr']
        ]
        fig3 = go.Figure(go.Bar(
            x=list(range(len(dti_dr))),
            y=dti_dr['dr'],
            marker_color=bar_c,
            text=[f'{v:.1f}%' for v in dti_dr['dr']],
            textposition='outside',
            textfont=dict(color='#94a3b8', size=10),
        ))
        fig3.add_hline(
            y=df['target'].mean()*100,
            line_dash='dash', line_color='#3b82f6',
            annotation_text='Portfolio avg',
            annotation_font_color='#3b82f6',
            annotation_font_size=10,
        )
        fig3.update_layout(
            title=dict(text='Default Rate by DTI Decile — Cliff at 30%',
                       font=dict(family='Space Grotesk', size=14, color='#e2e8f0')),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter', color='#94a3b8'),
            height=380,
            margin=dict(t=50, b=40, l=40, r=20),
            xaxis=dict(gridcolor='rgba(99,179,237,0.08)', title='DTI Decile'),
            yaxis=dict(gridcolor='rgba(99,179,237,0.08)', title='Default Rate (%)'),
            showlegend=False,
        )
        st.plotly_chart(fig3, use_container_width=True)

    with tabs[2]:
        if 'annual_inc' in df.columns:
            inc = df['annual_inc'].clip(0, 250000).dropna()
            fig4 = px.histogram(
                inc, nbins=60,
                color_discrete_sequence=['#3b82f6'],
                opacity=0.8,
                labels={'value':'Annual Income ($)', 'count':'Count'},
            )
            fig4.update_layout(
                title=dict(text='Annual Income Distribution (clipped $250k)',
                           font=dict(family='Space Grotesk', size=14, color='#e2e8f0')),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter', color='#94a3b8'),
                height=380,
                margin=dict(t=50, b=40, l=40, r=20),
                xaxis=dict(gridcolor='rgba(99,179,237,0.08)'),
                yaxis=dict(gridcolor='rgba(99,179,237,0.08)'),
                showlegend=False,
            )
            st.plotly_chart(fig4, use_container_width=True)

    with tabs[3]:
        st.dataframe(df.head(1000), use_container_width=True, hide_index=True)


# ============================================================
# PAGE 4 — MODEL PERFORMANCE
# ============================================================
elif '🤖' in page:
    section_header('Model Performance',
                   'Evaluation metrics and explanations', '🤖')

    csv_path = 'outputs/model_comparison.csv'
    if os.path.exists(csv_path):
        cdf = pd.read_csv(csv_path)
        for _, row in cdf.iterrows():
            auc   = row.get('test_roc_auc', 0)
            prauc = row.get('test_pr_auc', 0)
            mtype = row.get('model_type', '')
            color = '#10b981' if mtype == 'primary' else '#3b82f6'
            label = 'PRIMARY' if mtype == 'primary' else 'BASELINE'

            c_l, c_r = st.columns([2, 3])
            with c_l:
                st.markdown(
                    f'<div style="background:{color}0d;border:1px solid {color}30;'
                    f'border-radius:14px;padding:18px 20px;margin-bottom:12px;">'
                    f'<div style="font-family:\'Space Grotesk\',sans-serif;font-size:16px;'
                    f'font-weight:700;color:#e2e8f0;margin-bottom:6px;">{esc(row["model"])}</div>'
                    f'<div style="display:inline-block;background:{color}20;border:1px solid {color}40;'
                    f'border-radius:50px;padding:2px 12px;font-size:10px;font-weight:600;color:{color};'
                    f'font-family:Inter,sans-serif;letter-spacing:1px;">{esc(label)}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with c_r:
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric('Test ROC-AUC', f'{auc:.4f}')
                mc2.metric('Test PR-AUC',  f'{prauc:.4f}')
                mc3.metric('Train Time',   f"{row.get('train_time_s',0):.1f}s")

    st.divider()

    plots = {
        'Model Evaluation'   : 'outputs/model_evaluation.png',
        'SHAP Explanations'  : 'outputs/shap_explanations.png',
        'Feature Importance' : 'outputs/feature_importance_report.png',
        'Confusion Matrix'   : 'outputs/confusion_matrix.png',
        'Drift Dashboard'    : 'outputs/drift_dashboard.png',
    }
    for plot_name, plot_path in plots.items():
        if os.path.exists(plot_path):
            section_header(plot_name, '', '')
            st.image(plot_path, use_container_width=True)
            st.divider()


# ============================================================
# PAGE 5 — RISK TIERS
# ============================================================
elif '⚙️' in page:
    section_header('Risk Tier Configuration',
                   'Threshold definitions and business rules', '⚙️')

    tiers_def = [
        dict(tier='ACCEPT',  rng='0% – 12%',  color='#10b981',
             bg='rgba(16,185,129,0.1)',
             desc='Prime borrowers with strong credit profiles. Low DTI, long credit history, clean record.',
             action='Auto-approve at standard terms. No manual review required.',
             loss='Less than 3% portfolio loss expected.',
             volume='~35% of applications'),
        dict(tier='REVIEW',  rng='12% – 25%', color='#f59e0b',
             bg='rgba(245,158,11,0.1)',
             desc='Near-prime borrowers. Moderate risk factors. Human judgment adds value.',
             action='Route to underwriter. Request income documentation.',
             loss='3% to 8% portfolio loss expected.',
             volume='~30% of applications'),
        dict(tier='CAUTION', rng='25% – 40%', color='#f97316',
             bg='rgba(249,115,22,0.1)',
             desc='Subprime borrowers with elevated risk. Conditional approval with risk mitigation.',
             action='Offer reduced amount or require co-signer.',
             loss='8% to 15% portfolio loss expected.',
             volume='~20% of applications'),
        dict(tier='DECLINE', rng='40% +',     color='#ef4444',
             bg='rgba(239,68,68,0.1)',
             desc='Deep subprime. Risk exceeds acceptable portfolio limits. ECOA-compliant decline.',
             action='Issue adverse action notice. Provide top 3 decline reasons.',
             loss='Greater than 15% portfolio loss expected.',
             volume='~15% of applications'),
    ]

    for t in tiers_def:
        tier_info_card(
            t['tier'], t['rng'], t['color'], t['bg'],
            t['desc'], t['action'], t['loss'], t['volume']
        )

    st.divider()
    st.info(
        '⚖️ **Regulatory Compliance — ECOA**\n\n'
        'All DECLINE decisions must include an adverse action notice '
        'with the top 3 specific reasons, as required by the Equal Credit '
        'Opportunity Act (ECOA) and Fair Credit Reporting Act (FCRA). '
        'SHAP explanations power this requirement automatically.'
    )