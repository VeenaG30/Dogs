import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title='Dog App Analytics',
    layout='wide',
    page_icon='',
    initial_sidebar_state='collapsed'
)

# -----------------------------------------------------------------------------
# THEME TOGGLE (Light / Dark)
# -----------------------------------------------------------------------------
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

def get_theme_colors(dark):
    if dark:
        return {
            'bg': '#0f1117',
            'surface': '#1a1d27',
            'surface2': '#21242f',
            'border': '#2e3148',
            'text': '#e8eaf0',
            'text_muted': '#8b90a8',
            'primary': '#4f98a3',
            'plotly_template': 'plotly_dark',
            'paper_bg': '#1a1d27',
            'plot_bg': '#1a1d27',
            'font_color': '#e8eaf0',
            'grid_color': '#2e3148',
        }
    else:
        return {
            'bg': '#f7f8fc',
            'surface': '#ffffff',
            'surface2': '#f0f2f8',
            'border': '#dde1ef',
            'text': '#1a1d2e',
            'text_muted': '#6b7280',
            'primary': '#01696f',
            'plotly_template': 'plotly_white',
            'paper_bg': '#ffffff',
            'plot_bg': '#f7f8fc',
            'font_color': '#1a1d2e',
            'grid_color': '#e5e7eb',
        }

dark = st.session_state.dark_mode
C = get_theme_colors(dark)

# -----------------------------------------------------------------------------
# GLOBAL CSS INJECTION
# -----------------------------------------------------------------------------
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}

    .stApp {{
        background-color: {C['bg']};
        color: {C['text']};
    }}

    /* Hide default streamlit header decoration */
    header[data-testid="stHeader"] {{
        background: {C['bg']};
        border-bottom: 1px solid {C['border']};
    }}

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {{
        background-color: {C['surface2']};
        border-radius: 8px;
        padding: 4px;
        gap: 2px;
        border: 1px solid {C['border']};
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 6px;
        padding: 8px 18px;
        font-size: 0.85rem;
        font-weight: 500;
        color: {C['text_muted']};
        background: transparent;
        border: none;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {C['surface']} !important;
        color: {C['primary']} !important;
        font-weight: 600;
        box-shadow: 0 1px 4px rgba(0,0,0,0.1);
    }}

    /* Metric cards */
    [data-testid="metric-container"] {{
        background-color: {C['surface']};
        border: 1px solid {C['border']};
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }}
    [data-testid="metric-container"] label {{
        color: {C['text_muted']} !important;
        font-size: 0.75rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    [data-testid="metric-container"] [data-testid="stMetricValue"] {{
        color: {C['text']} !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
    }}

    /* Section cards */
    .insight-box {{
        background-color: {C['surface']};
        border: 1px solid {C['border']};
        border-left: 3px solid {C['primary']};
        border-radius: 8px;
        padding: 14px 18px;
        margin-top: 12px;
        font-size: 0.85rem;
        color: {C['text_muted']};
        line-height: 1.6;
    }}
    .insight-box strong {{
        color: {C['text']};
    }}
    .insight-title {{
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: {C['primary']};
        margin-bottom: 6px;
    }}

    /* Dashboard header */
    .dash-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 0 16px 0;
        border-bottom: 1px solid {C['border']};
        margin-bottom: 24px;
    }}
    .dash-title {{
        font-size: 1.25rem;
        font-weight: 700;
        color: {C['text']};
        letter-spacing: -0.01em;
    }}
    .dash-subtitle {{
        font-size: 0.78rem;
        color: {C['text_muted']};
        font-weight: 400;
    }}

    /* Filter row */
    .filter-label {{
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: {C['text_muted']};
        margin-bottom: 4px;
    }}

    /* Divider */
    .section-divider {{
        border: none;
        border-top: 1px solid {C['border']};
        margin: 20px 0;
    }}

    /* Plotly chart container */
    .js-plotly-plot .plotly {{
        border-radius: 8px;
    }}

    /* Streamlit expander */
    .streamlit-expanderHeader {{
        background-color: {C['surface2']};
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 500;
    }}

    /* Toggle button styling */
    .theme-toggle-btn {{
        background: {C['surface2']};
        border: 1px solid {C['border']};
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.8rem;
        color: {C['text']};
        cursor: pointer;
    }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {C['surface']};
        border-right: 1px solid {C['border']};
    }}

    /* Dataframe */
    [data-testid="stDataFrame"] {{
        border: 1px solid {C['border']};
        border-radius: 8px;
        overflow: hidden;
    }}

    /* Section header */
    .section-header {{
        font-size: 0.95rem;
        font-weight: 600;
        color: {C['text']};
        margin: 20px 0 12px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid {C['border']};
    }}

    /* Small badge */
    .badge {{
        display: inline-block;
        background: {C['primary']}20;
        color: {C['primary']};
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.72rem;
        font-weight: 600;
        margin-left: 8px;
    }}

    /* Hide Streamlit branding */
    #MainMenu, footer {{visibility: hidden;}}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# DATA LOADING
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, 'dog_app_data.csv')
    df = pd.read_csv(csv_path)
    df['ownership_experience_encoded'] = df['ownership_years'].map(
        {'<1': 1, '1-3': 2, '4-7': 3, '8+': 4}
    )
    df['will_adopt_app'] = (df['app_use_likelihood'] == 'Yes').astype(int)
    return df

df = load_data()

# -----------------------------------------------------------------------------
# HEADER ROW
# -----------------------------------------------------------------------------
h_col1, h_col2 = st.columns([6, 1])
with h_col1:
    st.markdown(
        f'<div class="dash-title"> India Dog Care App <span style="font-weight:300">- Survey Analytics</span></div>'
        f'<div class="dash-subtitle">MBA - Data Analytics in Decision Making - Group Project</div>',
        unsafe_allow_html=True
    )
with h_col2:
    theme_label = "[Light] Light" if dark else "[Dark] Dark"
    if st.button(theme_label, key='theme_toggle', use_container_width=True):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TABS
# -----------------------------------------------------------------------------
tabs = st.tabs([
    'Overview',
    'Spending',
    'Challenges',
    'Features',
    'Classification',
    'Clustering',
    'Regression',
])

# -----------------------------------------------------------------------------
# HELPER: Plotly layout defaults
# -----------------------------------------------------------------------------
def apply_theme(fig, title=''):
    fig.update_layout(
        template=C['plotly_template'],
        paper_bgcolor=C['paper_bg'],
        plot_bgcolor=C['plot_bg'],
        font=dict(family='Inter', color=C['font_color'], size=12),
        title=dict(text=title, font=dict(size=13, weight=600), x=0, xanchor='left'),
        margin=dict(l=16, r=16, t=44, b=16),
        legend=dict(bgcolor='rgba(0,0,0,0)', borderwidth=0, font=dict(size=11)),
    )
    fig.update_xaxes(gridcolor=C['grid_color'], linecolor=C['border'], zerolinecolor=C['border'])
    fig.update_yaxes(gridcolor=C['grid_color'], linecolor=C['border'], zerolinecolor=C['border'])
    return fig

def insight(text):
    st.markdown(
        f'<div class="insight-box"><di
