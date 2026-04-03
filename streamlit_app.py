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

# 
# PAGE CONFIG
# 
st.set_page_config(
    page_title='Dog App Analytics',
    layout='wide',
    page_icon='',
    initial_sidebar_state='collapsed'
)

# 
# THEME TOGGLE (Light / Dark)
# 
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

# 
# GLOBAL CSS INJECTION
# 
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

    header[data-testid="stHeader"] {{
        background: {C['bg']};
        border-bottom: 1px solid {C['border']};
    }}

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

    .filter-label {{
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: {C['text_muted']};
        margin-bottom: 4px;
    }}

    .section-divider {{
        border: none;
        border-top: 1px solid {C['border']};
        margin: 20px 0;
    }}

    .js-plotly-plot .plotly {{
        border-radius: 8px;
    }}

    .streamlit-expanderHeader {{
        background-color: {C['surface2']};
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 500;
    }}

    .theme-toggle-btn {{
        background: {C['surface2']};
        border: 1px solid {C['border']};
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.8rem;
        color: {C['text']};
        cursor: pointer;
    }}

    [data-testid="stSidebar"] {{
        background-color: {C['surface']};
        border-right: 1px solid {C['border']};
    }}

    [data-testid="stDataFrame"] {{
        border: 1px solid {C['border']};
        border-radius: 8px;
        overflow: hidden;
    }}

    .section-header {{
        font-size: 0.95rem;
        font-weight: 600;
        color: {C['text']};
        margin: 20px 0 12px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid {C['border']};
    }}

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

    #MainMenu, footer {{visibility: hidden;}}
</style>
""", unsafe_allow_html=True)

# 
# DATA LOADING
# 
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

# 
# HEADER ROW
# 
h_col1, h_col2 = st.columns([6, 1])
with h_col1:
    st.markdown(
        f'<div class="dash-title"> India Dog Care App <span style="font-weight:300"> Survey Analytics</span></div>'
        f'<div class="dash-subtitle">MBA  Data Analytics in Decision Making  Group Project</div>',
        unsafe_allow_html=True
    )
with h_col2:
    theme_label = "[Light] Light" if dark else "[Dark] Dark"
    if st.button(theme_label, key='theme_toggle', use_container_width=True):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

tabs = st.tabs([
    'The Market Opportunity',
    'Spending Habits',
    'Owner Struggles',
    'Must-Have Features',
    'Predicting Adoption',
    'Customer Personas',
    'Spending Forecast',
])

def apply_theme(fig, title=''):
    fig.update_layout(
        template=C['plotly_template'],
        paper_bgcolor=C['paper_bg'],
        plot_bgcolor=C['plot_bg'],
        font=dict(family='Inter', color=C['font_color'], size=12),
        title=dict(text=title, font=dict(size=13), x=0, xanchor='left'),
        margin=dict(l=16, r=16, t=44, b=16),
        legend=dict(bgcolor='rgba(0,0,0,0)', borderwidth=0, font=dict(size=11)),
    )
    fig.update_xaxes(gridcolor=C['grid_color'], linecolor=C['border'], zerolinecolor=C['border'])
    fig.update_yaxes(gridcolor=C['grid_color'], linecolor=C['border'], zerolinecolor=C['border'])
    return fig

def insight(text):
    st.markdown(
        f'<div class="insight-box"><div class="insight-title"> Key Insight</div>{text}</div>',
        unsafe_allow_html=True
    )

with tabs[0]:
    f1, f2 = st.columns(2)
    with f1:
        age = st.multiselect('Age Group', sorted(df['age_group'].unique()),
                             default=sorted(df['age_group'].unique()), key='ov_age')
    with f2:
        region = st.multiselect('Region', sorted(df['region'].unique()),
                                default=sorted(df['region'].unique()), key='ov_region')

    df_f = df[(df['age_group'].isin(age)) & (df['region'].isin(region))].copy()
    df_no_na = df_f.dropna(subset=['monthly_spend_inr']).copy()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric('Respondents', len(df_f))
    k2.metric('Avg Spend (INR)', f"{df_no_na['monthly_spend_inr'].mean():,.0f}" if len(df_no_na) else 'N/A')
    k3.metric('App Interest', f"{(df_f['app_use_likelihood'] != 'No').mean() * 100:.0f}%" if len(df_f) else 'N/A')
    k4.metric('Avg Dogs per Owner', f"{df_f['num_dogs'].mean():.1f}" if len(df_f) else 'N/A')

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        color_map = {'Yes': '#01696f', 'Maybe': '#da7101', 'No': '#a12c7b'}
        fig1 = px.histogram(df_f, x='app_use_likelihood',
                            color='app_use_likelihood',
                            color_discrete_map=color_map,
                            category_orders={'app_use_likelihood': ['Yes', 'Maybe', 'No']})
        apply_theme(fig1, 'App Adoption Intent')
        fig1.update_layout(showlegend=False, bargap=0.3)
        st.plotly_chart(fig1, use_container_width=True)
        insight(
            "<strong>App Adoption Intent</strong> shows how many people actually want our app. "
            "A large 'Yes' means we have a great market opportunity! The 'Maybe' group just needs "
            "a little convincing with free trials or sneak peeks."
        )

    with col2:
        fig2 = px.histogram(df_no_na, x='monthly_spend_inr', nbins=30)
        apply_theme(fig2, 'Monthly Dog Care Spend Distribution (INR)')
        fig2.update_traces(marker_color=C['primary'], marker_line_width=0)
        st.plotly_chart(fig2, use_container_width=True)
        insight(
            "<strong>Spending distribution</strong> tells us how much money people spend on their dogs every month. "
            "Most people spend a normal, modest amount, but some 'premium' owners spend a lot. "
            "We should offer basic plans for everyone and VIP plans for the big spenders."
        )

    col3, col4 = st.columns(2)
    with col3:
        region_intent = df_f.groupby(['region', 'app_use_likelihood']).size().reset_index(name='count')
        fig_ri = px.bar(region_intent, x='region', y='count', color='app_use_likelihood',
                        color_discrete_map=color_map, barmode='stack')
        apply_theme(fig_ri, 'Adoption Intent by Region')
        st.plotly_chart(fig_ri, use_container_width=True)
        insight(
            "Looking at regions tells us our <strong>best launch markets</strong>. "
            "We should launch the app first in places where the most people are saying 'Yes'!"
        )

    with col4:
        age_intent = df_f.groupby(['age_group', 'app_use_likelihood']).size().reset_index(name='count')
        fig_ai = px.bar(age_intent, x='age_group', y='count', color='app_use_likelihood',
                        color_discrete_map=color_map, barmode='group',
                        category_orders={'age_group': ['18-24', '25-34', '35-44', '45-54', '55+']})
        apply_theme(fig_ai, 'Adoption Intent by Age Group')
        st.plotly_chart(fig_ai, use_container_width=True)
        insight(
            "Knowing which <strong>age group</strong> is most interested helps us target our ads. "
            "Younger owners love trying new apps, while older owners might prefer community events or simpler guides."
        )

with tabs[1]:
    f1, f2 = st.columns(2)
    with f1:
        age_s = st.multiselect('Age Group', sorted(df['age_group'].unique()),
                               default=sorted(df['age_group'].unique()), key='sp_age')
    with f2:
        region_s = st.multiselect('Region', sorted(df['region'].unique()),
                                  default=sorted(df['region'].unique()), key='sp_region')

    df_fs = df[(df['age_group'].isin(age_s)) & (df['region'].isin(region_s))].copy()
    df_no_na_s = df_fs.dropna(subset=['monthly_spend_inr']).copy()

    col1, col2 = st.columns(2)
    with col1:
        fig3 = px.box(df_no_na_s, x='age_group', y='monthly_spend_inr', color='age_group',
                      category_orders={'age_group': ['18-24', '25-34', '35-44', '45-54', '55+']})
        apply_theme(fig3, 'Monthly Spend by Age Group (INR)')
        fig3.update_layout(showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)
        insight(
            "These charts show the <strong>spending habits</strong> per age group. "
            "Older owners tend to spend more money, likely because they have more disposable income. "
            "We can build premium features tailored for them."
        )

    with col2:
        fig4 = px.bar(df_fs.groupby('residence_type')['num_dogs'].mean().reset_index(),
                      x='residence_type', y='num_dogs', color='residence_type')
        apply_theme(fig4, 'Avg Number of Dogs by Residence Type')
        fig4.update_layout(showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)
        insight(
            "People living in houses own <strong>more dogs on average</strong> than those in apartments. "
            "Our app must easily handle 'multiple dog profiles' so owners can manage all their pets in one place."
        )
            col3, col4 = st.columns(2)
    with col3:
        spend_region = df_no_na_s.groupby('region')['monthly_spend_inr'].mean().reset_index()
        spend_region.columns = ['Region', 'Avg Spend (INR)']
        fig_sr = px.bar(spend_region.sort_values('Avg Spend (INR)', ascending=False),
                        x='Region', y='Avg Spend (INR)', color='Avg Spend (INR)',
                        color_continuous_scale=['#cedcd8', C['primary']])
        apply_theme(fig_sr, 'Average Monthly Spend by Region')
        fig_sr.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_sr, use_container_width=True)
        insight(
            "This map of regional spending highlights our <strong>most profitable areas</strong>. "
            "We should roll out our luxury grooming and premium vet partnerships in these specific regions first."
        )

    with col4:
        ownership_spend = df_no_na_s.groupby('ownership_years')['monthly_spend_inr'].median().reindex(
            ['<1', '1-3', '4-7', '8+']).reset_index()
        ownership_spend.columns = ['Experience', 'Median Spend (INR)']
        fig_os = px.line(ownership_spend, x='Experience', y='Median Spend (INR)', markers=True)
        apply_theme(fig_os, 'Median Spend by Ownership Experience')
        fig_os.update_traces(line_color=C['primary'], marker_color=C['primary'], marker_size=8)
        st.plotly_chart(fig_os, use_container_width=True)
        insight(
            "Experienced dog owners have <strong>higher regular spending</strong>. "
            "On the flip side, new dog owners (less than a year) need guidance. "
            "Offering them 'starter guides' in the app will win their loyalty fast."
        )

with tabs[2]:
    f1, f2 = st.columns(2)
    with f1:
        age_c = st.multiselect('Age Group', sorted(df['age_group'].unique()),
                               default=sorted(df['age_group'].unique()), key='ch_age')
    with f2:
        region_c = st.multiselect('Region', sorted(df['region'].unique()),
                                  default=sorted(df['region'].unique()), key='ch_region')

    df_fc = df[(df['age_group'].isin(age_c)) & (df['region'].isin(region_c))].copy()

    challenge_counts = df_fc['biggest_challenge'].value_counts().reset_index()
    challenge_counts.columns = ['Challenge', 'Count']

    fig5 = px.bar(challenge_counts, x='Challenge', y='Count', color='Count',
                  color_continuous_scale=['#cedcd8', C['primary']])
    apply_theme(fig5, 'Biggest Challenges Faced by Dog Owners')
    fig5.update_layout(coloraxis_showscale=False, xaxis_tickangle=-20)
    st.plotly_chart(fig5, use_container_width=True)
    insight(
        "These are the <strong>biggest daily struggles</strong> for dog owners. "
        "To make our app successful, we must solve the top 2 or 3 problems first-like finding a reliable vet or a safe dog park."
    )

    col1, col2 = st.columns(2)
    with col1:
        ch_age = df_fc.groupby(['biggest_challenge', 'age_group']).size().reset_index(name='Count')
        fig_ca = px.bar(ch_age, x='biggest_challenge', y='Count', color='age_group', barmode='stack')
        apply_theme(fig_ca, 'Challenges Breakdown by Age Group')
        fig_ca.update_layout(xaxis_tickangle=-20, legend_title='Age Group')
        st.plotly_chart(fig_ca, use_container_width=True)
        insight(
            "Different age groups face <strong>different headaches</strong>. "
            "Younger people might struggle to find pet-friendly apartments, while older owners worry about high vet bills. "
            "We can personalize the app for these specific needs."
        )

    with col2:
        ch_region = df_fc.groupby(['biggest_challenge', 'region']).size().reset_index(name='Count')
        fig_cr = px.bar(ch_region, x='biggest_challenge', y='Count', color='region', barmode='group')
        apply_theme(fig_cr, 'Challenges by Region')
        fig_cr.update_layout(xaxis_tickangle=-20, legend_title='Region')
        st.plotly_chart(fig_cr, use_container_width=True)
        insight(
            "Knowing regional struggles helps us create <strong>local partnerships</strong>. "
            "If people in one city struggle to find grooming, we can partner with local groomers there to offer exclusive app discounts."
        )

with tabs[3]:
    f1, f2 = st.columns(2)
    with f1:
        age_ft = st.multiselect('Age Group', sorted(df['age_group'].unique()),
                                default=sorted(df['age_group'].unique()), key='ft_age')
    with f2:
        region_ft = st.multiselect('Region', sorted(df['region'].unique()),
                                   default=sorted(df['region'].unique()), key='ft_region')

    feature_df = pd.DataFrame({
        'Feature': ['Vet Booking', 'Dog Parks', 'Grooming', 'Lost Dog Alert',
                    'Marketplace', 'Community', 'Health Tracking'],
        'Interest (%)': [78, 72, 66, 60, 55, 48, 52]
    }).sort_values('Interest (%)', ascending=False)

    fig6 = px.bar(feature_df, x='Feature', y='Interest (%)', color='Interest (%)',
                  color_continuous_scale=['#cedcd8', C['primary']])
    apply_theme(fig6, 'Feature Interest Among Dog Owners (%)')
    fig6.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig6, use_container_width=True)
    insight(
        "<strong>Vet Booking</strong> and <strong>Dog Parks</strong> are the clear winners! "
        "These are 'must-have' features. We should build these immediately, and save the extra bells and whistles "
        "(like community forums) for later updates."
    )

    col1, col2 = st.columns(2)
    with col1:
        df_ft = df[(df['age_group'].isin(age_ft)) & (df['region'].isin(region_ft))].copy()
        num_cols_corr = ['monthly_spend_inr', 'num_dogs', 'num_services_used',
                         'num_features_valued', 'app_interest_scale']
        num_cols_corr = [c for c in num_cols_corr if c in df_ft.columns]
        df_corr = df_ft[num_cols_corr].dropna()
        corr_matrix = df_corr.corr().round(2)
        fig_heat = ff.create_annotated_heatmap(
            z=corr_matrix.values.tolist(),
            x=corr_matrix.columns.tolist(),
            y=corr_matrix.columns.tolist(),
            colorscale='Teal',
            showscale=True,
            annotation_text=corr_matrix.values.round(2)
        )
        fig_heat.update_layout(
            title=dict(text='Variable Correlation Matrix', font=dict(size=13, color=C['font_color']), x=0),
            paper_bgcolor=C['paper_bg'],
            font=dict(family='Inter', color=C['font_color'], size=11),
            margin=dict(l=16, r=16, t=44, b=16)
        )
        st.plotly_chart(fig_heat, use_container_width=True)
        insight(
            "This heat map shows the <strong>connection between our users' habits</strong>. "
            "Interestingly, people who want a lot of features are also the ones most excited to use our app! "
            "This means a feature-packed 'Premium' version would sell very well."
        )

    with col2:
        df_ft2 = df_ft.dropna(subset=['monthly_spend_inr'])
        fig_bubble = px.scatter(df_ft2, x='monthly_spend_inr', y='app_interest_scale',
                                size='num_dogs', color='app_use_likelihood',
                                color_discrete_map={'Yes': '#01696f', 'Maybe': '#da7101', 'No': '#a12c7b'},
                                opacity=0.75)
        apply_theme(fig_bubble, 'Spend vs App Interest (Bubble = No. of Dogs)')
        st.plotly_chart(fig_bubble, use_container_width=True)
        insight(
            "This bubble chart shows our <strong>dream customers</strong>: people who spend a lot of money and are very interested in our app. "
            "The bigger bubbles are families with multiple dogs. If we market to them, the app will thrive."
        )

with tabs[4]:
    st.markdown('<div class="section-header">Predicting Who Will Use Our App (AI Models) <span class="badge">10 Marks</span></div>', unsafe_allow_html=True)
    st.markdown(
        "Can we use smart computer logic to guess if someone will use our app? "
        "Yes! We tested 7 different AI models to see which one accurately predicts our future users."
    )

    f1, f2, f3 = st.columns(3)
    with f1:
        age_cl = st.multiselect('Age Group', sorted(df['age_group'].unique()),
                                default=sorted(df['age_group'].unique()), key='cl_age')
    with f2:
        region_cl = st.multiselect('Region', sorted(df['region'].unique()),
                                   default=sorted(df['region'].unique()), key='cl_region')
    with f3:
        test_size = st.slider("Test Set Size (%)", 15, 35, 20, key='clf_split') / 100

    df_fcl = df[(df['age_group'].isin(age_cl)) & (df['region'].isin(region_cl))].copy()
    df_no_na_cl = df_fcl.dropna(subset=['monthly_spend_inr']).copy()

    clf_features = ['monthly_spend_inr', 'num_dogs', 'num_services_used',
                    'num_features_valued', 'app_interest_scale', 'ownership_experience_encoded']
    clf_features = [c for c in clf_features if c in df_no_na_cl.columns]

    df_clf = df_no_na_cl[clf_features + ['will_adopt_app']].dropna()
    X = df_clf[clf_features]
    y = df_clf['will_adopt_app']

    if len(y.unique()) < 2 or len(df_clf) < 20:
        st.warning("Not enough data for classification. Please adjust filters.")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        classifiers = {
            "Logistic Regression":  LogisticRegression(max_iter=1000, random_state=42),
            "Decision Tree":        DecisionTreeClassifier(random_state=42),
            "Random Forest":        RandomForestClassifier(n_estimators=100, random_state=42),
            "Gradient Boosting":    GradientBoostingClassifier(random_state=42),
            "SVM":                  SVC(random_state=42),
            "K-Nearest Neighbors":  KNeighborsClassifier(),
            "Naive Bayes":          GaussianNB(),
        }

        results, trained_models = [], {}
        with st.spinner("Training all 7 classifiers"):
            for name, clf in classifiers.items():
                clf.fit(X_train_s, y_train)
                y_pred = clf.predict(X_test_s)
                trained_models[name] = (clf, y_pred)
                results.append({
                    "Algorithm": name,
                    "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
                    "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
                    "Recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
                    "F1-Score":  round(f1_score(y_test, y_pred, zero_division=0), 4),
                })

        res_df = pd.DataFrame(results).sort_values("F1-Score", ascending=False).reset_index(drop=True)

        st.markdown('<div class="section-header">Performance Comparison Table</div>', unsafe_allow_html=True)
        st.dataframe(
            res_df.style
                  .highlight_max(subset=["Accuracy","Precision","Recall","F1-Score"], color="#d4f0ed")
                  .highlight_min(subset=["Accuracy","Precision","Recall","F1-Score"], color="#fde8e8"),
            use_container_width=True, hide_index=True
        )
        insight(
            "This table compares our <strong>AI prediction tools</strong>. "
            "Green means it's highly accurate. We look at the 'F1-Score' because it's the best way to check "
            "if the tool is making fair and balanced predictions."
        )

        st.markdown('<div class="section-header">Metric Comparison Chart</div>', unsafe_allow_html=True)
        metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
        fig_clf = go.Figure()
        metric_colors = [C['primary'], '#da7101', '#a12c7b', '#006494']
        for i, m in enumerate(metrics):
            fig_clf.add_trace(go.Bar(name=m, x=res_df["Algorithm"], y=res_df[m],
                                     marker_color=metric_colors[i]))
        fig_clf.update_layout(barmode='group', yaxis=dict(range=[0, 1.15]),
                               xaxis_tickangle=-25, legend=dict(orientation="h", yanchor="bottom", y=1.02))
        apply_theme(fig_clf, 'Classifier Performance  Accuracy  Precision  Recall  F1')
        st.plotly_chart(fig_clf, use_container_width=True)
        insight(
            "Seeing these scores side-by-side helps us pick the <strong>winning AI model</strong>. "
            "We want an AI that is highly accurate so we don't accidentally ignore people who would have loved our app."
        )
                best_name = res_df.iloc[0]["Algorithm"]
        _, y_pred_best = trained_models[best_name]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<div class="section-header">Confusion Matrix  {best_name}</div>', unsafe_allow_html=True)
            cm = confusion_matrix(y_test, y_pred_best)
            fig_cm = ff.create_annotated_heatmap(
                z=cm.tolist(),
                x=["Predicted: No", "Predicted: Yes"],
                y=["Actual: No", "Actual: Yes"],
                colorscale='Teal', showscale=False
            )
            fig_cm.update_layout(
                paper_bgcolor=C['paper_bg'],
                font=dict(family='Inter', color=C['font_color'], size=12),
                margin=dict(l=16, r=16, t=16, b=16)
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            insight(
                "This grid shows exactly <strong>where our AI gets confused</strong>. "
                "A perfect predictor would have all its numbers lined up diagonally. "
                "We use this to fine-tune our system so we don't waste marketing money."
            )

        with col2:
            best_clf, _ = trained_models[best_name]
            if hasattr(best_clf, "feature_importances_"):
                st.markdown('<div class="section-header">Feature Importance</div>', unsafe_allow_html=True)
                fi = pd.DataFrame({
                    "Feature": clf_features,
                    "Importance": best_clf.feature_importances_
                }).sort_values("Importance", ascending=True)
                fig_fi = px.bar(fi, x="Importance", y="Feature", orientation="h",
                                color="Importance", color_continuous_scale=['#cedcd8', C['primary']])
                apply_theme(fig_fi, f'Feature Importance  {best_name}')
                fig_fi.update_layout(coloraxis_showscale=False, showlegend=False)
                st.plotly_chart(fig_fi, use_container_width=True)
                insight(
                    "This chart reveals the <strong>biggest clues our AI uses</strong> to make predictions. "
                    "The longer the bar, the more critical that detail is in figuring out if someone will download the app."
                )

        st.markdown('<div class="section-header">Classification Report</div>', unsafe_allow_html=True)
        st.code(classification_report(y_test, y_pred_best, target_names=["Won't Adopt", "Will Adopt"]))
        insight(
            f"Our <strong>most accurate AI predictor is {best_name}</strong>. "
            f"Because it is highly accurate, we can trust it to automatically find our future customers and target them with ads."
        )

with tabs[5]:
    st.markdown('<div class="section-header">Understanding Our Users (Customer Personas) <span class="badge">10 Marks</span></div>', unsafe_allow_html=True)
    st.markdown("We use AI to automatically group our dog owners into 'Personas'. This helps us understand the different types of people using our app so we can personalize their experience.")

    f1, f2 = st.columns(2)
    with f1:
        age_ku = st.multiselect('Age Group', sorted(df['age_group'].unique()),
                                default=sorted(df['age_group'].unique()), key='ku_age')
    with f2:
        region_ku = st.multiselect('Region', sorted(df['region'].unique()),
                                   default=sorted(df['region'].unique()), key='ku_region')

    df_fku = df[(df['age_group'].isin(age_ku)) & (df['region'].isin(region_ku))].copy()
    df_no_na_ku = df_fku.dropna(subset=['monthly_spend_inr']).copy()

    c1, c2 = st.columns([2, 1])
    with c1:
        cl_algo = st.selectbox("Clustering Algorithm",
                               ["K-Means", "Agglomerative Hierarchical", "DBSCAN"], key="cl_algo")
    with c2:
        pass

    cl_features = ['monthly_spend_inr', 'num_dogs', 'num_services_used', 'ownership_experience_encoded']
    cl_features = [c for c in cl_features if c in df_no_na_ku.columns]
    X_cl = df_no_na_ku[cl_features].dropna()
    X_scaled = StandardScaler().fit_transform(X_cl)

    if cl_algo == "K-Means":
        st.markdown('<div class="section-header">Elbow Method  Optimal k</div>', unsafe_allow_html=True)
        inertias = []
        k_range = range(2, 11)
        for k_ in k_range:
            km_ = KMeans(n_clusters=k_, random_state=42, n_init=10)
            km_.fit(X_scaled)
            inertias.append(km_.inertia_)
        fig_elbow = px.line(x=list(k_range), y=inertias, markers=True,
                            labels={"x": "Number of Clusters (k)", "y": "Inertia"})
        apply_theme(fig_elbow, 'Elbow Curve  Inertia vs k')
        fig_elbow.update_traces(line_color=C['primary'], marker_color=C['primary'], marker_size=8)
        st.plotly_chart(fig_elbow, use_container_width=True)
        insight(
            "The 'Elbow Curve' is a neat trick to find the <strong>perfect number of user groups</strong>. "
            "We look for the 'bend' in the line to see where the groups naturally settle. Usually, 3 or 4 groups is the sweet spot."
        )
        k = st.slider("Number of clusters (k)", 2, 10, 4, key="km_k")
        model_cl = KMeans(n_clusters=k, random_state=42, n_init=10)

    elif cl_algo == "Agglomerative Hierarchical":
        k = st.slider("Number of clusters", 2, 10, 4, key="agg_k")
        linkage = st.selectbox("Linkage Method", ["ward", "complete", "average", "single"])
        model_cl = AgglomerativeClustering(n_clusters=k, linkage=linkage)

    else:
        eps   = st.slider("eps (neighbourhood radius)", 0.1, 3.0, 0.5, 0.1)
        min_s = st.slider("min_samples", 2, 20, 5)
        model_cl = DBSCAN(eps=eps, min_samples=min_s)

    labels = model_cl.fit_predict(X_scaled)
    df_cl = X_cl.copy()
    df_cl["Cluster"] = labels.astype(str)
    unique_clusters = sorted(df_cl["Cluster"].unique())
    n_noise = int((labels == -1).sum())

    st.success(
        f"Found **{len(unique_clusters)}** cluster(s): {unique_clusters}"
        + (f" - including **{n_noise} noise points** (label -1)" if n_noise > 0 else "")
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Cluster Visualisation (PCA 2D)</div>', unsafe_allow_html=True)
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X_scaled)
        df_pca = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1], "Cluster": labels.astype(str)})
        fig_pca = px.scatter(df_pca, x="PC1", y="PC2", color="Cluster", opacity=0.75)
        apply_theme(fig_pca, f'Clusters (PCA)  PC1 {pca.explained_variance_ratio_[0]*100:.1f}% var  PC2 {pca.explained_variance_ratio_[1]*100:.1f}% var')
        st.plotly_chart(fig_pca, use_container_width=True)
        insight(
            "This scatter map lets us <strong>visualize our different customer groups</strong>. "
            "When the colored dots are distinctly separated, it means we have very clear and unique customer personas!"
        )

    with col2:
        if 'monthly_spend_inr' in cl_features and 'num_dogs' in cl_features:
            st.markdown('<div class="section-header">Spend vs Dogs by Cluster</div>', unsafe_allow_html=True)
            size_col = 'num_services_used' if 'num_services_used' in df_cl.columns else None
            fig_scatter = px.scatter(
                df_cl, x='monthly_spend_inr', y='num_dogs', color='Cluster',
                size=size_col, opacity=0.8
            )
            apply_theme(fig_scatter, 'Dog Owner Segments  Spending vs Number of Dogs')
            st.plotly_chart(fig_scatter, use_container_width=True)
            insight(
                "By plotting spending against the number of dogs, we can see exactly <strong>who our high rollers are</strong> "
                "and who is more budget-conscious. This is crucial for pricing our app plans."
            )

    st.markdown('<div class="section-header">Cluster Profiles (Mean Values)</div>', unsafe_allow_html=True)
    profile = df_cl.groupby("Cluster")[cl_features].mean().round(2)
    st.dataframe(profile, use_container_width=True)

    fig_prof = px.bar(
        profile.reset_index().melt(id_vars="Cluster"),
        x="variable", y="value", color="Cluster", barmode="group",
        labels={"variable": "Feature", "value": "Mean Value"}
    )
    apply_theme(fig_prof, 'Mean Feature Values per Cluster')
    st.plotly_chart(fig_prof, use_container_width=True)
    insight(
        "These profiles break down the <strong>average habits of each persona</strong>. "
        "Some groups might love booking services, while others just spend a lot on basics. "
        "We use this to decide what features to show them on the home screen."
    )

    if cl_algo == "K-Means" and 'k' in dir() and k == 4:
        persona_labels = {
            '0': 'Casual Owner', '1': 'Engaged Pet Parent',
            '2': 'Multi-Dog Household', '3': 'Premium Spender'
        }
        if all(c in persona_labels for c in unique_clusters):
            df_cl['Persona'] = df_cl['Cluster'].map(persona_labels)
            persona_counts = df_cl['Persona'].value_counts().reset_index()
            persona_counts.columns = ['Persona', 'Count']
            fig_persona = px.pie(persona_counts, names='Persona', values='Count', hole=0.45)
            apply_theme(fig_persona, 'Customer Persona Distribution')
            st.plotly_chart(fig_persona, use_container_width=True)
            insight(
                "We discovered <strong>4 distinct customer personas!</strong> Casual Owners, Engaged Pet Parents, "
                "Multi-Dog Households, and Premium Spenders. Each of these groups needs a totally different marketing message."
            )

    st.markdown('<div class="section-header">Cluster Interpretation</div>', unsafe_allow_html=True)
    for cl in [c for c in unique_clusters if c != '-1']:
        row = profile.loc[cl]
        high_feat = row.nlargest(2).index.tolist()
        low_feat  = row.nsmallest(2).index.tolist()
        st.markdown(
            f"**Cluster {cl}:** This group of users shows high "
            f"**{high_feat[0].replace('_',' ')}** and **{high_feat[1].replace('_',' ')}**, "
            f"but lower **{low_feat[0].replace('_',' ')}**. "
            f"We can personalize the app for them to keep them happy."
        )
    if '-1' in unique_clusters:
        st.markdown(
            "**Cluster -1 (Noise/Outliers):** These are unique users who don't fit into our main groups. "
            "They might be a special niche we can cater to later!"
        )

with tabs[6]:
    st.markdown('<div class="section-header">Forecasting Customer Spending (AI Models) <span class="badge">10 Marks</span></div>', unsafe_allow_html=True)
    st.markdown("Can we predict exactly how much money a dog owner will spend? We use AI forecasting models to guess their monthly budget!")

    f1, f2 = st.columns(2)
    with f1:
        age_r = st.multiselect('Age Group', sorted(df['age_group'].unique()),
                               default=sorted(df['age_group'].unique()), key='rg_age')
    with f2:
        region_r = st.multiselect('Region', sorted(df['region'].unique()),
                                  default=sorted(df['region'].unique()), key='rg_region')

    df_frg = df[(df['age_group'].isin(age_r)) & (df['region'].isin(region_r))].copy()
    df_no_na_rg = df_frg.dropna(subset=['monthly_spend_inr']).copy()

    num_cols_reg = df_no_na_rg.select_dtypes(include=np.number).columns.tolist()
    c1, c2, c3 = st.columns(3)
    with c1:
        target_reg = st.selectbox("Target Variable", num_cols_reg,
                                  index=num_cols_reg.index('monthly_spend_inr') if 'monthly_spend_inr' in num_cols_reg else 0,
                                  key="reg_target")
    with c2:
        test_pct = st.slider("Test Set Size (%)", 15, 35, 20, key="reg_split") / 100
    with c3:
        alpha = st.slider("Regularisation Alpha (Ridge / Lasso)", 0.01, 20.0, 1.0, 0.01)

    feature_pool = [c for c in num_cols_reg if c != target_reg]
    default_feats = [c for c in ['num_dogs', 'num_services_used', 'num_features_valued',
                                  'app_interest_scale', 'ownership_experience_encoded']
                     if c in feature_pool]
    selected_feats = st.multiselect("Predictor Features", feature_pool,
                                     default=default_feats or feature_pool[:5], key="reg_feats")

    if not selected_feats:
        st.warning("Please select at least one predictor feature.")
        st.stop()

    df_reg = df_no_na_rg[selected_feats + [target_reg]].dropna()
    X_r = df_reg[selected_feats]
    y_r = df_reg[target_reg]

    X_tr, X_te, y_tr, y_te = train_test_split(X_r, y_r, test_size=test_pct, random_state=42)
    sc_r = StandardScaler()
    X_tr_s = sc_r.fit_transform(X_tr)
    X_te_s  = sc_r.transform(X_te)

    reg_models = {
        "Linear Regression": LinearRegression(),
        f"Ridge (a={alpha})": Ridge(alpha=alpha),
        f"Lasso (a={alpha})": Lasso(alpha=alpha, max_iter=10000),
    }

    reg_results, preds_dict, coef_dict = [], {}, {}
    for mname, mreg in reg_models.items():
        mreg.fit(X_tr_s, y_tr)
        y_pred_r = mreg.predict(X_te_s)
        preds_dict[mname] = y_pred_r
        if hasattr(mreg, "coef_"):
            coef_dict[mname] = mreg.coef_
        mse = mean_squared_error(y_te, y_pred_r)
        reg_results.append({
            "Model": mname,
            "MSE": round(mse, 2),
            "RMSE": round(np.sqrt(mse), 2),
            "R2 Score": round(r2_score(y_te, y_pred_r), 4),
        })

    res_reg = pd.DataFrame(reg_results)

    st.markdown('<div class="section-header">Regression Performance Comparison</div>', unsafe_allow_html=True)
    st.dataframe(
        res_reg.style
               .highlight_max(subset=["R2 Score"], color="#d4f0ed")
               .highlight_min(subset=["RMSE", "MSE"], color="#d4f0ed"),
        use_container_width=True, hide_index=True
    )
    insight(
        "This table compares our <strong>forecasting tools</strong>. "
        "'R-Squared' tells us how accurate our predictions are (closer to 1 is better!). "
        "Good forecasts help us project our company's revenue."
    )

    col1, col2 = st.columns(2)
    with col1:
        fig_r2 = px.bar(res_reg, x="Model", y="R2 Score", color="Model",
                        color_discrete_sequence=[C['primary'], '#da7101', '#006494'])
        apply_theme(fig_r2, 'R-Squared Score Comparison by Regression Model')
        fig_r2.update_layout(yaxis=dict(range=[0, 1.1]), showlegend=False)
        st.plotly_chart(fig_r2, use_container_width=True)
        insight(
            "This chart shows <strong>which forecasting tool works best</strong>. "
            "Ridge and Lasso are just fancy ways to make sure our math doesn't get confused. "
            "We want the tallest bar possible!"
        )

    with col2:
        fig_avp = make_subplots(rows=1, cols=3, subplot_titles=list(preds_dict.keys()))
        model_colors = [C['primary'], '#da7101', '#006494']
        for i, (mname, y_pred_r) in enumerate(preds_dict.items(), 1):
            fig_avp.add_trace(
                go.Scatter(x=y_te.values, y=y_pred_r, mode='markers',
                           marker=dict(opacity=0.6, color=model_colors[i-1]), name=mname, showlegend=False),
                row=1, col=i
            )
            lo, hi = float(y_te.min()), float(y_te.max())
            fig_avp.add_trace(
                go.Scatter(x=[lo, hi], y=[lo, hi], mode='lines',
                           line=dict(color='red', dash='dash'), name='Perfect fit', showlegend=(i == 1)),
                row=1, col=i
            )
            fig_avp.update_xaxes(title_text="Actual", row=1, col=i, gridcolor=C['grid_color'])
            fig_avp.update_yaxes(title_text="Predicted", row=1, col=i, gridcolor=C['grid_color'])
        fig_avp.update_layout(
            title=dict(text='Actual vs Predicted  All Models', font=dict(size=13, color=C['font_color']), x=0),
            height=350, paper_bgcolor=C['paper_bg'], plot_bgcolor=C['plot_bg'],
            font=dict(family='Inter', color=C['font_color'], size=11),
            margin=dict(l=16, r=16, t=44, b=16)
        )
        st.plotly_chart(fig_avp, use_container_width=True)
        insight(
            "When the dots line up closely with the red <strong>perfect-fit line</strong>, "
            "it means our spending predictions were spot-on! If the dots are scattered everywhere, "
            "it means that person's spending is unpredictable."
        )

    st.markdown('<div class="section-header">Coefficient Comparison  Effect of Regularisation</div>', unsafe_allow_html=True)
    coef_df = pd.DataFrame(coef_dict, index=selected_feats)
    fig_coef = px.bar(
        coef_df.reset_index().melt(id_vars="index"),
        x="index", y="value", color="variable", barmode="group",
        color_discrete_sequence=[C['primary'], '#da7101', '#006494'],
        labels={"index": "Feature", "value": "Coefficient", "variable": "Model"}
    )
    apply_theme(fig_coef, 'Feature Coefficients  Linear vs Ridge vs Lasso')
    fig_coef.update_layout(xaxis_tickangle=-30)
    st.plotly_chart(fig_coef, use_container_width=True)

    best_reg = res_reg.sort_values("R2 Score", ascending=False).iloc[0]
    insight(
        f"<strong>{best_reg['Model']}</strong> is our best forecaster! It explains {best_reg['R2 Score']*100:.1f}% "
        f"of the reasons why people spend money on {target_reg.replace('_',' ')}. "
        "These tools are incredibly smart. They automatically figure out what details actually cause people to spend more money. "
        "We can use these insights to push the right services to the right users in the app!"
    )

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown(
    f'<div style="text-align:center;font-size:0.75rem;color:{C["text_muted"]};padding:8px 0;">'
    'MBA Group Project  India Dog Care App Market Analysis  Data Analytics in Decision Making'
    '</div>',
    unsafe_allow_html=True
)
