import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
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

st.set_page_config(
    page_title="PawsIndia Analytics",
    layout="wide",
    page_icon="🐾",
    initial_sidebar_state="collapsed"
)

# ── DARK THEME ONLY ──────────────────────────────────────────────────────────
C = {
    "bg":              "#0f1117",
    "surface":         "#1a1d27",
    "surface2":        "#21242f",
    "border":          "#2e3148",
    "text":            "#e8eaf0",
    "text_muted":      "#8b90a8",
    "primary":         "#4f98a3",
    "primary_soft":    "#2a5c63",
    "paper_bg":        "#1a1d27",
    "plot_bg":         "#1a1d27",
    "font_color":      "#e8eaf0",
    "grid_color":      "#2e3148",
    "accent_orange":   "#da7101",
    "accent_pink":     "#a12c7b",
    "accent_blue":     "#3b78ab",
    "plotly_template": "plotly_dark",
}

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
    .stApp {{ background-color: {C['bg']}; color: {C['text']}; }}
    header[data-testid="stHeader"] {{
        background: {C['bg']}; border-bottom: 1px solid {C['border']};
    }}
    .stTabs [data-baseweb="tab-list"] {{
        background-color: {C['surface2']}; border-radius: 8px;
        padding: 4px; gap: 2px; border: 1px solid {C['border']};
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 6px; padding: 8px 18px; font-size: 0.85rem;
        font-weight: 500; color: {C['text_muted']}; background: transparent; border: none;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {C['surface']} !important; color: {C['primary']} !important;
        font-weight: 600; box-shadow: 0 1px 4px rgba(0,0,0,0.3);
    }}
    [data-testid="metric-container"] {{
        background-color: {C['surface']}; border: 1px solid {C['border']};
        border-radius: 10px; padding: 16px 20px; box-shadow: 0 1px 6px rgba(0,0,0,0.3);
    }}
    [data-testid="metric-container"] label {{
        color: {C['text_muted']} !important; font-size: 0.72rem !important;
        font-weight: 600 !important; text-transform: uppercase; letter-spacing: 0.06em;
    }}
    [data-testid="metric-container"] [data-testid="stMetricValue"] {{
        color: {C['text']} !important; font-size: 1.55rem !important; font-weight: 700 !important;
    }}
    .insight-box {{
        background-color: {C['surface']}; border: 1px solid {C['border']};
        border-left: 3px solid {C['primary']}; border-radius: 8px;
        padding: 14px 18px; margin-top: 12px; font-size: 0.84rem;
        color: {C['text_muted']}; line-height: 1.65;
    }}
    .insight-box strong {{ color: {C['text']}; }}
    .insight-title {{
        font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.1em; color: {C['primary']}; margin-bottom: 6px;
    }}
    .section-divider {{ border: none; border-top: 1px solid {C['border']}; margin: 18px 0; }}
    .section-header {{
        font-size: 0.92rem; font-weight: 600; color: {C['text']};
        margin: 20px 0 12px 0; padding-bottom: 8px; border-bottom: 1px solid {C['border']};
    }}
    .logo-wrap {{
        display: flex; align-items: center; gap: 14px; padding-bottom: 4px;
    }}
    .logo-svg {{ flex-shrink: 0; }}
    .app-name {{
        font-size: 1.3rem; font-weight: 800; color: {C['text']};
        letter-spacing: -0.02em; line-height: 1.1;
    }}
    .app-name span {{ color: {C['primary']}; }}
    .app-sub {{
        font-size: 0.75rem; color: {C['text_muted']}; font-weight: 400; margin-top: 2px;
    }}
    [data-testid="stSidebar"] {{
        background-color: {C['surface']}; border-right: 1px solid {C['border']};
    }}
    [data-testid="stDataFrame"] {{
        border: 1px solid {C['border']}; border-radius: 8px; overflow: hidden;
    }}
    .stSelectbox > div, .stMultiSelect > div {{
        background-color: {C['surface2']} !important;
    }}
    #MainMenu, footer {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)


# ── DATA ─────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "dog_app_data.csv")
    df = pd.read_csv(csv_path)
    df["ownership_experience_encoded"] = df["ownership_years"].map(
        {"<1": 1, "1-3": 2, "4-7": 3, "8+": 4}
    )
    df["will_adopt_app"] = (df["app_use_likelihood"] == "Yes").astype(int)
    return df

df = load_data()

# ── HEADER WITH LOGO ─────────────────────────────────────────────────────────
st.markdown(f"""
<div class="logo-wrap">
  <svg class="logo-svg" width="52" height="52" viewBox="0 0 52 52" fill="none" xmlns="http://www.w3.org/2000/svg">
    <rect width="52" height="52" rx="14" fill="{C['primary_soft']}"/>
    <!-- paw pad -->
    <ellipse cx="26" cy="30" rx="9" ry="7.5" fill="{C['primary']}"/>
    <!-- toes -->
    <ellipse cx="17" cy="22" rx="3.2" ry="4" fill="{C['primary']}"/>
    <ellipse cx="23" cy="19" rx="3.2" ry="4" fill="{C['primary']}"/>
    <ellipse cx="29" cy="19" rx="3.2" ry="4" fill="{C['primary']}"/>
    <ellipse cx="35" cy="22" rx="3.2" ry="4" fill="{C['primary']}"/>
  </svg>
  <div>
    <div class="app-name">Paws<span>India</span> Analytics</div>
    <div class="app-sub">MBA &nbsp;|&nbsp; Data Analytics in Decision Making &nbsp;|&nbsp; Group Project Dashboard</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ── TABS ─────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🗺 Market Opportunity",
    "💰 Spending Habits",
    "😤 Owner Struggles",
    "⭐ Must-Have Features",
    "🤖 Predicting Adoption",
    "👥 Customer Personas",
    "📈 Spending Forecast",
])


# ── HELPERS ──────────────────────────────────────────────────────────────────
def apply_theme(fig, title=""):
    fig.update_layout(
        template=C["plotly_template"],
        paper_bgcolor=C["paper_bg"],
        plot_bgcolor=C["plot_bg"],
        font=dict(family="Inter", color=C["font_color"], size=12),
        title=dict(text=title, font=dict(size=13, color=C["font_color"]), x=0, xanchor="left"),
        margin=dict(l=16, r=16, t=44, b=16),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0, font=dict(size=11)),
    )
    fig.update_xaxes(gridcolor=C["grid_color"], linecolor=C["border"], zerolinecolor=C["border"])
    fig.update_yaxes(gridcolor=C["grid_color"], linecolor=C["border"], zerolinecolor=C["border"])
    return fig


def insight(text):
    st.markdown(
        f'<div class="insight-box"><div class="insight-title">💡 Key Insight</div>{text}</div>',
        unsafe_allow_html=True
    )

COLOR_MAP = {"Yes": "#4f98a3", "Maybe": "#da7101", "No": "#a12c7b"}

# India region → approximate lat/lon centroids
REGION_COORDS = {
    "North":     {"lat": 28.7, "lon": 77.1,  "state": "Delhi/UP"},
    "South":     {"lat": 13.0, "lon": 80.2,  "state": "Tamil Nadu/Karnataka"},
    "East":      {"lat": 22.5, "lon": 88.3,  "state": "West Bengal/Odisha"},
    "West":      {"lat": 19.0, "lon": 72.8,  "state": "Maharashtra/Gujarat"},
    "Central":   {"lat": 23.2, "lon": 77.4,  "state": "MP/Chhattisgarh"},
    "Northeast": {"lat": 26.1, "lon": 91.7,  "state": "Assam/Meghalaya"},
}


# ── TAB 0 : MARKET OPPORTUNITY ───────────────────────────────────────────────
with tabs[0]:
    f1, f2 = st.columns(2)
    with f1:
        age = st.multiselect("Age Group", sorted(df["age_group"].unique()),
                             default=sorted(df["age_group"].unique()), key="ov_age")
    with f2:
        region = st.multiselect("Region", sorted(df["region"].unique()),
                                default=sorted(df["region"].unique()), key="ov_region")

    df_f = df[(df["age_group"].isin(age)) & (df["region"].isin(region))].copy()
    df_no_na = df_f.dropna(subset=["monthly_spend_inr"]).copy()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Respondents", len(df_f))
    k2.metric("Avg Monthly Spend (INR)",
              f"{df_no_na['monthly_spend_inr'].mean():,.0f}" if len(df_no_na) else "N/A")
    k3.metric("Interested in App",
              f"{(df_f['app_use_likelihood'] != 'No').mean()*100:.0f}%" if len(df_f) else "N/A")
    k4.metric("Avg Dogs per Owner",
              f"{df_f['num_dogs'].mean():.1f}" if len(df_f) else "N/A")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── GEOGRAPHICAL MAP ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📍 Region-wise App Adoption Map — Click a bubble to explore</div>',
                unsafe_allow_html=True)

    # Build per-region summary
    map_rows = []
    for reg in df_f["region"].unique():
        sub = df_f[df_f["region"] == reg]
        sub_spend = sub.dropna(subset=["monthly_spend_inr"])
        yes_pct  = round((sub["app_use_likelihood"] == "Yes").mean() * 100, 1)
        maybe_pct= round((sub["app_use_likelihood"] == "Maybe").mean() * 100, 1)
        no_pct   = round((sub["app_use_likelihood"] == "No").mean() * 100, 1)
        avg_spend= round(sub_spend["monthly_spend_inr"].mean(), 0) if len(sub_spend) else 0
        coords   = REGION_COORDS.get(reg, {"lat": 22, "lon": 78, "state": reg})
        map_rows.append({
            "Region": reg,
            "State": coords["state"],
            "lat": coords["lat"],
            "lon": coords["lon"],
            "Respondents": len(sub),
            "Will Use App (%)": yes_pct,
            "Maybe (%)": maybe_pct,
            "Won't Use (%)": no_pct,
            "Avg Spend (INR)": avg_spend,
            "Bubble Label": (
                f"<b>{reg}</b><br>"
                f"Respondents: {len(sub)}<br>"
                f"Will Use App: {yes_pct}%<br>"
                f"Avg Spend: ₹{avg_spend:,.0f}"
            ),
        })
    map_df = pd.DataFrame(map_rows)

    fig_map = go.Figure()

    # Distinct colour per region for easy visual separation
    REGION_COLORS = {
        "North":     "#4f98a3",
        "South":     "#da7101",
        "East":      "#6daa45",
        "West":      "#a86fdf",
        "Central":   "#e8af34",
        "Northeast": "#dd6974",
    }

    for _, row in map_df.iterrows():
        reg   = row["Region"]
        color = REGION_COLORS.get(reg, "#4f98a3")
        hover = (
            f"<b style='font-size:14px'>{reg}</b><br>"
            f"<span style='color:#aaa'>📍 {row['State']}</span><br><br>"
            f"👥 Respondents: <b>{row['Respondents']}</b><br>"
            f"✅ Will Use App: <b>{row['Will Use App (%)']}%</b><br>"
            f"🤔 Maybe: <b>{row['Maybe (%)']}%</b><br>"
            f"❌ Won't Use: <b>{row['Won\'t Use (%)']:.1f}%</b><br>"
            f"💰 Avg Spend: <b>₹{row['Avg Spend (INR)']:,.0f}</b>"
        )
        bubble_size = row["Respondents"] / map_df["Respondents"].max() * 44 + 16
        fig_map.add_trace(go.Scattergeo(
            lat=[row["lat"]],
            lon=[row["lon"]],
            name=reg,
            text=[f"<b>{reg}</b>"],
            hovertemplate=hover + "<extra></extra>",
            mode="markers+text",
            textposition="top center",
            textfont=dict(color=color, size=12, family="Inter"),
            marker=dict(
                size=bubble_size,
                color=color,
                opacity=0.82,
                line=dict(color="#0f1117", width=2),
            ),
            showlegend=True,
        ))

    fig_map.update_layout(
        geo=dict(
            scope="asia",
            showland=True,
            landcolor="#1a1d27",
            showocean=True,
            oceancolor="#0f1117",
            showcountries=True,
            countrycolor="#2e3148",
            showframe=False,
            showcoastlines=True,
            coastlinecolor="#2e3148",
            projection_type="natural earth",
            center=dict(lat=22.5, lon=80.0),
            lataxis=dict(range=[6, 38]),
            lonaxis=dict(range=[65, 100]),
            bgcolor="#0f1117",
        ),
        paper_bgcolor="#0f1117",
        plot_bgcolor="#0f1117",
        font=dict(family="Inter", color="#e8eaf0", size=12),
        margin=dict(l=0, r=0, t=8, b=0),
        height=500,
    )
    st.plotly_chart(fig_map, use_container_width=True)
    insight(
        "<strong>Hover or click any bubble</strong> to see that region's adoption rate, "
        "average spend, and survey breakdown. <strong>Larger bubbles = more respondents.</strong> "
        "Brighter colour = higher app interest. Use this to decide <strong>where to launch first</strong> "
        "and where to invest in marketing."
    )

    # Regional summary table (clickable context)
    with st.expander("📊 Full Region-wise Breakdown Table"):
        st.dataframe(
            map_df[["Region", "State", "Respondents",
                    "Will Use App (%)", "Maybe (%)", "Won't Use (%)", "Avg Spend (INR)"]]
            .sort_values("Will Use App (%)", ascending=False)
            .reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.histogram(df_f, x="app_use_likelihood", color="app_use_likelihood",
                            color_discrete_map=COLOR_MAP,
                            category_orders={"app_use_likelihood": ["Yes", "Maybe", "No"]})
        apply_theme(fig1, "Would You Use a Dog Care App?")
        fig1.update_layout(showlegend=False, bargap=0.3,
                           xaxis_title="Response", yaxis_title="No. of Dog Owners")
        st.plotly_chart(fig1, use_container_width=True)
        insight(
            "A large <strong>'Yes'</strong> bar means real demand for our product! "
            "The 'Maybe' group just needs a free trial or sneak peek to convert."
        )

    with col2:
        fig2 = px.histogram(df_no_na, x="monthly_spend_inr", nbins=30)
        apply_theme(fig2, "Monthly Dog Care Spend Distribution (INR)")
        fig2.update_traces(marker_color=C["primary"], marker_line_width=0)
        fig2.update_layout(xaxis_title="Monthly Spend (INR)", yaxis_title="No. of Dog Owners")
        st.plotly_chart(fig2, use_container_width=True)
        insight(
            "Most owners spend a <strong>moderate amount</strong>, but premium owners spend significantly more. "
            "A tiered pricing model (Free + VIP) will capture both segments."
        )

    col3, col4 = st.columns(2)
    with col3:
        region_intent = df_f.groupby(["region", "app_use_likelihood"]).size().reset_index(name="count")
        fig_ri = px.bar(region_intent, x="region", y="count", color="app_use_likelihood",
                        color_discrete_map=COLOR_MAP, barmode="stack")
        apply_theme(fig_ri, "Adoption Intent by Region")
        fig_ri.update_layout(xaxis_title="Region", yaxis_title="No. of Dog Owners",
                             legend_title="Response")
        st.plotly_chart(fig_ri, use_container_width=True)
        insight(
            "Regions with the tallest <strong>teal bars</strong> are our best launch markets — "
            "focus marketing budgets here first."
        )

    with col4:
        age_intent = df_f.groupby(["age_group", "app_use_likelihood"]).size().reset_index(name="count")
        fig_ai = px.bar(age_intent, x="age_group", y="count", color="app_use_likelihood",
                        color_discrete_map=COLOR_MAP, barmode="group",
                        category_orders={"age_group": ["18-24","25-34","35-44","45-54","55+"]})
        apply_theme(fig_ai, "Adoption Intent by Age Group")
        fig_ai.update_layout(xaxis_title="Age Group", yaxis_title="No. of Dog Owners",
                             legend_title="Response")
        st.plotly_chart(fig_ai, use_container_width=True)
        insight(
            "Younger owners (18-34) are quickest to adopt new apps. "
            "Older users may need <strong>simpler onboarding</strong> or social proof to convert."
        )


# ── TAB 1 : SPENDING HABITS ──────────────────────────────────────────────────
with tabs[1]:
    f1, f2 = st.columns(2)
    with f1:
        age_s = st.multiselect("Age Group", sorted(df["age_group"].unique()),
                               default=sorted(df["age_group"].unique()), key="sp_age")
    with f2:
        region_s = st.multiselect("Region", sorted(df["region"].unique()),
                                  default=sorted(df["region"].unique()), key="sp_region")

    df_fs = df[(df["age_group"].isin(age_s)) & (df["region"].isin(region_s))].copy()
    df_no_na_s = df_fs.dropna(subset=["monthly_spend_inr"]).copy()

    col1, col2 = st.columns(2)
    with col1:
        fig3 = px.box(df_no_na_s, x="age_group", y="monthly_spend_inr", color="age_group",
                      category_orders={"age_group": ["18-24","25-34","35-44","45-54","55+"]})
        apply_theme(fig3, "Monthly Spend by Age Group (INR)")
        fig3.update_layout(showlegend=False,
                           xaxis_title="Age Group", yaxis_title="Monthly Spend (INR)")
        st.plotly_chart(fig3, use_container_width=True)
        insight(
            "Older owners have <strong>higher and wider spending ranges</strong>. "
            "Premium subscription plans should target the 35+ segment first."
        )

    with col2:
        fig4 = px.bar(df_fs.groupby("residence_type")["num_dogs"].mean().reset_index(),
                      x="residence_type", y="num_dogs", color="residence_type")
        apply_theme(fig4, "Avg Dogs by Residence Type")
        fig4.update_layout(showlegend=False,
                           xaxis_title="Residence Type", yaxis_title="Avg No. of Dogs")
        st.plotly_chart(fig4, use_container_width=True)
        insight(
            "House owners keep <strong>more dogs on average</strong>. "
            "A multi-pet profile feature would be a strong selling point for this segment."
        )

    col3, col4 = st.columns(2)
    with col3:
        spend_region = df_no_na_s.groupby("region")["monthly_spend_inr"].mean().reset_index()
        spend_region.columns = ["Region", "Avg Spend (INR)"]
        fig_sr = px.bar(spend_region.sort_values("Avg Spend (INR)", ascending=False),
                        x="Region", y="Avg Spend (INR)", color="Avg Spend (INR)",
                        color_continuous_scale=["#2a5c63", C["primary"]])
        apply_theme(fig_sr, "Average Monthly Spend by Region")
        fig_sr.update_layout(coloraxis_showscale=False,
                             xaxis_title="Region", yaxis_title="Avg Monthly Spend (INR)")
        st.plotly_chart(fig_sr, use_container_width=True)
        insight(
            "High-spending regions should get <strong>premium partnerships</strong> first — "
            "vet bookings, grooming services, and curated pet products."
        )

    with col4:
        ownership_spend = df_no_na_s.groupby("ownership_years")["monthly_spend_inr"].median().reindex(
            ["<1","1-3","4-7","8+"]).reset_index()
        ownership_spend.columns = ["Experience", "Median Spend (INR)"]
        fig_os = px.line(ownership_spend, x="Experience", y="Median Spend (INR)", markers=True)
        apply_theme(fig_os, "Median Spend by Ownership Experience")
        fig_os.update_traces(line_color=C["primary"], marker_color=C["primary"], marker_size=8)
        fig_os.update_layout(xaxis_title="Years Owning a Dog", yaxis_title="Median Spend (INR)")
        st.plotly_chart(fig_os, use_container_width=True)
        insight(
            "Spending grows with experience. <strong>New dog owners</strong> are a huge opportunity — "
            "win them early with starter guides and they will grow into premium users."
        )


# ── TAB 2 : OWNER STRUGGLES ──────────────────────────────────────────────────
with tabs[2]:
    f1, f2 = st.columns(2)
    with f1:
        age_c = st.multiselect("Age Group", sorted(df["age_group"].unique()),
                               default=sorted(df["age_group"].unique()), key="ch_age")
    with f2:
        region_c = st.multiselect("Region", sorted(df["region"].unique()),
                                  default=sorted(df["region"].unique()), key="ch_region")

    df_fc = df[(df["age_group"].isin(age_c)) & (df["region"].isin(region_c))].copy()
    challenge_counts = df_fc["biggest_challenge"].value_counts().reset_index()
    challenge_counts.columns = ["Challenge", "Count"]

    fig5 = px.bar(challenge_counts, x="Challenge", y="Count", color="Count",
                  color_continuous_scale=["#2a5c63", C["primary"]])
    apply_theme(fig5, "Biggest Daily Challenges for Dog Owners")
    fig5.update_layout(coloraxis_showscale=False, xaxis_tickangle=-20,
                       xaxis_title="Challenge", yaxis_title="No. of Dog Owners")
    st.plotly_chart(fig5, use_container_width=True)
    insight(
        "The tallest bars represent problems <strong>most owners face daily</strong>. "
        "Solving the top 2-3 challenges should be the core value proposition of our app."
    )

    col1, col2 = st.columns(2)
    with col1:
        ch_age = df_fc.groupby(["biggest_challenge","age_group"]).size().reset_index(name="Count")
        fig_ca = px.bar(ch_age, x="biggest_challenge", y="Count", color="age_group", barmode="stack")
        apply_theme(fig_ca, "Challenges by Age Group")
        fig_ca.update_layout(xaxis_tickangle=-20, legend_title="Age Group",
                             xaxis_title="Challenge", yaxis_title="Count")
        st.plotly_chart(fig_ca, use_container_width=True)
        insight(
            "Different age groups face <strong>different pain points</strong>. "
            "Personalising the app home screen per age group will improve retention."
        )

    with col2:
        ch_region = df_fc.groupby(["biggest_challenge","region"]).size().reset_index(name="Count")
        fig_cr = px.bar(ch_region, x="biggest_challenge", y="Count", color="region", barmode="group")
        apply_theme(fig_cr, "Challenges by Region")
        fig_cr.update_layout(xaxis_tickangle=-20, legend_title="Region",
                             xaxis_title="Challenge", yaxis_title="Count")
        st.plotly_chart(fig_cr, use_container_width=True)
        insight(
            "Region-specific problems enable <strong>local service partnerships</strong>. "
            "If vet access is the top issue in a region, prioritise vet booking there."
        )


# ── TAB 3 : MUST-HAVE FEATURES ───────────────────────────────────────────────
with tabs[3]:
    f1, f2 = st.columns(2)
    with f1:
        age_ft = st.multiselect("Age Group", sorted(df["age_group"].unique()),
                                default=sorted(df["age_group"].unique()), key="ft_age")
    with f2:
        region_ft = st.multiselect("Region", sorted(df["region"].unique()),
                                   default=sorted(df["region"].unique()), key="ft_region")

    feature_df = pd.DataFrame({
        "Feature": ["Vet Booking","Dog Parks","Grooming","Lost Dog Alert",
                    "Marketplace","Community","Health Tracking"],
        "Interest (%)": [78, 72, 66, 60, 55, 48, 52]
    }).sort_values("Interest (%)", ascending=False)

    fig6 = px.bar(feature_df, x="Feature", y="Interest (%)", color="Interest (%)",
                  color_continuous_scale=["#2a5c63", C["primary"]])
    apply_theme(fig6, "Which Features Do Dog Owners Want Most?")
    fig6.update_layout(coloraxis_showscale=False,
                       xaxis_title="Feature", yaxis_title="% Interested")
    st.plotly_chart(fig6, use_container_width=True)
    insight(
        "<strong>Vet Booking and Dog Parks</strong> are must-haves — nearly 7 in 10 owners want them. "
        "Build these in v1. Community and Marketplace can wait for v2."
    )

    col1, col2 = st.columns(2)
    with col1:
        df_ft = df[(df["age_group"].isin(age_ft)) & (df["region"].isin(region_ft))].copy()
        num_cols_corr = ["monthly_spend_inr","num_dogs","num_services_used",
                         "num_features_valued","app_interest_scale"]
        num_cols_corr = [c for c in num_cols_corr if c in df_ft.columns]
        df_corr = df_ft[num_cols_corr].dropna()
        corr_matrix = df_corr.corr().round(2)
        fig_heat = ff.create_annotated_heatmap(
            z=corr_matrix.values.tolist(),
            x=corr_matrix.columns.tolist(),
            y=corr_matrix.columns.tolist(),
            colorscale="Teal", showscale=True,
            annotation_text=corr_matrix.values.round(2)
        )
        fig_heat.update_layout(
            title=dict(text="Variable Correlation Matrix",
                       font=dict(size=13, color=C["font_color"]), x=0),
            paper_bgcolor=C["paper_bg"],
            font=dict(family="Inter", color=C["font_color"], size=11),
            margin=dict(l=16, r=16, t=44, b=16)
        )
        st.plotly_chart(fig_heat, use_container_width=True)
        insight(
            "Values near <strong>1.0</strong> show strong links. "
            "Owners who want more features also show higher app interest — "
            "a feature-rich Premium tier will sell well."
        )

    with col2:
        df_ft2 = df_ft.dropna(subset=["monthly_spend_inr"])
        fig_bubble = px.scatter(df_ft2, x="monthly_spend_inr", y="app_interest_scale",
                                size="num_dogs", color="app_use_likelihood",
                                color_discrete_map=COLOR_MAP, opacity=0.75)
        apply_theme(fig_bubble, "Spend vs App Interest (Bubble = No. of Dogs)")
        fig_bubble.update_layout(xaxis_title="Monthly Spend (INR)",
                                 yaxis_title="App Interest Scale",
                                 legend_title="Will They Use It?")
        st.plotly_chart(fig_bubble, use_container_width=True)
        insight(
            "Top-right corner = <strong>dream customers</strong>: high spenders who love the app idea. "
            "Larger bubbles own multiple dogs — prime targets for multi-pet premium plans."
        )


# ── TAB 4 : PREDICTING ADOPTION ──────────────────────────────────────────────
with tabs[4]:
    st.markdown('<div class="section-header">🤖 Classification Models — Predicting App Adoption</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box" style="margin-bottom:20px; border-left: 3px solid #4f98a3;">
        <div class="insight-title">🎯 What Are We Doing Here?</div>
        We are training <strong>7 machine learning classifiers</strong> to predict whether a dog owner
        will adopt our app based on their profile — spending habits, number of dogs, services used,
        and experience level.<br><br>
        <strong>Why does this matter for the business?</strong> Instead of marketing to everyone,
        we can use the best model to <em>automatically score and rank</em> potential users —
        focusing ad spend only on people most likely to convert.
        This directly reduces <strong>Customer Acquisition Cost (CAC)</strong> and improves ROI.
    </div>
    """, unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3)
    with f1:
        age_cl = st.multiselect("Age Group", sorted(df["age_group"].unique()),
                                default=sorted(df["age_group"].unique()), key="cl_age")
    with f2:
        region_cl = st.multiselect("Region", sorted(df["region"].unique()),
                                   default=sorted(df["region"].unique()), key="cl_region")
    with f3:
        test_size = st.slider("Test Set Size (%)", 15, 35, 20, key="clf_split") / 100

    df_fcl = df[(df["age_group"].isin(age_cl)) & (df["region"].isin(region_cl))].copy()
    df_no_na_cl = df_fcl.dropna(subset=["monthly_spend_inr"]).copy()

    clf_features = ["monthly_spend_inr","num_dogs","num_services_used",
                    "num_features_valued","app_interest_scale","ownership_experience_encoded"]
    clf_features = [c for c in clf_features if c in df_no_na_cl.columns]
    df_clf = df_no_na_cl[clf_features + ["will_adopt_app"]].dropna()
    X = df_clf[clf_features]
    y = df_clf["will_adopt_app"]

    if len(y.unique()) < 2 or len(df_clf) < 20:
        st.warning("Not enough data. Please adjust your filters.")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        classifiers = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Decision Tree":       DecisionTreeClassifier(random_state=42),
            "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
            "Gradient Boosting":   GradientBoostingClassifier(random_state=42),
            "SVM":                 SVC(random_state=42),
            "K-Nearest Neighbors": KNeighborsClassifier(),
            "Naive Bayes":         GaussianNB(),
        }

        results, trained_models = [], {}
        with st.spinner("Training 7 classifiers..."):
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

        st.markdown('<div class="section-header">Performance Comparison Table</div>',
                    unsafe_allow_html=True)
        st.dataframe(
            res_df.style
                  .highlight_max(subset=["Accuracy","Precision","Recall","F1-Score"], color="#1e4a4f")
                  .highlight_min(subset=["Accuracy","Precision","Recall","F1-Score"], color="#4a1e2e"),
            use_container_width=True, hide_index=True
        )
        insight(
            "<strong>Green = highest score, Red = lowest</strong> for each column. "
            "F1-Score is our key metric — it balances precision and recall for the fairest evaluation."
        )

        st.markdown('<div class="section-header">📊 Metric Comparison — One Chart per Metric</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="insight-box" style="margin-bottom:16px;">' +
            '<div class="insight-title">📖 How to Read These Charts</div>' +
            'Each chart below shows how all 7 classifiers score on a single metric. ' +
            '<strong>Accuracy</strong> = overall correct predictions. ' +
            '<strong>Precision</strong> = of those predicted as "Will Adopt", how many actually will. ' +
            '<strong>Recall</strong> = of all real adopters, how many did the model catch. ' +
            '<strong>F1-Score</strong> = harmonic mean of Precision & Recall — our primary ranking metric.' +
            '</div>',
            unsafe_allow_html=True
        )

        metric_info = {
            "Accuracy":  ("Overall correct predictions out of all predictions made.", C["primary"]),
            "Precision": ("Of users predicted to adopt, how many actually will — avoids wasted outreach.", C["accent_orange"]),
            "Recall":    ("Of all real adopters, how many did we correctly identify — avoids missed users.", C["accent_pink"]),
            "F1-Score":  ("Balances Precision and Recall. Best single metric to rank our models.", C["accent_blue"]),
        }

        col_a, col_b = st.columns(2)
        col_c, col_d = st.columns(2)
        metric_cols = [col_a, col_b, col_c, col_d]

        for idx, (metric, (desc, color)) in enumerate(metric_info.items()):
            with metric_cols[idx]:
                fig_m = go.Figure(go.Bar(
                    x=res_df["Algorithm"],
                    y=res_df[metric],
                    marker=dict(
                        color=res_df[metric],
                        colorscale=[[0, "#1a2a2a"], [1, color]],
                        showscale=False,
                        line=dict(color="#0f1117", width=1),
                    ),
                    text=[f"{v:.3f}" for v in res_df[metric]],
                    textposition="outside",
                    textfont=dict(size=10, color="#e8eaf0"),
                    hovertemplate="<b>%{x}</b><br>" + metric + ": %{y:.4f}<extra></extra>",
                ))
                fig_m.update_layout(
                    template=C["plotly_template"],
                    paper_bgcolor=C["paper_bg"],
                    plot_bgcolor=C["plot_bg"],
                    font=dict(family="Inter", color=C["font_color"], size=11),
                    title=dict(text=metric, font=dict(size=13, color=color), x=0, xanchor="left"),
                    margin=dict(l=12, r=12, t=40, b=90),
                    yaxis=dict(range=[0, 1.18], gridcolor=C["grid_color"], tickformat=".2f"),
                    xaxis=dict(tickangle=-35, linecolor=C["border"],
                               tickfont=dict(size=10)),
                    height=300,
                )
                st.plotly_chart(fig_m, use_container_width=True)
                st.markdown(
                    f'<div class="insight-box" style="margin-top:4px;padding:10px 14px;">' +
                    f'<div class="insight-title">💡 {metric}</div>{desc}</div>',
                    unsafe_allow_html=True
                )

        insight(
            "Compare all four charts together. The <strong>best model</strong> will have tall bars "
            "across all four metrics — not just one. A high-Accuracy but low-Recall model will miss "
            "real users and <strong>cost the business marketing budget</strong> on poor targeting."
        )

        best_name = res_df.iloc[0]["Algorithm"]
        _, y_pred_best = trained_models[best_name]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<div class="section-header">Confusion Matrix — {best_name}</div>',
                        unsafe_allow_html=True)
            cm = confusion_matrix(y_test, y_pred_best)
            fig_cm = ff.create_annotated_heatmap(
                z=cm.tolist(),
                x=["Predicted: No","Predicted: Yes"],
                y=["Actual: No","Actual: Yes"],
                colorscale="Teal", showscale=False
            )
            fig_cm.update_layout(
                paper_bgcolor=C["paper_bg"],
                font=dict(family="Inter", color=C["font_color"], size=12),
                margin=dict(l=16, r=16, t=16, b=16)
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            insight(
                "Diagonal = correct predictions. Off-diagonal = errors. "
                "<strong>Minimise false negatives</strong> (bottom-left) to avoid missing real users."
            )

        with col2:
            best_clf, _ = trained_models[best_name]
            if hasattr(best_clf, "feature_importances_"):
                st.markdown('<div class="section-header">Feature Importance</div>',
                            unsafe_allow_html=True)
                fi = pd.DataFrame({
                    "Feature":    clf_features,
                    "Importance": best_clf.feature_importances_
                }).sort_values("Importance", ascending=True)
                fig_fi = px.bar(fi, x="Importance", y="Feature", orientation="h",
                                color="Importance",
                                color_continuous_scale=["#2a5c63", C["primary"]])
                apply_theme(fig_fi, f"Feature Importance — {best_name}")
                fig_fi.update_layout(coloraxis_showscale=False, showlegend=False)
                st.plotly_chart(fig_fi, use_container_width=True)
                insight(
                    "Longer bar = stronger predictor. "
                    "Focus data collection on the <strong>top 2-3 features</strong> for the best ROI."
                )

        st.markdown('<div class="section-header">Full Classification Report</div>',
                    unsafe_allow_html=True)
        st.code(classification_report(y_test, y_pred_best,
                                      target_names=["Won't Adopt","Will Adopt"]))
        insight(
            f"Best model: <strong>{best_name}</strong> (F1 = {res_df.iloc[0]['F1-Score']:.4f}). "
            "Deploy this model to automatically score and rank marketing leads."
        )


# ── TAB 5 : CUSTOMER PERSONAS ────────────────────────────────────────────────
with tabs[5]:
    st.markdown('<div class="section-header">👥 Clustering — Customer Personas</div>',
                unsafe_allow_html=True)

    f1, f2 = st.columns(2)
    with f1:
        age_ku = st.multiselect("Age Group", sorted(df["age_group"].unique()),
                                default=sorted(df["age_group"].unique()), key="ku_age")
    with f2:
        region_ku = st.multiselect("Region", sorted(df["region"].unique()),
                                   default=sorted(df["region"].unique()), key="ku_region")

    df_fku = df[(df["age_group"].isin(age_ku)) & (df["region"].isin(region_ku))].copy()
    df_no_na_ku = df_fku.dropna(subset=["monthly_spend_inr"]).copy()

    c1, _ = st.columns([2,1])
    with c1:
        cl_algo = st.selectbox("Clustering Algorithm",
                               ["K-Means","Agglomerative Hierarchical","DBSCAN"], key="cl_algo")

    cl_features = ["monthly_spend_inr","num_dogs","num_services_used","ownership_experience_encoded"]
    cl_features = [c for c in cl_features if c in df_no_na_ku.columns]
    X_cl = df_no_na_ku[cl_features].dropna()
    X_scaled = StandardScaler().fit_transform(X_cl)

    if cl_algo == "K-Means":
        st.markdown('<div class="section-header">Elbow Curve — Optimal k</div>', unsafe_allow_html=True)
        inertias = []
        for k_ in range(2, 11):
            km_ = KMeans(n_clusters=k_, random_state=42, n_init=10)
            km_.fit(X_scaled)
            inertias.append(km_.inertia_)
        fig_elbow = px.line(x=list(range(2,11)), y=inertias, markers=True,
                            labels={"x":"Number of Clusters","y":"Inertia"})
        apply_theme(fig_elbow, "Elbow Curve")
        fig_elbow.update_traces(line_color=C["primary"], marker_color=C["primary"], marker_size=8)
        st.plotly_chart(fig_elbow, use_container_width=True)
        insight(
            "Look for the <strong>bend in the line</strong> — that is the optimal k. "
            "Adding more clusters beyond the elbow gives diminishing returns."
        )
        k = st.slider("Number of Clusters (k)", 2, 10, 4, key="km_k")
        model_cl = KMeans(n_clusters=k, random_state=42, n_init=10)

    elif cl_algo == "Agglomerative Hierarchical":
        k = st.slider("Number of Clusters", 2, 10, 4, key="agg_k")
        linkage = st.selectbox("Linkage Method", ["ward","complete","average","single"])
        model_cl = AgglomerativeClustering(n_clusters=k, linkage=linkage)

    else:
        eps   = st.slider("eps", 0.1, 3.0, 0.5, 0.1)
        min_s = st.slider("min_samples", 2, 20, 5)
        model_cl = DBSCAN(eps=eps, min_samples=min_s)

    labels = model_cl.fit_predict(X_scaled)
    df_cl  = X_cl.copy()
    df_cl["Cluster"] = labels.astype(str)
    unique_clusters  = sorted(df_cl["Cluster"].unique())
    n_noise = int((labels == -1).sum())

    st.success(
        f"Found **{len(unique_clusters)}** cluster(s)."
        + (f" ({n_noise} noise points excluded)" if n_noise > 0 else "")
    )

    col1, col2 = st.columns(2)
    with col1:
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X_scaled)
        df_pca = pd.DataFrame({"PC1":coords[:,0],"PC2":coords[:,1],"Cluster":labels.astype(str)})
        fig_pca = px.scatter(df_pca, x="PC1", y="PC2", color="Cluster", opacity=0.75)
        apply_theme(fig_pca,
            f"PCA Cluster Map — PC1 {pca.explained_variance_ratio_[0]*100:.1f}% | "
            f"PC2 {pca.explained_variance_ratio_[1]*100:.1f}%")
        st.plotly_chart(fig_pca, use_container_width=True)
        insight(
            "Well-separated colour blobs = <strong>distinct personas</strong>. "
            "Overlapping blobs suggest those customer types have similar habits."
        )

    with col2:
        if "monthly_spend_inr" in cl_features and "num_dogs" in cl_features:
            size_col = "num_services_used" if "num_services_used" in df_cl.columns else None
            fig_scatter = px.scatter(df_cl, x="monthly_spend_inr", y="num_dogs",
                                     color="Cluster", size=size_col, opacity=0.8)
            apply_theme(fig_scatter, "Spending vs Dogs — by Cluster")
            fig_scatter.update_layout(xaxis_title="Monthly Spend (INR)",
                                      yaxis_title="No. of Dogs")
            st.plotly_chart(fig_scatter, use_container_width=True)
            insight(
                "Clusters in the <strong>top-right corner</strong> are our most valuable segments — "
                "high spend, multiple dogs. Prioritise them for premium plan offers."
            )

    st.markdown('<div class="section-header">Cluster Profiles (Mean Values)</div>',
                unsafe_allow_html=True)
    profile = df_cl.groupby("Cluster")[cl_features].mean().round(2)
    st.dataframe(profile, use_container_width=True)

    fig_prof = px.bar(
        profile.reset_index().melt(id_vars="Cluster"),
        x="variable", y="value", color="Cluster", barmode="group",
        labels={"variable":"Feature","value":"Mean Value"}
    )
    apply_theme(fig_prof, "Feature Comparison Across Clusters")
    st.plotly_chart(fig_prof, use_container_width=True)
    insight(
        "Each cluster's unique profile guides <strong>personalisation strategy</strong>. "
        "High-service-usage clusters should see the booking features front-and-centre on login."
    )

    st.markdown('<div class="section-header">Cluster Interpretation</div>', unsafe_allow_html=True)
    for cl in [c for c in unique_clusters if c != "-1"]:
        row = profile.loc[cl]
        high_feat = row.nlargest(2).index.tolist()
        st.markdown(
            f"**Cluster {cl}:** High **{high_feat[0].replace('_',' ')}** and "
            f"**{high_feat[1].replace('_',' ')}** — "
            "personalise the app experience to highlight features most relevant to this group."
        )
    if "-1" in unique_clusters:
        st.markdown("**Noise (-1):** Outlier respondents — potential niche segment worth exploring.")


# ── TAB 6 : SPENDING FORECAST ────────────────────────────────────────────────
with tabs[6]:
    st.markdown('<div class="section-header">📈 Regression Models — Spending Forecast</div>',
                unsafe_allow_html=True)

    f1, f2 = st.columns(2)
    with f1:
        age_r = st.multiselect("Age Group", sorted(df["age_group"].unique()),
                               default=sorted(df["age_group"].unique()), key="rg_age")
    with f2:
        region_r = st.multiselect("Region", sorted(df["region"].unique()),
                                  default=sorted(df["region"].unique()), key="rg_region")

    df_frg = df[(df["age_group"].isin(age_r)) & (df["region"].isin(region_r))].copy()
    df_no_na_rg = df_frg.dropna(subset=["monthly_spend_inr"]).copy()

    num_cols_reg = df_no_na_rg.select_dtypes(include=np.number).columns.tolist()
    c1, c2, c3 = st.columns(3)
    with c1:
        target_reg = st.selectbox(
            "Target Variable", num_cols_reg,
            index=num_cols_reg.index("monthly_spend_inr") if "monthly_spend_inr" in num_cols_reg else 0,
            key="reg_target"
        )
    with c2:
        test_pct = st.slider("Test Set Size (%)", 15, 35, 20, key="reg_split") / 100
    with c3:
        alpha = st.slider("Regularisation Alpha", 0.01, 20.0, 1.0, 0.01)

    feature_pool = [c for c in num_cols_reg if c != target_reg]
    default_feats = [c for c in ["num_dogs","num_services_used","num_features_valued",
                                  "app_interest_scale","ownership_experience_encoded"]
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
        "Linear Regression":         LinearRegression(),
        f"Ridge (a={alpha})":        Ridge(alpha=alpha),
        f"Lasso (a={alpha})":        Lasso(alpha=alpha, max_iter=10000),
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
            "Model":    mname,
            "RMSE":     round(np.sqrt(mse), 2),
            "R2 Score": round(r2_score(y_te, y_pred_r), 4),
        })

    res_reg = pd.DataFrame(reg_results)

    st.markdown('<div class="section-header">Regression Model Performance</div>', unsafe_allow_html=True)
    st.dataframe(
        res_reg.style
               .highlight_max(subset=["R2 Score"], color="#1e4a4f")
               .highlight_min(subset=["RMSE"],     color="#1e4a4f"),
        use_container_width=True, hide_index=True
    )
    insight(
        "<strong>R2 closer to 1.0 = more accurate forecast.</strong> "
        "Lower RMSE = smaller average prediction error in INR. "
        "Good forecasts directly inform subscription pricing strategy."
    )

    col1, col2 = st.columns(2)
    with col1:
        fig_r2 = px.bar(res_reg, x="Model", y="R2 Score", color="Model",
                        color_discrete_sequence=[C["primary"], C["accent_orange"], C["accent_blue"]])
        apply_theme(fig_r2, "R-Squared Score Comparison")
        fig_r2.update_layout(yaxis=dict(range=[0,1.1]), showlegend=False,
                             xaxis_title="Model", yaxis_title="R2 Score")
        st.plotly_chart(fig_r2, use_container_width=True)
        insight(
            "Ridge and Lasso add a <strong>regularisation penalty</strong> to prevent overfitting. "
            "If they match Linear Regression, the simpler model is preferred for deployment."
        )

    with col2:
        fig_avp = make_subplots(rows=1, cols=3, subplot_titles=list(preds_dict.keys()))
        mc = [C["primary"], C["accent_orange"], C["accent_blue"]]
        for i, (mname, y_pred_r) in enumerate(preds_dict.items(), 1):
            fig_avp.add_trace(
                go.Scatter(x=y_te.values, y=y_pred_r, mode="markers",
                           marker=dict(opacity=0.6, color=mc[i-1]),
                           name=mname, showlegend=False),
                row=1, col=i
            )
            lo, hi = float(y_te.min()), float(y_te.max())
            fig_avp.add_trace(
                go.Scatter(x=[lo,hi], y=[lo,hi], mode="lines",
                           line=dict(color="red", dash="dash"),
                           name="Perfect Fit", showlegend=(i==1)),
                row=1, col=i
            )
            fig_avp.update_xaxes(title_text="Actual", row=1, col=i, gridcolor=C["grid_color"])
            fig_avp.update_yaxes(title_text="Predicted", row=1, col=i, gridcolor=C["grid_color"])
        fig_avp.update_layout(
            title=dict(text="Actual vs Predicted — All Models",
                       font=dict(size=13, color=C["font_color"]), x=0),
            height=350, paper_bgcolor=C["paper_bg"], plot_bgcolor=C["plot_bg"],
            font=dict(family="Inter", color=C["font_color"], size=11),
            margin=dict(l=16, r=16, t=44, b=16)
        )
        st.plotly_chart(fig_avp, use_container_width=True)
        insight(
            "Dots hugging the <strong>red perfect-fit line</strong> = accurate predictions. "
            "Scattered dots indicate the model struggles with that segment's spending behaviour."
        )

    st.markdown('<div class="section-header">Coefficient Comparison — Regularisation Effect</div>',
                unsafe_allow_html=True)
    coef_df = pd.DataFrame(coef_dict, index=selected_feats)
    fig_coef = px.bar(
        coef_df.reset_index().melt(id_vars="index"),
        x="index", y="value", color="variable", barmode="group",
        color_discrete_sequence=[C["primary"], C["accent_orange"], C["accent_blue"]],
        labels={"index":"Feature","value":"Coefficient","variable":"Model"}
    )
    apply_theme(fig_coef, "Feature Coefficients — Linear vs Ridge vs Lasso")
    fig_coef.update_layout(xaxis_tickangle=-30)
    st.plotly_chart(fig_coef, use_container_width=True)

    best_reg = res_reg.sort_values("R2 Score", ascending=False).iloc[0]
    insight(
        f"Best model: <strong>{best_reg['Model']}</strong> "
        f"(R2 = {best_reg['R2 Score']:.4f}, RMSE = {best_reg['RMSE']:.0f} INR). "
        "Lasso's zero coefficients reveal which features have <strong>no real effect</strong> on spending — "
        "helping us focus data collection on what truly matters."
    )


# ── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown(
    f'<div style="text-align:center;font-size:0.73rem;color:{C["text_muted"]};padding:6px 0;">'
    '🐾 PawsIndia Analytics &nbsp;|&nbsp; MBA Group Project &nbsp;|&nbsp; '
    'Data Analytics in Decision Making'
    '</div>',
    unsafe_allow_html=True
)
