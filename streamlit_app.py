 -*- coding: utf-8 -*-
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# â”€â”€ Scikit-learn â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
st.set_page_config(page_title='Dog App Analytics', layout='wide', page_icon='ðŸ¾')

@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, 'dog_app_data.csv')
    df = pd.read_csv(csv_path)
    df['ownership_experience_encoded'] = df['ownership_years'].map(
        {'<1': 1, '1-3': 2, '4-7': 3, '8+': 4}
    )
    # Binary target for classification: 1 = likely to use app, 0 = not
    df['will_adopt_app'] = (df['app_use_likelihood'] == 'Yes').astype(int)
    return df

df = load_data()

st.title('ðŸ¾ India Dog Care App â€” Survey Analytics Dashboard')
st.markdown("**MBA Â· Data Analytics in Decision Making â€” Group Project**")

# â”€â”€ Sidebar Filters â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
st.sidebar.title("ðŸ”Ž Global Filters")
age = st.sidebar.multiselect(
    'Age Group', sorted(df['age_group'].unique()),
    default=sorted(df['age_group'].unique())
)
region = st.sidebar.multiselect(
    'Region', sorted(df['region'].unique()),
    default=sorted(df['region'].unique())
)

df_f = df[(df['age_group'].isin(age)) & (df['region'].isin(region))].copy()
df_no_na = df_f.dropna(subset=['monthly_spend_inr']).copy()

# â”€â”€ KPI Metrics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
c1, c2, c3, c4 = st.columns(4)
c1.metric('Respondents', len(df_f))
c2.metric('Avg Spend (INR)', f"{df_no_na['monthly_spend_inr'].mean():.0f}")
c3.metric('App Interest %', f"{(df_f['app_use_likelihood'] != 'No').mean() * 100:.0f}%")
c4.metric('Avg Dogs', f"{df_f['num_dogs'].mean():.1f}")

# â”€â”€ Tabs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
tabs = st.tabs([
    'ðŸ“Š Overview',
    'ðŸ’° Spending',
    'âš ï¸ Challenges',
    'â­ Features',
    'ðŸ¤– Classification',
    'ðŸ”µ Clustering',
    'ðŸ“ˆ Regression',
])

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# TAB 0 â€” OVERVIEW
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
with tabs[0]:
    col1, col2 = st.columns(2)
    color_map = {'Yes': '#2ecc71', 'Maybe': '#f39c12', 'No': '#e74c3c'}
    fig1 = px.histogram(df_f, x='app_use_likelihood', title='App Adoption Intent',
                        color='app_use_likelihood', color_discrete_map=color_map)
    col1.plotly_chart(fig1, use_container_width=True)
    fig2 = px.histogram(df_no_na, x='monthly_spend_inr', nbins=30,
                        title='Monthly Dog Spending Distribution (INR)')
    col2.plotly_chart(fig2, use_container_width=True)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# TAB 1 â€” SPENDING
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
with tabs[1]:
    col1, col2 = st.columns(2)
    fig3 = px.box(df_no_na, x='age_group', y='monthly_spend_inr',
                  title='Age Group vs Monthly Spending', color='age_group')
    col1.plotly_chart(fig3, use_container_width=True)
    fig4 = px.bar(df_f.groupby('residence_type')['num_dogs'].mean().reset_index(),
                  x='residence_type', y='num_dogs',
                  title='Residence Type vs Avg Number of Dogs', color='residence_type')
    col2.plotly_chart(fig4, use_container_width=True)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# TAB 2 â€” CHALLENGES
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
with tabs[2]:
    challenge_counts = df_f['biggest_challenge'].value_counts().reset_index()
    challenge_counts.columns = ['challenge', 'count']
    fig5 = px.bar(challenge_counts, x='challenge', y='count',
                  title='Biggest Challenges for Dog Owners', color='challenge')
    st.plotly_chart(fig5, use_container_width=True)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# TAB 3 â€” FEATURES
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
with tabs[3]:
    feature_df = pd.DataFrame({
        'Feature': ['Vet Booking', 'Dog Parks', 'Grooming', 'Lost Dog Alert',
                    'Marketplace', 'Community', 'Health Tracking'],
        'Interest': [78, 72, 66, 60, 55, 48, 52]
    }).sort_values('Interest', ascending=False)
    fig6 = px.bar(feature_df, x='Feature', y='Interest',
                  title='Feature Interest (%)', color='Interest',
                  color_continuous_scale='Blues')
    st.plotly_chart(fig6, use_container_width=True)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# TAB 4 â€” CLASSIFICATION (10 marks)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
with tabs[4]:
    st.header("ðŸ¤– Classification Algorithms â€” Performance Comparison")
    st.markdown(
        "**Target:** Predict whether a respondent will adopt the app (`app_use_likelihood = Yes`).  \n"
        "All classifiers are trained and evaluated on the same train/test split."
    )

    # â”€â”€ Feature prep â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    clf_features = [
        'monthly_spend_inr', 'num_dogs', 'num_services_used',
        'num_features_valued', 'app_interest_scale', 'ownership_experience_encoded'
    ]
    # Keep only columns that exist in the dataset
    clf_features = [c for c in clf_features if c in df_no_na.columns]

    df_clf = df_no_na[clf_features + ['will_adopt_app']].dropna()
    X = df_clf[clf_features]
    y = df_clf['will_adopt_app']

    test_size = st.slider("Test set size (%)", 15, 35, 20, key='clf_split') / 100
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
    with st.spinner("Training all 7 classifiers â€¦"):
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

    res_df = (pd.DataFrame(results)
                .sort_values("F1-Score", ascending=False)
                .reset_index(drop=True))

    # â”€â”€ Performance table â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    st.subheader("ðŸ“‹ Performance Comparison Table")
    st.dataframe(
        res_df.style
              .highlight_max(subset=["Accuracy","Precision","Recall","F1-Score"], color="#d4edda")
              .highlight_min(subset=["Accuracy","Precision","Recall","F1-Score"], color="#f8d7da"),
        use_container_width=True
    )

    # â”€â”€ Grouped bar chart â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    st.subheader("ðŸ“Š Metric Comparison Chart")
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    fig_clf = go.Figure()
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA']
    for i, m in enumerate(metrics):
        fig_clf.add_trace(go.Bar(
            name=m, x=res_df["Algorithm"], y=res_df[m],
            marker_color=colors[i]
        ))
    fig_clf.update_layout(
        barmode='group', title="All Classifiers â€” Accuracy / Precision / Recall / F1",
        yaxis=dict(range=[0, 1.15]), xaxis_tickangle=-30,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig_clf, use_container_width=True)

    # â”€â”€ Confusion matrix for best model â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    best_name = res_df.iloc[0]["Algorithm"]
    _, y_pred_best = trained_models[best_name]
    st.subheader(f"ðŸ” Confusion Matrix â€” Best Model: **{best_name}**")
    cm = confusion_matrix(y_test, y_pred_best)
    fig_cm = ff.create_annotated_heatmap(
        z=cm.tolist(),
        x=["Predicted: No", "Predicted: Yes"],
        y=["Actual: No", "Actual: Yes"],
        colorscale='Blues', showscale=False
    )
    fig_cm.update_layout(title=f"Confusion Matrix â€” {best_name}")
    st.plotly_chart(fig_cm, use_container_width=True)

    # â”€â”€ Detailed report â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    st.subheader("ðŸ“„ Classification Report")
    st.code(classification_report(y_test, y_pred_best,
                                  target_names=["Won't Adopt", "Will Adopt"]))

    # â”€â”€ Feature importance â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    best_clf, _ = trained_models[best_name]
    if hasattr(best_clf, "feature_importances_"):
        st.subheader("ðŸŒŸ Feature Importance")
        fi = pd.DataFrame({
            "Feature":   clf_features,
            "Importance": best_clf.feature_importances_
        }).sort_values("Importance", ascending=True)
        fig_fi = px.bar(fi, x="Importance", y="Feature", orientation="h",
                        title=f"Feature Importance â€” {best_name}",
                        color="Importance", color_continuous_scale="Oranges")
        st.plotly_chart(fig_fi, use_container_width=True)

    # â”€â”€ Insight box â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    st.info(
        f"**Key Insight:** **{best_name}** achieved the highest F1-Score of "
        f"**{res_df.iloc[0]['F1-Score']:.4f}**, making it the recommended model for "
        f"predicting app adoption. High recall ensures we identify the maximum number "
        f"of potential app users, while precision minimises wasted marketing effort on "
        f"unlikely adopters."
    )

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# TAB 5 â€” CLUSTERING (10 marks)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
with tabs[5]:
    st.header("ðŸ”µ Clustering Analysis â€” Customer Personas")
    st.markdown(
        "Segment dog owners into distinct personas using unsupervised clustering. "
        "Select an algorithm and tune parameters below."
    )

    cl_algo = st.selectbox(
        "Clustering Algorithm",
        ["K-Means", "Agglomerative Hierarchical", "DBSCAN"],
        key="cl_algo"
    )

    cl_features = ['monthly_spend_inr', 'num_dogs', 'num_services_used',
                   'ownership_experience_encoded']
    cl_features = [c for c in cl_features if c in df_no_na.columns]

    X_cl = df_no_na[cl_features].dropna()
    X_scaled = StandardScaler().fit_transform(X_cl)

    # â”€â”€ Algorithm-specific controls â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if cl_algo == "K-Means":
        # Elbow curve
        st.subheader("ðŸ“ Elbow Method â€” Choose Optimal k")
        inertias = []
        k_range = range(2, 11)
        for k_ in k_range:
            km_ = KMeans(n_clusters=k_, random_state=42, n_init=10)
            km_.fit(X_scaled)
            inertias.append(km_.inertia_)
        fig_elbow = px.line(x=list(k_range), y=inertias, markers=True,
                            labels={"x": "Number of Clusters (k)", "y": "Inertia"},
                            title="Elbow Curve")
        st.plotly_chart(fig_elbow, use_container_width=True)

        k = st.slider("Number of clusters (k)", 2, 10, 4, key="km_k")
        model_cl = KMeans(n_clusters=k, random_state=42, n_init=10)

    elif cl_algo == "Agglomerative Hierarchical":
        k = st.slider("Number of clusters", 2, 10, 4, key="agg_k")
        linkage = st.selectbox("Linkage", ["ward", "complete", "average", "single"])
        model_cl = AgglomerativeClustering(n_clusters=k, linkage=linkage)

    else:  # DBSCAN
        eps    = st.slider("eps (neighbourhood radius)", 0.1, 3.0, 0.5, 0.1)
        min_s  = st.slider("min_samples", 2, 20, 5)
        model_cl = DBSCAN(eps=eps, min_samples=min_s)

    labels = model_cl.fit_predict(X_scaled)
    df_cl = X_cl.copy()
    df_cl["Cluster"] = labels.astype(str)

    unique_clusters = sorted(df_cl["Cluster"].unique())
    n_noise = int((labels == -1).sum())
    st.success(
        f"Found **{len(unique_clusters)}** cluster(s): {unique_clusters}"
        + (f" â€” including **{n_noise} noise points** (label -1)" if n_noise > 0 else "")
    )

    # â”€â”€ PCA scatter â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    st.subheader("ðŸ—ºï¸ Cluster Visualisation (PCA 2D projection)")
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    df_pca = pd.DataFrame({
        "PC1": coords[:, 0], "PC2": coords[:, 1],
        "Cluster": labels.astype(str)
    })
    fig_pca = px.scatter(
        df_pca, x="PC1", y="PC2", color="Cluster",
        title=(f"Clusters (PCA) â€” PC1 {pca.explained_variance_ratio_[0]*100:.1f}% var, "
               f"PC2 {pca.explained_variance_ratio_[1]*100:.1f}% var"),
        opacity=0.75
    )
    st.plotly_chart(fig_pca, use_container_width=True)

    # â”€â”€ Spend vs Dogs scatter (original space) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if 'monthly_spend_inr' in cl_features and 'num_dogs' in cl_features:
        st.subheader("ðŸ’¡ Spending vs Number of Dogs â€” by Cluster")
        df_cl_plot = df_cl.copy()
        if 'num_services_used' in df_cl_plot.columns:
            fig_scatter = px.scatter(
                df_cl_plot, x='monthly_spend_inr', y='num_dogs',
                color='Cluster', size='num_services_used',
                title="Dog Owner Segments", opacity=0.8
            )
        else:
            fig_scatter = px.scatter(
                df_cl_plot, x='monthly_spend_inr', y='num_dogs',
                color='Cluster', title="Dog Owner Segments", opacity=0.8
            )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # â”€â”€ Cluster profiles â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    st.subheader("ðŸ“‹ Cluster Profiles (Mean Values)")
    profile = df_cl.groupby("Cluster")[cl_features].mean().round(2)
    st.dataframe(profile, use_container_width=True)

    # Radar-style grouped bar
    fig_prof = px.bar(
        profile.reset_index().melt(id_vars="Cluster"),
        x="variable", y="value", color="Cluster", barmode="group",
        title="Mean Feature Values per Cluster",
        labels={"variable": "Feature", "value": "Mean Value"}
    )
    st.plotly_chart(fig_prof, use_container_width=True)

    # â”€â”€ Persona labels (K-Means default) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if cl_algo == "K-Means" and k == 4:
        persona_labels = {
            '0': 'ðŸ¶ Casual Owner',
            '1': 'â¤ï¸ Engaged Pet Parent',
            '2': 'ðŸ  Multi-Dog Household',
            '3': 'ðŸ’Ž Premium Spender'
        }
        df_cl['Persona'] = df_cl['Cluster'].map(persona_labels)
        st.subheader("ðŸŽ­ Persona Distribution")
        persona_counts = df_cl['Persona'].value_counts().reset_index()
        persona_counts.columns = ['Persona', 'Count']
        fig_persona = px.pie(persona_counts, names='Persona', values='Count',
                             title="Customer Persona Breakdown", hole=0.4)
        st.plotly_chart(fig_persona, use_container_width=True)

    # â”€â”€ Interpretation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    st.subheader("ðŸ“ Cluster Interpretation")
    for cl in [c for c in unique_clusters if c != '-1']:
        row = profile.loc[cl]
        high_feat = row.nlargest(2).index.tolist()
        low_feat  = row.nsmallest(2).index.tolist()
        st.markdown(
            f"**Cluster {cl}:** Dogs owners in this segment show relatively high "
            f"**{high_feat[0].replace('_',' ')}** and **{high_feat[1].replace('_',' ')}**, "
            f"with lower **{low_feat[0].replace('_',' ')}**. "
            f"This group represents a distinct behavioural profile that can be targeted "
            f"with tailored in-app features and pricing strategies."
        )
    if '-1' in unique_clusters:
        st.markdown(
            "**Cluster -1 (Noise/Outliers):** These respondents have unusual combinations "
            "of features and do not fit any main segment â€” they may represent niche users "
            "worth investigating separately."
        )

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# TAB 6 â€” REGRESSION (10 marks)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
with tabs[6]:
    st.header("ðŸ“ˆ Regression Analysis â€” Linear, Ridge & Lasso")
    st.markdown(
        "Predict **monthly spending** (or another numeric target) using regularised "
        "regression. Compare all three models side-by-side."
    )

    num_cols_reg = df_no_na.select_dtypes(include=np.number).columns.tolist()
    target_reg = st.selectbox(
        "Target variable to predict", num_cols_reg,
        index=num_cols_reg.index('monthly_spend_inr') if 'monthly_spend_inr' in num_cols_reg else 0,
        key="reg_target"
    )
    feature_pool = [c for c in num_cols_reg if c != target_reg]
    default_feats = [c for c in [
        'num_dogs', 'num_services_used', 'num_features_valued',
        'app_interest_scale', 'ownership_experience_encoded'
    ] if c in feature_pool]
    selected_feats = st.multiselect(
        "Predictor features", feature_pool,
        default=default_feats or feature_pool[:5],
        key="reg_feats"
    )
    if not selected_feats:
        st.warning("Please select at least one predictor feature.")
        st.stop()

    df_reg = df_no_na[selected_feats + [target_reg]].dropna()
    X_r = df_reg[selected_feats]
    y_r = df_reg[target_reg]

    test_pct = st.slider("Test set size (%)", 15, 35, 20, key="reg_split") / 100
    alpha    = st.slider("Regularisation alpha (Ridge / Lasso)", 0.01, 20.0, 1.0, 0.01)

    X_tr, X_te, y_tr, y_te = train_test_split(X_r, y_r, test_size=test_pct, random_state=42)
    sc_r = StandardScaler()
    X_tr_s = sc_r.fit_transform(X_tr)
    X_te_s  = sc_r.transform(X_te)

    reg_models = {
        "Linear Regression": LinearRegression(),
        f"Ridge (Î±={alpha})": Ridge(alpha=alpha),
        f"Lasso (Î±={alpha})": Lasso(alpha=alpha, max_iter=10000),
    }

    reg_results, preds_dict, coef_dict = [], {}, {}
    for mname, mreg in reg_models.items():
        mreg.fit(X_tr_s, y_tr)
        y_pred_r = mreg.predict(X_te_s)
        preds_dict[mname] = y_pred_r
        if hasattr(mreg, "coef_"):
            coef_dict[mname] = mreg.coef_
        mse  = mean_squared_error(y_te, y_pred_r)
        reg_results.append({
            "Model":    mname,
            "MSE":      round(mse, 2),
            "RMSE":     round(np.sqrt(mse), 2),
            "RÂ² Score": round(r2_score(y_te, y_pred_r), 4),
        })

    res_reg = pd.DataFrame(reg_results)

    # â”€â”€ Performance table â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    st.subheader("ðŸ“‹ Regression Performance Comparison")
    st.dataframe(
        res_reg.style
               .highlight_max(subset=["RÂ² Score"], color="#d4edda")
               .highlight_min(subset=["RMSE", "MSE"], color="#d4edda"),
        use_container_width=True
    )

    # â”€â”€ Metric bar chart â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    st.subheader("ðŸ“Š RÂ² Score Comparison")
    fig_r2 = px.bar(res_reg, x="Model", y="RÂ² Score", color="Model",
                    title="RÂ² Score by Regression Model",
                    color_discrete_sequence=px.colors.qualitative.Set2)
    fig_r2.update_layout(yaxis=dict(range=[0, 1.1]))
    st.plotly_chart(fig_r2, use_container_width=True)

    # â”€â”€ Actual vs Predicted â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    st.subheader("ðŸŽ¯ Actual vs Predicted")
    fig_avp = make_subplots(rows=1, cols=3, subplot_titles=list(preds_dict.keys()))
    for i, (mname, y_pred_r) in enumerate(preds_dict.items(), 1):
        fig_avp.add_trace(
            go.Scatter(x=y_te.values, y=y_pred_r, mode='markers',
                       marker=dict(opacity=0.6), name=mname,
                       showlegend=False),
            row=1, col=i
        )
        # Perfect-fit line
        lo, hi = float(y_te.min()), float(y_te.max())
        fig_avp.add_trace(
            go.Scatter(x=[lo, hi], y=[lo, hi], mode='lines',
                       line=dict(color='red', dash='dash'),
                       name='Perfect fit', showlegend=(i == 1)),
            row=1, col=i
        )
        fig_avp.update_xaxes(title_text="Actual", row=1, col=i)
        fig_avp.update_yaxes(title_text="Predicted", row=1, col=i)
    fig_avp.update_layout(title="Actual vs Predicted â€” All Models", height=400)
    st.plotly_chart(fig_avp, use_container_width=True)

    # â”€â”€ Coefficient comparison â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    st.subheader("ðŸ”¢ Coefficient Comparison (Effect of Regularisation)")
    coef_df = pd.DataFrame(coef_dict, index=selected_feats)
    fig_coef = px.bar(
        coef_df.reset_index().melt(id_vars="index"),
        x="index", y="value", color="variable", barmode="group",
        title="Feature Coefficients â€” Linear vs Ridge vs Lasso",
        labels={"index": "Feature", "value": "Coefficient", "variable": "Model"}
    )
    fig_coef.update_layout(xaxis_tickangle=-30)
    st.plotly_chart(fig_coef, use_container_width=True)

    # â”€â”€ Interpretation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    best_reg = res_reg.sort_values("RÂ² Score", ascending=False).iloc[0]
    st.info(
        f"**Key Insight:** **{best_reg['Model']}** achieves the highest RÂ² of "
        f"**{best_reg['RÂ² Score']:.4f}**, explaining {best_reg['RÂ² Score']*100:.1f}% of "
        f"variance in {target_reg.replace('_',' ')}. "
        f"**Lasso** drives less important feature coefficients to zero (automatic feature selection), "
        f"while **Ridge** shrinks all coefficients â€” both reduce overfitting vs plain Linear Regression. "
        f"A higher alpha increases regularisation strength."
    )

st.caption('MBA Project Â· India Dog Care App Market Analysis Â· Data Analytics in Decision Making')
