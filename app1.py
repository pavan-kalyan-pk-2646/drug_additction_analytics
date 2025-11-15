# app.py
"""
Youth Substance Risk Explorer — Professional Edition

Features added/updated:
- Polished international-grade UI (tabbed dashboard, dark mode toggle)
- Rich visualizations with selectable chart types (boxplot, violin, histogram, bar, pie, scatter)
- Per-model visualization options (ROC, Precision-Recall, Confusion Matrix, Prob boxplots)
- Individual prediction with clear SHAP-like fallback explanations (no external shap dependency)
- Save / Load trained model (.pkl) using joblib
- Export cleaned dataset and predictions
- Q&A natural language module (rules + suggestions)
- Robust error handling and graceful fallbacks

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import base64
import io
import joblib
import os
import textwrap
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, precision_recall_curve)
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Youth Substance Risk Explorer — Pro", layout="wide")

# -------------------------
# Global CSS and header
# -------------------------
HEADER = """
<style>
body { background-color: #f4f6fb; }
.header { padding: 16px; border-radius:12px; background: linear-gradient(90deg,#0f172a,#3b82f6); color: white; margin-bottom: 12px; }
.h1 { font-size: 26px; font-weight:800; }
.h2 { font-size:14px; color:#e6eefc; }
.card { background: white; padding:12px; border-radius:10px; box-shadow: 0 6px 18px rgba(2,6,23,0.06); }
.button-link{ background:#4F46E5; color:white; padding:8px 12px; border-radius:8px; text-decoration:none; }
</style>
<div class='header'><div class='h1'>Youth Substance Risk Explorer — Pro</div><div class='h2'>Advanced interactive analytics, model explainability, and exports</div></div>
"""
st.markdown(HEADER, unsafe_allow_html=True)

# -------------------------
# Helper functions
# -------------------------

def make_substance_risk(df, col='Drug_Experimentation'):
    df = df.copy()
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        med = df[col].median(skipna=True)
        if pd.isna(med) or df[col].nunique(dropna=True) <= 1:
            df['Substance_Risk'] = 0
        else:
            df['Substance_Risk'] = (df[col] > med).astype(int)
    else:
        if 'Substance_Risk' not in df.columns:
            df['Substance_Risk'] = 0
    return df


def safe_label_encode(series):
    le = LabelEncoder()
    arr = le.fit_transform(series.fillna('MISSING').astype(str))
    return arr, le


def compute_metrics(y_true, y_pred, y_score=None):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = None
    if y_score is not None:
        try:
            auc = roc_auc_score(y_true, y_score)
        except Exception:
            auc = None
    return {'accuracy':acc, 'precision':prec, 'recall':rec, 'f1':f1, 'auc':auc}


def to_download_link(df, name='data.csv'):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f"data:text/csv;base64,{b64}"


def plot_confusion_matrix_fig(cm, labels=None):
    fig, ax = plt.subplots(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                xticklabels=labels or ['Pred 0','Pred 1'], yticklabels=labels or ['Actual 0','Actual 1'])
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    plt.tight_layout()
    return fig


def fallback_explain(model, sample_df, X_train, label_encoders, model_name='Model', top_n=5):
    # Returns list of tuples (feature, contribution, readable_value)
    contributions = []
    try:
        if model_name == 'Logistic Regression' and hasattr(model, 'coef_'):
            coefs = pd.Series(model.coef_[0], index=X_train.columns)
            contrib = coefs * sample_df.iloc[0]
            contrib_sorted = contrib.abs().sort_values(ascending=False)
            for feat in contrib_sorted.index[:top_n]:
                raw = sample_df.iloc[0][feat]
                readable = raw
                if feat.endswith('_enc'):
                    orig = feat.replace('_enc','')
                    if orig in label_encoders:
                        try:
                            readable = label_encoders[orig].inverse_transform([int(raw)])[0]
                        except Exception:
                            readable = raw
                contributions.append((feat, float(contrib[feat]), readable))
        elif hasattr(model, 'feature_importances_'):
            imps = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
            medians = X_train.median()
            for feat in imps.index[:top_n]:
                raw = sample_df.iloc[0][feat]
                readable = raw
                if feat.endswith('_enc'):
                    orig = feat.replace('_enc','')
                    if orig in label_encoders:
                        try:
                            readable = label_encoders[orig].inverse_transform([int(raw)])[0]
                        except Exception:
                            readable = raw
                contributions.append((feat, float(imps[feat]), readable))
    except Exception as e:
        contributions = [('explain_error', str(e), '')]
    return contributions

# -------------------------
# Load/prepare data
# -------------------------
@st.cache_data
def load_data(uploaded_file, use_sample, path_input):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    if use_sample:
        np.random.seed(1)
        n=2000
        df = pd.DataFrame({
            'Age_Group': np.random.choice(['10-13','14-16','17-19'], n),
            'Gender': np.random.choice(['Male','Female','Other'], n),
            'Socioeconomic_Status': np.random.choice(['Low','Medium','High'], n),
            'School_Programs': np.random.choice(['Yes','No'], n),
            'Smoking_Prevalence': np.random.normal(0.2,0.1,n).clip(0,1),
            'Mental_Health': np.random.choice(['Good','Average','Poor'], n),
            'Family_Background': np.random.choice(['Stable','Unstable'], n),
            'Access_to_Counseling': np.random.choice(['Yes','No'], n),
            'Substance_Education': np.random.choice(['Yes','No'], n),
            'Community_Support': np.random.choice(['Strong','Weak'], n),
            'Drug_Experimentation': np.floor(np.clip(np.random.normal(2,2,n) + (np.random.choice([0,1], n, p=[0.7,0.3]) * 5), 0, 10))
        })
        return df
    if path_input:
        try:
            return pd.read_csv(path_input)
        except Exception:
            return None
    return None

# -------------------------
# App layout & state
# -------------------------

def main():
    st.sidebar.title('Controls')
    uploaded_file = st.sidebar.file_uploader('Upload CSV (optional)', type=['csv'])
    use_sample = st.sidebar.checkbox('Use sample dataset')
    path_input = st.sidebar.text_input('CSV path (optional)', value='')

    model_choice = st.sidebar.selectbox('Model', ['Logistic Regression','Decision Tree','Random Forest'])
    test_size = st.sidebar.slider('Test fraction', 0.05, 0.5, 0.2, 0.05)
    seed = st.sidebar.number_input('Random seed', 0, 9999, 42)
    prob_threshold = st.sidebar.slider('Prob threshold (P >= )', 0.01, 0.99, 0.5, 0.01)

    ui_dark = st.sidebar.checkbox('Dark mode (toggle)')
    if ui_dark:
        st.markdown('<style>body{background:#071226;color:#dbeafe;}</style>', unsafe_allow_html=True)

    show_roc = st.sidebar.checkbox('Show ROC Curve', value=True)
    show_predict_table = st.sidebar.checkbox('Show predictions table', value=False)

    # visualization options: available chart types
    chart_types = ['Bar','Pie','Histogram','Boxplot','Violin','Scatter']

    df = load_data(uploaded_file, use_sample, path_input if path_input else None)
    if df is None:
        st.error('No data loaded. Upload file, enter path or use sample.')
        st.stop()

    st.title('Overview & Quick Actions')
    col_a, col_b = st.columns([3,1])
    with col_a:
        st.subheader('Dataset Preview')
        st.dataframe(df.head(10), use_container_width=True)
    with col_b:
        st.subheader('Quick Actions')
        if st.button('Download raw CSV'):
            tmp = to_download_link(df, 'raw_dataset.csv')
            st.markdown(f"[Download raw CSV]({tmp})")
        st.markdown('---')
        if st.button('Reset session state'):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.experimental_rerun()

    # Preprocess
    df_proc = make_substance_risk(df.copy())
    cat_cols = ['Age_Group','Gender','Socioeconomic_Status','School_Programs','Mental_Health','Family_Background','Access_to_Counseling','Substance_Education','Community_Support']
    label_encoders = {}
    for c in cat_cols:
        if c in df_proc.columns:
            arr, le = safe_label_encode(df_proc[c])
            df_proc[f'{c}_enc'] = arr
            label_encoders[c] = le

    if 'Smoking_Prevalence' in df_proc.columns:
        df_proc['Smoking_Prevalence'] = pd.to_numeric(df_proc['Smoking_Prevalence'], errors='coerce').fillna(df_proc['Smoking_Prevalence'].median())
    else:
        df_proc['Smoking_Prevalence'] = 0.0

    features = []
    for base in ['Age_Group','Gender','Socioeconomic_Status','Smoking_Prevalence','Mental_Health','Family_Background','Access_to_Counseling','Substance_Education','Community_Support']:
        if base == 'Smoking_Prevalence':
            features.append('Smoking_Prevalence')
        else:
            enc = base + '_enc'
            if enc in df_proc.columns:
                features.append(enc)

    if len(features) == 0:
        st.error('No features available. Ensure dataset has expected columns.')
        st.stop()

    df_model = df_proc.dropna(subset=['Substance_Risk']).copy()
    X = df_model[features]
    y = df_model['Substance_Risk'].astype(int)

    stratify_param = y if len(np.unique(y))>1 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=stratify_param)

    # Train model
    model = None
    y_score = None
    y_pred = None
    try:
        if model_choice == 'Logistic Regression':
            model = LogisticRegression(max_iter=2000, solver='liblinear', random_state=seed)
            model.fit(X_train, y_train)
            if hasattr(model, 'predict_proba'):
                y_score = model.predict_proba(X_test)[:,1]
            y_pred = (model.predict_proba(X_test)[:,1] >= prob_threshold).astype(int) if hasattr(model, 'predict_proba') else model.predict(X_test)
        elif model_choice == 'Decision Tree':
            model = DecisionTreeClassifier(random_state=seed)
            model.fit(X_train, y_train)
            if hasattr(model, 'predict_proba'):
                y_score = model.predict_proba(X_test)[:,1]
            y_pred = model.predict(X_test)
        else:
            model = RandomForestClassifier(random_state=seed, n_estimators=250, n_jobs=-1)
            model.fit(X_train, y_train)
            if hasattr(model, 'predict_proba'):
                y_score = model.predict_proba(X_test)[:,1]
            y_pred = model.predict(X_test)
    except Exception as e:
        st.error(f'Model training failed: {e}')
        st.stop()

    metrics = compute_metrics(y_test, y_pred, y_score=y_score)

    # Tabs for UI
    tab_data, tab_visual, tab_models, tab_individual, tab_insights, tab_qa = st.tabs(['Data','Visualize','Models','Individual','Insights','Q&A'])

    # -------------------------
    # Tab: Data
    # -------------------------
    with tab_data:
        st.header('Data & Exports')
        st.write('Cleaned dataset used for modeling (Substance_Risk created by median split).')
        st.dataframe(df_model.head(200), use_container_width=True)
        col1, col2 = st.columns([2,1])
        with col1:
            st.markdown('**Download cleaned dataset**')
            tmp = to_download_link(df_model, 'cleaned_dataset.csv')
            st.markdown(f"[Download cleaned dataset]({tmp})")
        with col2:
            st.markdown('**Save/Load Model**')
            model_name = st.text_input('Model filename', value=f'model_{model_choice.replace(" ","_")}.pkl')
            if st.button('Save model'):
                try:
                    joblib.dump({'model':model, 'features':features, 'label_encoders':label_encoders, 'model_choice':model_choice}, model_name)
                    st.success(f'Model saved to {model_name}')
                except Exception as e:
                    st.error(f'Save failed: {e}')
            load_file = st.file_uploader('Upload saved model (.pkl)', type=['pkl'])
            if load_file is not None:
                try:
                    loaded = joblib.load(load_file)
                    if 'model' in loaded:
                        model = loaded['model']
                        features = loaded.get('features', features)
                        label_encoders = loaded.get('label_encoders', label_encoders)
                        st.success('Model loaded and will be used for predictions.')
                    else:
                        st.error('Uploaded file did not contain a valid model.')
                except Exception as e:
                    st.error(f'Load failed: {e}')

    # -------------------------
    # Tab: Visualize (interactive chart picker)
    # -------------------------
    with tab_visual:
        st.header('Interactive Visualizations')
        st.markdown('Pick variables and chart type. For categorical columns, bar/pie are recommended. For numeric: histogram/box/violin/scatter.')
        all_cols = df_model.columns.tolist()
        colx = st.selectbox('X variable', all_cols, index=all_cols.index('Age_Group') if 'Age_Group' in all_cols else 0)
        coly = st.selectbox('Y variable (optional)', [''] + all_cols, index=0)
        chart = st.selectbox('Chart Type', chart_types)
        apply = st.button('Render Chart')
        if apply:
            try:
                if chart == 'Bar':
                    if df_model[colx].dtype == 'O' or df_model[colx].nunique() < 20:
                        v = df_model.groupby(colx)['Substance_Risk'].mean().reset_index()
                        v['pct'] = v['Substance_Risk']
                        fig = px.bar(v, x=colx, y='pct', text=v['pct'].apply(lambda x: f"{x:.1%}"), title=f'Avg High Risk by {colx}')
                        fig.update_layout(yaxis_tickformat='.0%')
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info('Bar chart best for categorical X. Consider histogram for numeric.')
                elif chart == 'Pie':
                    if df_model[colx].dtype == 'O' or df_model[colx].nunique() < 30:
                        fig = px.pie(df_model, names=colx, title=f'Distribution of {colx}')
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info('Too many unique values for pie chart.')
                elif chart == 'Histogram':
                    if coly and coly != '':
                        st.info('Histogram uses a single numeric variable (X). Ignoring Y.')
                    if pd.api.types.is_numeric_dtype(df_model[colx]):
                        fig = px.histogram(df_model, x=colx, nbins=30, marginal='box', title=f'Histogram of {colx}')
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info('Choose a numeric variable for histogram (e.g., Smoking_Prevalence).')
                elif chart == 'Boxplot':
                    if coly and coly != '':
                        if pd.api.types.is_numeric_dtype(df_model[coly]):
                            fig = px.box(df_model, x=colx, y=coly, points='outliers', title=f'Boxplot of {coly} by {colx}')
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info('Y must be numeric for boxplot.')
                    else:
                        if pd.api.types.is_numeric_dtype(df_model[colx]):
                            fig = px.box(df_model, y=colx, points='outliers', title=f'Boxplot of {colx}')
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info('Choose numeric variable for boxplot.')
                elif chart == 'Violin':
                    if coly and coly != '':
                        if pd.api.types.is_numeric_dtype(df_model[coly]):
                            fig = px.violin(df_model, x=colx, y=coly, box=True, title=f'Violin of {coly} by {colx}')
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info('Y must be numeric for violin plot.')
                    else:
                        st.info('Select Y numeric variable for violin plot.')
                elif chart == 'Scatter':
                    if coly and coly != '' and pd.api.types.is_numeric_dtype(df_model[colx]) and pd.api.types.is_numeric_dtype(df_model[coly]):
                        fig = px.scatter(df_model, x=colx, y=coly, color='Substance_Risk', title=f'{colx} vs {coly} (colored by risk)')
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info('Choose two numeric variables for scatter.')
            except Exception as e:
                st.error(f'Could not render chart: {e}')

    # -------------------------
    # Tab: Models (per-model visuals and metrics)
    # -------------------------
    with tab_models:
        st.header('Model Dashboard')
        st.markdown('Evaluate the trained model: confusion matrix, ROC, PR curve, feature importance, and probability distributions.')
        st.subheader('Key Metrics')
        c1,c2,c3,c4 = st.columns(4)
        c1.metric('Accuracy', f"{metrics['accuracy']:.3f}")
        c2.metric('Precision', f"{metrics['precision']:.3f}")
        c3.metric('Recall', f"{metrics['recall']:.3f}")
        c4.metric('F1', f"{metrics['f1']:.3f}")
        if metrics.get('auc') is not None:
            st.markdown(f"**ROC AUC:** {metrics['auc']:.3f}")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        st.subheader('Confusion Matrix')
        st.pyplot(plot_confusion_matrix_fig(cm))

        # ROC and PR
        if y_score is not None and show_roc:
            st.subheader('ROC & Precision-Recall')
            fpr, tpr, _ = roc_curve(y_test, y_score)
            prec, rec, _ = precision_recall_curve(y_test, y_score)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC'))
            fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random', line=dict(dash='dash')))
            fig.update_layout(xaxis_title='FPR', yaxis_title='TPR', title='ROC Curve')
            st.plotly_chart(fig, use_container_width=True)

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=rec, y=prec, mode='lines', name='PR'))
            fig2.update_layout(xaxis_title='Recall', yaxis_title='Precision', title='Precision-Recall Curve')
            st.plotly_chart(fig2, use_container_width=True)

        # Probabilities boxplot by actual
        if y_score is not None:
            st.subheader('Predicted Probability Distribution by Actual Class')
            prob_df = pd.DataFrame({'prob': y_score, 'actual': y_test.reset_index(drop=True)})
            figp = px.box(prob_df, x='actual', y='prob', points='all', title='Predicted prob by actual label')
            st.plotly_chart(figp, use_container_width=True)

        # Feature importance
        st.subheader('Feature Importance / Coefficients')
        try:
            if hasattr(model, 'feature_importances_'):
                imps = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
                fig_imp = px.bar(x=imps.values, y=imps.index, orientation='h', title='Feature Importances')
                st.plotly_chart(fig_imp, use_container_width=True)
                st.dataframe(imps.sort_values(ascending=False).rename('Importance').to_frame())
            elif hasattr(model, 'coef_'):
                coefs = pd.Series(model.coef_[0], index=features).sort_values(key=abs, ascending=True)
                figc = px.bar(x=coefs.values, y=coefs.index, orientation='h', title='Coefficients (signed)')
                st.plotly_chart(figc, use_container_width=True)
                st.dataframe(pd.Series(model.coef_[0], index=features).sort_values(ascending=False).rename('Coefficient').to_frame())
            else:
                st.info('Model does not expose feature importances.')
        except Exception as e:
            st.error(f'Feature importance error: {e}')

    # -------------------------
    # Tab: Individual prediction
    # -------------------------
    with tab_individual:
        st.header('Individual Prediction & Explainability')
        st.markdown('Enter an individual profile (Age Group) and get prediction + human-friendly explanation.')
        # build choices
        def safe_vals(col, default):
            return sorted(df[col].dropna().unique().tolist()) if col in df.columns else default
        age_choices = safe_vals('Age_Group', ['10-13','14-16','17-19'])
        gender_choices = safe_vals('Gender', ['Male','Female','Other'])
        ses_choices = safe_vals('Socioeconomic_Status', ['Low','Medium','High'])
        mh_choices = safe_vals('Mental_Health', ['Good','Average','Poor'])
        fb_choices = safe_vals('Family_Background', ['Stable','Unstable'])
        school_choices = safe_vals('School_Programs', ['Yes','No'])
        counsel_choices = safe_vals('Access_to_Counseling', ['Yes','No'])
        edu_choices = safe_vals('Substance_Education', ['Yes','No'])
        comm_choices = safe_vals('Community_Support', ['Strong','Weak'])

        c1,c2,c3 = st.columns(3)
        with c1:
            inp_age = st.selectbox('Age Group', age_choices)
            inp_gender = st.selectbox('Gender', gender_choices)
            inp_ses = st.selectbox('Socioeconomic Status', ses_choices)
            inp_school = st.selectbox('School Programs', school_choices)
        with c2:
            inp_smoke = st.number_input('Smoking Prevalence (0-1)', min_value=0.0, max_value=1.0, value=0.2, step=0.01, format='%.2f')
            inp_mh = st.selectbox('Mental Health', mh_choices)
            inp_family = st.selectbox('Family Background', fb_choices)
            inp_counsel = st.selectbox('Access to Counseling', counsel_choices)
        with c3:
            inp_edu = st.selectbox('Substance Education', edu_choices)
            inp_comm = st.selectbox('Community Support', comm_choices)
            run_ind = st.button('Run Analysis')

        if run_ind:
            # build sample aligned to features
            sample = {}
            for feat in features:
                if feat == 'Smoking_Prevalence':
                    sample[feat] = inp_smoke
                else:
                    orig = feat.replace('_enc','')
                    val = None
                    if orig == 'Age_Group': val = inp_age
                    elif orig == 'Gender': val = inp_gender
                    elif orig == 'Socioeconomic_Status': val = inp_ses
                    elif orig == 'Mental_Health': val = inp_mh
                    elif orig == 'Family_Background': val = inp_family
                    elif orig == 'Access_to_Counseling': val = inp_counsel
                    elif orig == 'Substance_Education': val = inp_edu
                    elif orig == 'Community_Support': val = inp_comm
                    if orig in label_encoders and val is not None:
                        try:
                            sample[feat] = int(label_encoders[orig].transform([val])[0])
                        except Exception:
                            sample[feat] = 0
                    else:
                        sample[feat] = 0
            sample_df = pd.DataFrame([sample])[features]
            try:
                prob = float(model.predict_proba(sample_df)[:,1][0]) if hasattr(model,'predict_proba') else float(model.predict(sample_df)[0])
                pred_label = int(prob >= prob_threshold)
                st.metric('Predicted Risk', 'HIGH (1)' if pred_label==1 else 'LOW (0)')
                st.write(f'Probability of High Risk: {prob:.3f} (threshold {prob_threshold})')

                # show readable input summary
                disp = {'Age Group':inp_age,'Gender':inp_gender,'SES':inp_ses,'Smoking':inp_smoke,'Mental Health':inp_mh,'Family':inp_family,'Counseling':inp_counsel,'Education':inp_edu,'Community':inp_comm}
                st.table(pd.DataFrame([disp]).T.rename(columns={0:'Value'}))

                # explanation
                contribs = fallback_explain(model, sample_df, X_train, label_encoders, model_choice)
                st.subheader('Top contributing features (heuristic)')
                if contribs and contribs[0][0] != 'explain_error':
                    cont_df = pd.DataFrame([{'Feature':f, 'Contribution':c, 'Value':v} for f,c,v in contribs])
                    st.dataframe(cont_df)
                    expl_lines = []
                    for f,c,v in cont_df.itertuples(index=False):
                        dirn = 'increases' if c>0 else 'decreases'
                        expl_lines.append(f"{f}={v} ({dirn} risk, score {c:.3f})")
                    st.write('Summary:', ' | '.join(expl_lines))
                else:
                    st.info('Could not compute detailed contributions for this model type.')
            except Exception as e:
                st.error(f'Prediction failed: {e}')

    # -------------------------
    # Tab: Insights (population-level)
    # -------------------------
    with tab_insights:
        st.header('Population Insights & Advanced Charts')
        st.markdown('High-quality visual summaries for stakeholders.')
        st.subheader('Risk distribution overall')
        st.plotly_chart(px.pie(df_model, names='Substance_Risk', title='Overall Risk Distribution'), use_container_width=True)

        # Correlation heatmap for numeric columns
        st.subheader('Correlation (numeric features)')
        num_cols = df_model.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 1:
            corr = df_model[num_cols].corr()
            fig_corr, ax = plt.subplots(figsize=(8,6))
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
            st.pyplot(fig_corr)
        else:
            st.info('Not enough numeric columns for correlation heatmap.')

        # Risk by SES
        if 'Socioeconomic_Status' in df_model.columns:
            st.subheader('Risk by Socioeconomic Status')
            ses_df = df_model.groupby('Socioeconomic_Status')['Substance_Risk'].mean().reset_index()
            fig_ses = px.bar(ses_df, x='Socioeconomic_Status', y='Substance_Risk', text=ses_df['Substance_Risk'].apply(lambda x: f"{x:.1%}"))
            fig_ses.update_layout(yaxis_tickformat='.0%')
            st.plotly_chart(fig_ses, use_container_width=True)

    # -------------------------
    # Tab: Q&A
    # -------------------------
    with tab_qa:
        st.header('Q&A — Ask the data or model')
        st.markdown('Ask natural-language questions. Examples are suggested below.')
        q1,q2 = st.columns([3,1])
        with q1:
            user_q = st.text_input('Ask a question about the dataset/model:')
        with q2:
            if st.button('Ask'):
                if not user_q:
                    st.info('Type a question first.')
                else:
                    ans = answer_question(user_q, df_model, model=model, X_test_local=X_test, y_test_local=y_test, label_encoders_local=label_encoders, features_local=features)
                    st.markdown('**Answer:**')
                    st.code(ans)
                    # quick viz suggestions
                    if 'age' in user_q.lower() and 'Age_Group' in df_model.columns:
                        ag = df_model.groupby('Age_Group')['Substance_Risk'].mean().reset_index().sort_values('Substance_Risk', ascending=False)
                        st.plotly_chart(px.bar(ag, x='Age_Group', y='Substance_Risk', text=ag['Substance_Risk'].apply(lambda x: f"{x:.1%}")), use_container_width=True)
        st.markdown('**Suggested questions:**')
        st.write('- How many rows are in the dataset?')
        st.write('- What proportion are high risk?')
        st.write('- Which age group has highest risk?')
        st.write('- Show top 5 features')
        st.write('- Explain prediction for row 3')

    # Footer
    st.markdown('---')
    st.markdown("<div style='text-align:center; color:gray;'>Built with Streamlit — Pro Edition. Explanations are heuristic fallbacks when SHAP is not installed.</div>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
