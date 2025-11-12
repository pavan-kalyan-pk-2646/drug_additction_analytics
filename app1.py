# app.py
"""
Streamlit app: Youth Substance Risk - Logistic / Decision Tree / Random Forest
Single-file app that replicates & extends R workflow in Python/streamlit.
"""

# -------------------------
# Required packages:
# pip install streamlit pandas numpy scikit-learn matplotlib seaborn plotly graphviz
# -------------------------

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import base64
import textwrap
import warnings
warnings.filterwarnings("ignore")

# -------------------------
# Internal CSS / HTML for styling
# -------------------------
st.set_page_config(page_title="Youth Substance Risk Explorer", layout="wide")

# Enhanced CSS for better visual appeal
_html_style = """
<style>
/* Page background + card style */
body { background-color: #f6f8fb; }
.app-header { 
    padding: 20px 20px; 
    border-radius: 12px; 
    background: linear-gradient(90deg, #4F46E5, #3B82F6); /* Richer gradient */
    margin-bottom: 25px; 
    color: white;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.app-title { 
    font-size: 28px; 
    font-weight: 800; 
    color: #ffffff; 
}
.app-sub { 
    font-size: 15px; 
    color: #e0e7ff; 
    margin-top: 6px; 
}

/* Sidebar tweaks */
[data-testid="stSidebar"] { background-color: #f8fafc; border-right: 1px solid #e2e8f0; }
[data-testid="stSidebar"] [data-testid="stTextInput"] > div > input { background-color: #f1f5f9; }

/* Metric Card Styling */
.metric-card-container {
    display: flex;
    justify-content: space-between;
    margin-bottom: 20px;
}
.metric-card { 
    background: #ffffff; 
    padding: 15px; 
    border-radius: 10px; 
    box-shadow: 0 2px 8px rgba(16, 24, 40, 0.05); 
    border-left: 5px solid #4F46E5; /* Accent color */
    flex: 1;
    margin: 0 10px;
}
.metric-card-title {
    font-size: 14px;
    color: #64748b;
    font-weight: 500;
}
.metric-card-value {
    font-size: 24px;
    font-weight: 700;
    color: #0f172a;
    margin-top: 5px;
}

/* Footer */
.footer { font-size:12px; color:#64748b; margin-top:20px; text-align: center; }

/* Custom button link style */
.button-link {
    background-color: #4F46E5;
    color: white;
    padding: 8px 15px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    border-radius: 6px;
    margin-top: 15px;
}
.button-link:hover {
    background-color: #3B82F6;
    color: white;
}
</style>
<div class='app-header'>
  <div class='app-title'>Youth Substance Risk Explorer 🧠</div>
  <div class='app-sub'>Train Logistic, Decision Tree, or Random Forest models to predict risk.</div>
</div>
"""
st.markdown(_html_style, unsafe_allow_html=True)

# -------------------------
# Helper functions
# -------------------------

def make_substance_risk(df, col_name="Drug_Experimentation"):
    """Create binary Substance_Risk based on median > median -> 1 else 0."""
    try:
        df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
        med = df[col_name].median(skipna=True)
        # Handle case where all values are the same (median = max)
        if df[col_name].nunique() > 1:
            df["Substance_Risk"] = (df[col_name] > med).astype(int)
        else: # If all values are the same, mark all as 0
            df["Substance_Risk"] = 0
    except Exception as e:
        st.error(f"Error while creating Substance_Risk: {e}")
    return df

def safe_label_encode(series):
    le = LabelEncoder()
    # fillna then encode
    filled = series.fillna("MISSING")
    return le.fit_transform(filled.astype(str)), le

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
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc}

def plot_confusion_matrix(cm):
    """Improved confusion matrix visualization."""
    fig, ax = plt.subplots(figsize=(4.5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", cbar=False,
                xticklabels=["Predicted 0","Predicted 1"], yticklabels=["Actual 0","Actual 1"], ax=ax)
    ax.set_ylabel("Actual Risk Level", fontsize=10)
    ax.set_xlabel("Predicted Risk Level", fontsize=10)
    ax.set_title("Confusion Matrix", fontsize=12)
    plt.tight_layout()
    return fig

def plot_pie_distribution(series, title="Distribution"):
    """Plots the distribution of the target variable with custom colors."""
    counts = series.value_counts().sort_index()
    labels = {0: "Low Risk (0)", 1: "High Risk (1)"}
    names = [labels.get(idx, f"Unknown ({idx})") for idx in counts.index]
    
    # Custom color mapping
    color_map = {0: '#22C55E', 1: '#EF4444'} 

    fig = px.pie(
        values=counts.values, 
        names=names, 
        title=title,
        # Use the mapped index values for 'color' argument (Series of colors)
        color=counts.index.map(color_map), 
    )
    fig.update_layout(title_x=0.5, margin=dict(t=50, b=0, l=0, r=0))
    return fig

def show_tree_plot(model, feature_names, max_depth=3):
    """Plots the Decision Tree visualization."""
    fig, ax = plt.subplots(figsize=(18, 9)) # Larger figure for clarity
    plot_tree(model, feature_names=feature_names, class_names=["Low Risk (0)","High Risk (1)"], 
              filled=True, rounded=True, max_depth=max_depth, fontsize=10, ax=ax)
    plt.title(f"Decision Tree Visualization (Max Depth: {max_depth})", fontsize=14)
    plt.tight_layout()
    return fig

def get_table_download_link(df_out, filename="predictions.csv"):
    """Generates a styled download link for a DataFrame."""
    csv = df_out.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a class="button-link" download="{filename}" href="data:text/csv;base64,{b64}">⬇️ Download predictions as CSV</a>'
    return href

@st.cache_data
def load_data(uploaded_file, use_sample, path_input):
    """Handles data loading from upload, path, or synthetic sample."""
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return df
    if use_sample:
        # Create a small synthetic sample for demo
        np.random.seed(0)
        n = 1000
        df = pd.DataFrame({
            "Age_Group": np.random.choice(["10-13","14-16","17-19"], n),
            "Gender": np.random.choice(["Male","Female","Other"], n),
            "Socioeconomic_Status": np.random.choice(["Low","Medium","High"], n),
            "School_Programs": np.random.choice(["Yes","No"], n),
            "Smoking_Prevalence": np.random.normal(0.2, 0.1, n).clip(0,1),
            "Mental_Health": np.random.choice(["Good","Average","Poor"], n, p=[0.5,0.3,0.2]),
            "Family_Background": np.random.choice(["Stable","Unstable"], n, p=[0.8,0.2]),
            "Access_to_Counseling": np.random.choice(["Yes","No"], n, p=[0.3,0.7]),
            "Substance_Education": np.random.choice(["Yes","No"], n, p=[0.4,0.6]),
            "Community_Support": np.random.choice(["Strong","Weak"], n, p=[0.6,0.4]),
            "Drug_Experimentation": np.floor(np.clip(
                np.random.normal(2, 2, n) + (
                    (np.random.choice([0,1], n, p=[0.7,0.3]) * 5)
                ), 0, 10
            )) 
        })
        return df
    if path_input:
        try:
            df = pd.read_csv(path_input)
            return df
        except Exception:
            return None
    return None

def main():
    # -------------------------
    # Sidebar: Data upload & controls
    # -------------------------
    st.sidebar.header("Data & Model Controls ⚙️")

    uploaded_file = st.sidebar.file_uploader("Upload CSV file (or leave blank to use sample)", type=["csv"])
    use_sample = False
    path_input = None

    if uploaded_file is None:
        st.sidebar.write("No upload detected.")
        col1, col2 = st.sidebar.columns([2,1])
        with col1:
            path_input = st.sidebar.text_input("Enter CSV path (optional)", value="D:/da datasets/youth_smoking_drug_data_10000_rows_expanded.csv")
        with col2:
            if st.sidebar.button("Use Sample (Synthetic)", type="primary"):
                use_sample = True
    else:
        path_input = None

    st.sidebar.markdown("---")
    
    # Model Selection
    model_choice = st.sidebar.selectbox("1. Select Model", ("Logistic Regression", "Decision Tree", "Random Forest"))
    
    st.sidebar.markdown("---")
    
    # Hyperparameters Expander
    with st.sidebar.expander("2. Model & Split Parameters"):
        test_size = st.slider("Test fraction", 0.05, 0.5, 0.2, 0.05)
        seed = st.number_input("Random seed", min_value=0, value=42, step=1)
        prob_threshold = st.slider("Classification threshold (P > 1)", 0.01, 0.99, 0.5, 0.01)

    # Display Toggles Expander
    with st.sidebar.expander("3. Optional Visuals"):
        show_roc = st.checkbox("Show ROC curve", value=True)
        show_predict_table = st.checkbox("Show predictions table", value=False)
        
        # Decision Tree specific control
        max_tree_depth = None
        if model_choice == "Decision Tree":
            max_tree_depth = st.slider("Max Tree Depth for Plot", 1, 10, 3)

    # -------------------------
    # Load dataset
    # -------------------------
    df = load_data(uploaded_file, use_sample, path_input)
    if df is None:
        st.error("No dataset loaded. Please upload a file, specify a path, or click 'Use Sample'.")
        st.stop()
        
    if uploaded_file is not None:
        st.success("Uploaded dataset loaded.")
    elif use_sample:
        st.info("Using built-in synthetic sample dataset.")
    elif path_input:
        st.success(f"Loaded file from path: {path_input}")


    # -------------------------
    # Data Preview and Check
    # -------------------------
    st.header("1. Data Overview")
    
    col_prev, col_stats = st.columns([3, 1])
    
    with col_prev:
        st.subheader("Dataset Preview (First 10 Rows)")
        st.dataframe(df.head(10), use_container_width=True)

    required_cols = [
        "Age_Group", "Gender", "Socioeconomic_Status", "School_Programs",
        "Smoking_Prevalence", "Mental_Health", "Family_Background",
        "Access_to_Counseling", "Substance_Education", "Community_Support",
        "Drug_Experimentation"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    
    with col_stats:
        st.subheader("Data Status")
        st.info(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        if missing:
            st.error(f"⚠️ Missing columns: {', '.join(missing)}. The app will attempt to proceed.")
        else:
            st.success("✅ All required columns found.")

    # -------------------------
    # Preprocessing
    # -------------------------
    st.header("2. Preprocessing & Splitting")
    
    df_proc = df.copy()
    df_proc = make_substance_risk(df_proc, col_name="Drug_Experimentation")
    st.markdown(f"**Target:** Created `Substance_Risk` (0/1) using median split on `Drug_Experimentation`.")

    cat_cols = ["Age_Group","Gender","Socioeconomic_Status","School_Programs",
                "Mental_Health","Family_Background","Access_to_Counseling",
                "Substance_Education","Community_Support"]

    label_encoders = {}
    for cc in cat_cols:
        if cc in df_proc.columns:
            arr, le = safe_label_encode(df_proc[cc])
            df_proc[f"{cc}_enc"] = arr
            label_encoders[cc] = le

    if "Smoking_Prevalence" in df_proc.columns:
        df_proc["Smoking_Prevalence"] = pd.to_numeric(df_proc["Smoking_Prevalence"], errors='coerce').fillna(df_proc["Smoking_Prevalence"].median())
    else:
        df_proc["Smoking_Prevalence"] = 0.0

    features = []
    for base in ["Age_Group","Gender","Socioeconomic_Status","Smoking_Prevalence",
                "Mental_Health","Family_Background","Access_to_Counseling",
                "Substance_Education","Community_Support"]:
        if base == "Smoking_Prevalence":
            features.append("Smoking_Prevalence")
        else:
            enc = base + "_enc"
            if enc in df_proc.columns:
                features.append(enc)

    st.info(f"**Using Features:** {', '.join(features)}")

    # Drop rows with missing target
    df_model = df_proc.dropna(subset=["Substance_Risk"]).copy()
    X = df_model[features]
    y = df_model["Substance_Risk"].astype(int)

    # Train/Test split
    stratify_param = y if len(np.unique(y)) > 1 else None 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=stratify_param)
    
    st.info(f"Split completed: **Train** rows: {len(X_train)}  |  **Test** rows: {len(X_test)}")

    # -------------------------
    # Model training
    # -------------------------
    st.header(f"3. Model Training: {model_choice} 🚀")

    model = None
    y_score = None
    y_pred = None 

    try:
        if model_choice == "Logistic Regression":
            model = LogisticRegression(max_iter=2000, solver='liblinear', random_state=seed)
            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_test)[:,1]
            y_pred = (y_prob >= prob_threshold).astype(int)
            y_score = y_prob
            st.success("Trained Logistic Regression model.")
            
            # Show coefficients
            coefs = pd.Series(model.coef_[0], index=X.columns).sort_values(key=abs, ascending=False)
            st.subheader("Model Coefficients (Sorted by Absolute Value)")
            st.dataframe(coefs.rename("Coefficient").to_frame())
            
        elif model_choice == "Decision Tree":
            model = DecisionTreeClassifier(random_state=seed, max_depth=None)
            model.fit(X_train, y_train)
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:,1]
                y_score = y_prob
            y_pred = model.predict(X_test)
            st.success("Trained Decision Tree model.")
            
            # --- FIX APPLIED HERE: Restrict max_depth for clean text output ---
            try:
                # Use max_depth=3 for a clean, concise rule summary
                tree_txt = export_text(model, feature_names=list(X.columns), max_depth=3) 
                st.subheader("Decision Tree Rules (Top 3 Levels Only)")
                st.code(tree_txt) 
            except Exception:
                st.info("Could not display text tree rules.")
            # -----------------------------------------------------------------
                
        elif model_choice == "Random Forest":
            model = RandomForestClassifier(random_state=seed, n_estimators=250, n_jobs=-1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:,1]
                y_score = y_prob
            st.success("Trained Random Forest model.")
            
            st.subheader("Random Forest Summary")
            st.info(f"**n_estimators:** {model.n_estimators} | **n_features:** {model.n_features_in_}")
            
    except Exception as e:
        st.error(f"Model training error: {e}")
        st.stop()


    # -------------------------
    # Evaluation & Visuals
    # -------------------------
    st.header("4. Model Evaluation & Results 📊")

    metrics = compute_metrics(y_test, y_pred, y_score=y_score)
    
    # Custom metric display
    st.markdown("<div class='metric-card-container'>", unsafe_allow_html=True)
    
    for title, key in [("Accuracy", "accuracy"), ("Precision", "precision"), ("Recall", "recall"), ("F1-Score", "f1")]:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-card-title'>{title}</div>
            <div class='metric-card-value'>{metrics[key]:.3f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("</div>", unsafe_allow_html=True)
    
    if metrics.get("auc") is not None:
        st.markdown(f"**ROC AUC (Area Under Curve):** <span style='font-size: 20px; font-weight: 700; color: #4F46E5;'>{metrics['auc']:.3f}</span>", unsafe_allow_html=True)
    st.markdown("---")

    col_cm, col_dist = st.columns(2)

    with col_cm:
        cm = confusion_matrix(y_test, y_pred)
        st.pyplot(plot_confusion_matrix(cm))
    
    with col_dist:
        st.plotly_chart(plot_pie_distribution(df_model["Substance_Risk"], title="Overall Substance Risk Distribution"), use_container_width=True)

    # ROC curve
    if show_roc and y_score is not None:
        st.subheader("Receiver Operating Characteristic (ROC) Curve")
        fpr, tpr, thr = roc_curve(y_test, y_score)
        fig_roc = plt.figure(figsize=(6,5))
        plt.plot(fpr, tpr, linewidth=2, label=f'{model_choice} (AUC = {metrics["auc"]:.3f})', color='#4F46E5')
        plt.plot([0,1],[0,1], linestyle="--", linewidth=1, color='grey')
        plt.xlabel("False Positive Rate (1 - Specificity)")
        plt.ylabel("True Positive Rate (Sensitivity/Recall)")
        plt.title("ROC Curve")
        plt.legend()
        plt.grid(alpha=0.2)
        st.pyplot(fig_roc)

    # -------------------------
    # Feature importance / coefficients
    # -------------------------
    st.header("5. Feature Analysis")

    if model_choice == "Logistic Regression":
        st.subheader("Feature Importance: Absolute Coefficients")
        coef_df = pd.Series(model.coef_[0], index=X.columns).abs().sort_values(ascending=True)
        fig_imp = px.bar(
            x=coef_df.values, 
            y=coef_df.index, 
            orientation='h', 
            title="Feature Importance (Absolute Coefficient Magnitude)",
            color_discrete_sequence=['#10B981'] # Green color
        )
        st.plotly_chart(fig_imp, use_container_width=True)
        
    elif model_choice in ["Decision Tree", "Random Forest"]:
        st.subheader(f"Feature Importance: Gini Importance ({model_choice})")
        try:
            importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True)
            
            color_seq = ['#EF4444'] if model_choice == "Decision Tree" else ['#3B82F6'] 

            fig = px.bar(
                x=importances.values, 
                y=importances.index, 
                orientation='h', 
                title=f"{model_choice} Feature Importance",
                color_discrete_sequence=color_seq
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Importance Scores Table")
            imp_series = importances.sort_values(ascending=False).rename("Importance")
            st.dataframe(imp_series.to_frame())
            
        except Exception as e:
            st.write("Could not compute importances:", e)
        
        # Decision Tree specific plot
        if model_choice == "Decision Tree":
            try:
                st.subheader("Decision Tree Visualization (First Levels)")
                st.pyplot(show_tree_plot(model, feature_names=list(X.columns), max_depth=max_tree_depth))
            except Exception as e:
                st.info(f"Could not plot tree (graphviz may be needed): {e}")

    # -------------------------
    # Prediction Table & Export
    # -------------------------
    if show_predict_table:
        preds_df = X_test.copy()
        preds_df = preds_df.reset_index(drop=True)
        preds_df["Actual_Risk"] = y_test.reset_index(drop=True)
        if y_score is not None:
            preds_df["Prob_HighRisk"] = np.round(y_score, 4)
        preds_df["Predicted_Risk"] = y_pred
        
        st.header("6. Test Set Predictions")
        st.markdown("#### Predictions (Test Set Sample)")
        st.dataframe(preds_df.head(200), use_container_width=True)
        
        st.markdown(get_table_download_link(preds_df, filename="predictions_testset.csv"), unsafe_allow_html=True)

    # -------------------------
    # Footer
    # -------------------------
    st.markdown("---")
    st.markdown("<div class='footer'>Built using Python & Streamlit | Replicating R workflow | **Tip:** Change parameters in the sidebar and rerun.</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()