import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🚨",
    layout="wide"
)

st.title("🚨 AI Fraud Detection Dashboard")
st.markdown("Upload transaction data and detect potential fraud using a tuned XGBoost model.")


# -----------------------------
# Custom Fintech Dark Theme
# -----------------------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}

h1, h2, h3 {
    color: #f8fafc;
}

[data-testid="stMetricValue"] {
    color: #38bdf8;
    font-weight: bold;
}

[data-testid="stMetricLabel"] {
    color: #cbd5f5;
}

.stDataFrame {
    border-radius: 10px;
}

button[kind="primary"] {
    background-color: #2563eb;
    color: white;
    border-radius: 8px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Model Performance Summary
# -----------------------------

st.subheader("📊 Model Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric("ROC-AUC", "0.899")
col2.metric("PR-AUC", "0.180")
col3.metric("Best Threshold", "0.89")
col4.metric("Cost Reduction", "≈ 8%")

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("models/tuned_xgboost.pkl")

model = load_model()

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    col1, col2 = st.columns(2)

    with col1:
        threshold = st.slider(
            "Select Fraud Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.90,
            step=0.01
        )

    with col2:
        sample_size = st.number_input(
            "Rows to Process (for speed)",
            min_value=10,
            max_value=len(df),
            value=min(1000, len(df))
        )

    # Run fraud detection
    if st.button("🔍 Run Fraud Detection"):

        df_sample = df.head(sample_size).copy()

        # Run model prediction first
        probabilities = model.predict_proba(df_sample)[:, 1]
        predictions = (probabilities >= threshold).astype(int)

        df_sample["Fraud_Probability"] = probabilities
        df_sample["Prediction"] = predictions

        # Now check high risk transactions
        high_risk_count = (df_sample["Fraud_Probability"] >= 0.9).sum()

        if high_risk_count > 0:
            st.error(f"🚨 ALERT: {high_risk_count} HIGH-RISK TRANSACTIONS DETECTED")
        else:
            st.success("✅ No high-risk fraud transactions detected")

        def risk_level(p):
            if p >= 0.9:
                return "🔴 High Risk"
            elif p >= 0.6:
                return "🟡 Medium Risk"
            else:
                return "🟢 Low Risk"

        df_sample["Risk_Level"] = df_sample["Fraud_Probability"].apply(risk_level)

        # Save results in session state
        st.session_state["df_sample"] = df_sample


    # If results exist, display them
    if "df_sample" in st.session_state:

        df_sample = st.session_state["df_sample"]

        st.subheader("📈 Results")

        st.subheader("🚨 High-Risk Transactions")

        high_risk_df = df_sample[df_sample["Fraud_Probability"] >= 0.9]

        if len(high_risk_df) > 0:
            st.dataframe(high_risk_df)
        else:
            st.write("No high-risk transactions found.")

        st.subheader("🚨 Fraud Risk Monitor")

        avg_prob = df_sample["Fraud_Probability"].mean()

        st.write("Average Fraud Probability")

        st.progress(float(avg_prob))
        st.write(f"{avg_prob:.2%} fraud risk")

        # -----------------------------
        # Fraud Probability Distribution
        # -----------------------------
        st.subheader("📊 Fraud Probability Distribution")

        fig3, ax3 = plt.subplots(figsize=(8,5))

        ax3.hist(df_sample["Fraud_Probability"], bins=30)

        ax3.set_xlabel("Fraud Probability")
        ax3.set_ylabel("Number of Transactions")
        ax3.set_title("Distribution of Fraud Risk")

        st.pyplot(fig3)


        # -----------------------------
        # Top Suspicious Transactions Chart
        # -----------------------------
        st.subheader("📊 Fraud Probability by Transaction")

        top_risk = df_sample.sort_values(
            "Fraud_Probability", ascending=False
        ).head(10)

        fig4, ax4 = plt.subplots(figsize=(8,5))

        ax4.barh(
            top_risk.index.astype(str),
            top_risk["Fraud_Probability"]
        )

        ax4.set_xlabel("Fraud Probability")
        ax4.set_ylabel("Transaction Index")
        ax4.set_title("Top 10 Most Suspicious Transactions")

        ax4.invert_yaxis()

        st.pyplot(fig4)

        colA, colB, colC = st.columns(3)

        colA.metric("Total Transactions", len(df_sample))
        colB.metric("Predicted Frauds", int(df_sample["Prediction"].sum()))
        colC.metric("Fraud Rate (%)", round(100 * df_sample["Prediction"].mean(), 2))

        def highlight_risk(row):
            if row["Fraud_Probability"] >= 0.9:
                return ['background-color: #ffcccc; color: black'] * len(row)   # red
            elif row["Fraud_Probability"] >= 0.6:
                return ['background-color: #fff3cd; color: black'] * len(row)   # yellow
            else:
                return ['background-color: #d4edda; color: black'] * len(row)   # green

        st.dataframe(df_sample.style.apply(highlight_risk, axis=1))

        # Download button
        csv = df_sample.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Results",
            data=csv,
            file_name="fraud_predictions.csv",
            mime="text/csv"
        )

        # SHAP explainability
        st.subheader("🧠 Model Explainability (SHAP)")

        preprocessor = model.named_steps["preprocessor"]
        xgb_model = model.named_steps["classifier"]

        X_transformed = preprocessor.transform(df_sample)

        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_transformed)

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X_transformed, show=False)
        st.pyplot(fig)

        # Individual explanation
        st.subheader("🔍 Explain Individual Transaction")

        row_index = st.number_input(
            "Select transaction index",
            min_value=0,
            max_value=len(df_sample) - 1,
            value=0
        )

        if st.button("Explain Prediction"):

            single_row = df_sample.iloc[[row_index]]

            X_single = preprocessor.transform(single_row)

            shap_values_single = explainer(X_single)

            st.write("### SHAP Explanation (Waterfall Plot)")

            fig2, ax2 = plt.subplots(figsize=(8,5))

            shap.plots.waterfall(shap_values_single[0], show=False)

            st.pyplot(fig2)