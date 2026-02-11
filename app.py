# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 20:49:34 2025

@author: susha
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

st.set_page_config(page_title="E-commerce Customer Segmentation", layout="wide")

# ---------- Helper: load data & models ----------
@st.cache_data
def load_rfm():
    return pd.read_csv("rfm_output.csv")

@st.cache_resource
def load_models():
    scaler = joblib.load("scaler.pkl")
    kmeans = joblib.load("kmeans.pkl")
    rf_model = joblib.load("rf_model.pkl")
    return scaler, kmeans, rf_model


# ---------- Sidebar ----------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "RFM Table", "Cluster Visualization", "Predict Segment"]
)

st.sidebar.info("If you haven’t yet, run: `python main.py` in this folder.")


# ---------- Page 1: Overview ----------
if page == "Overview":
    st.title("📦 E-commerce Customer Segmentation & CLV")

    st.markdown(
        """
        This app uses **RFM (Recency, Frequency, Monetary)** and **CLV (Customer Lifetime Value)**  
        to segment customers and predict which segment a new customer belongs to.

        **Pipeline:**
        1. Data cleaning & TotalAmount creation  
        2. RFM calculation per customer  
        3. CLV = Monetary × Frequency  
        4. KMeans clustering into 4 segments  
        5. RandomForest model to predict segment  
        """
    )


# ---------- Page 2: RFM Table ----------
elif page == "RFM Table":
    st.title("📊 RFM & CLV Table")

    try:
        rfm = load_rfm()
        st.write("Sample of RFM data:")
        st.dataframe(rfm.head())

        st.write("Basic statistics:")
        st.dataframe(rfm[["Recency","Frequency","Monetary","CLV"]].describe())
    except FileNotFoundError:
        st.error("rfm_output.csv not found. Please run `python main.py` first.")


# ---------- Page 3: Cluster Visualization ----------
elif page == "Cluster Visualization":
    st.title("📈 Customer Segments (KMeans Clusters)")

    try:
        rfm = load_rfm()

        st.write("Cluster counts:")
        st.bar_chart(rfm["Cluster"].value_counts())

        st.write("Recency vs Monetary by Cluster:")
        fig, ax = plt.subplots()
        scatter = ax.scatter(
            rfm["Recency"], rfm["Monetary"], c=rfm["Cluster"]
        )
        ax.set_xlabel("Recency")
        ax.set_ylabel("Monetary")
        st.pyplot(fig)

        st.write("Frequency vs Monetary by Cluster:")
        fig2, ax2 = plt.subplots()
        ax2.scatter(
            rfm["Frequency"], rfm["Monetary"], c=rfm["Cluster"]
        )
        ax2.set_xlabel("Frequency")
        ax2.set_ylabel("Monetary")
        st.pyplot(fig2)

    except FileNotFoundError:
        st.error("rfm_output.csv not found. Please run `python main.py` first.")


# ---------- Page 4: Predict Segment ----------
elif page == "Predict Segment":
    st.title("🔮 Predict Customer Segment")

    try:
        _, _, rf_model = load_models()
    except FileNotFoundError:
        st.error("Model files not found. Please run `python main.py` first.")
    else:
        st.markdown("Enter RFM values for a customer:")

        recency = st.number_input("Recency (days since last purchase)", min_value=0, max_value=1000, value=30)
        frequency = st.number_input("Frequency (number of orders)", min_value=0, max_value=1000, value=10)
        monetary = st.number_input("Monetary (total spent)", min_value=0.0, max_value=100000.0, value=500.0)

        if st.button("Predict Segment"):
            import pandas as pd
            inp = pd.DataFrame([[recency, frequency, monetary]], columns=["Recency","Frequency","Monetary"])
            pred_cluster = int(rf_model.predict(inp)[0])
            st.success(f"Predicted Customer Segment (Cluster): **{pred_cluster}**")