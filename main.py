# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 20:47:47 2025

@author: susha
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

DATA_PATH = "data.csv"

def prepare_data():
    # 1. Load data
    data = pd.read_csv(DATA_PATH, encoding="latin1", low_memory=False)

    # 2. Basic cleaning
    data = data.dropna(subset=["CustomerID"])
    data = data[data["Quantity"] > 0]
    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"])
    data["TotalAmount"] = data["Quantity"] * data["UnitPrice"]

    # 3. RFM calculation
    snapshot_date = data["InvoiceDate"].max() + timedelta(days=1)

    rfm = data.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (snapshot_date - x.max()).days,  # Recency
        "InvoiceNo": "count",                                     # Frequency
        "TotalAmount": "sum"                                      # Monetary
    }).reset_index()

    rfm.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]

    # 4. Simple CLV (Monetary * Frequency) – like in the slide
    rfm["CLV"] = rfm["Monetary"] * rfm["Frequency"]

    # 5. Scale RFM features
    scaler = StandardScaler()
    X_rfm = rfm[["Recency", "Frequency", "Monetary"]]
    X_scaled = scaler.fit_transform(X_rfm)

    # 6. KMeans clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    rfm["Cluster"] = kmeans.fit_predict(X_scaled)

    # 7. Train a RandomForest to predict cluster from RFM
    X = rfm[["Recency", "Frequency", "Monetary"]]
    y = rfm["Cluster"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)

    # 8. Save outputs for Streamlit app
    rfm.to_csv("rfm_output.csv", index=False)
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(kmeans, "kmeans.pkl")
    joblib.dump(rf, "rf_model.pkl")

    print("✅ Data preparation complete.")
    print("Saved: rfm_output.csv, scaler.pkl, kmeans.pkl, rf_model.pkl")


if __name__ == "__main__":
    prepare_data()