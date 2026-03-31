import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Amazon Music Clustering", layout="wide")

# Title
st.title("🎵 Amazon Music Clustering App")
st.write("Cluster songs based on audio features using K-Means")

# ==============================
# File Upload
# ==============================
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:

    # --------------------------
    # Load Data
    # --------------------------
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Raw Data")
    st.dataframe(df.head())

    # --------------------------
    # Remove non-numeric columns
    # --------------------------
    df_model = df.select_dtypes(include=['number'])

    st.subheader("🧹 Cleaned Numeric Data")
    st.write("Columns used for clustering:")
    st.write(df_model.columns.tolist())

    st.write(f"Dataset shape after cleaning: {df_model.shape}")

    # Safety check
    if df_model.empty:
        st.error("No numeric columns found. Please upload a valid dataset.")
        st.stop()

    # --------------------------
    # Scaling
    # --------------------------
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_model)

    # --------------------------
    # Select K
    # --------------------------
    st.subheader("⚙️ Select Number of Clusters")
    k = st.slider("Choose number of clusters (K)", 2, 10, 4)

    # --------------------------
    # Apply KMeans
    # --------------------------
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)

    # Add cluster labels
    df['Cluster'] = clusters

    # --------------------------
    # Show Clustered Data
    # --------------------------
    st.subheader("📌 Clustered Data")
    st.dataframe(df.head())

    # --------------------------
    # PCA Visualization
    # --------------------------
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)

    st.subheader("📉 Cluster Visualization (PCA)")

    fig, ax = plt.subplots()
    scatter = ax.scatter(pca_data[:, 0], pca_data[:, 1], c=clusters)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title("Clusters Visualization")
    st.pyplot(fig)

    # --------------------------
    # Cluster Summary
    # --------------------------
    st.subheader("📊 Cluster Summary")

    summary = df_model.copy()
    summary['Cluster'] = clusters
    cluster_summary = summary.groupby('Cluster').mean()

    st.dataframe(cluster_summary)

    # --------------------------
    # Download Results
    # --------------------------
    st.subheader("⬇️ Download Results")

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Clustered Data",
        data=csv,
        file_name="clustered_songs.csv",
        mime="text/csv"
    )

else:
    st.info("Please upload a CSV file to proceed.")