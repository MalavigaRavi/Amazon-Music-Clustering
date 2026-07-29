import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import base64
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

# --- Page Configuration ---
st.set_page_config(page_title="Music Insight Dashboard", layout="wide")
sns.set_theme(style="whitegrid")

def get_base64(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except:
        return ""

# --- Background Styling ---
# Using the selected background image
img_path = "/content/Black Friday 2025 music unlimited.webp"
bin_str = get_base64(img_path)


# Updated CSS for dark blurred background behind text
page_bg_style = f"""

"""
st.markdown(page_bg_style, unsafe_allow_html=True)

# --- Header ---
st.title("፨ Song Clustering Discovery App 🎵")
st.markdown("### Discover patterns in music using machine learning.")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️\u2005 Settings")
    uploaded_file = st.file_uploader("Upload Song CSV", type=["csv"])
    k_clusters = st.slider("Select Clusters (k)", 2, 10, 2)
    st.divider()

if uploaded_file is not None:
    @st.cache_data
    def load_data(file):
        return pd.read_csv(file).dropna(subset=["danceability", "energy"])

    df = load_data(uploaded_file)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    default_feats = ["danceability", "energy", "loudness", "acousticness", "instrumentalness", "valence", "tempo"]
    selected_features = [f for f in default_feats if f in numeric_cols]
    selected_features = st.multiselect("📑 Select features for clustering:", numeric_cols, default=selected_features)

    if len(selected_features) >= 2:
        X_scaled = StandardScaler().fit_transform(df[selected_features])
        kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
        df["cluster"] = kmeans.fit_predict(X_scaled)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("✨ Clustering Quality")
            sample_idx = np.random.choice(len(X_scaled), min(1000, len(X_scaled)), replace=False)
            score = silhouette_score(X_scaled[sample_idx], df["cluster"].iloc[sample_idx])
            st.metric("Silhouette Score", f"{score:.3f}")

        with col2:
            st.subheader("🏷️ Cluster Profiles")
            c_means = df.groupby("cluster")[selected_features].mean()
            g_means = df[selected_features].mean()
            for i in range(k_clusters):
                label = "Chill/Acoustic 🎧" if c_means.loc[i, "acousticness"] > g_means["acousticness"] else "Energetic/Party 🎸"
                st.write(f"**Cluster {i}:** {label}")

        st.divider()
        v_tab1, v_tab2, v_tab3, v_tab4, v_tab5 = st.tabs(["📍 2D Projections", "🌌 4D Explorer", "📊 Analysis", "📈 Distributions", "📈 Optimization"])

        with v_tab1:
            c1, c2 = st.columns(2)
            with c1:
                st.write("**PCA (Linear Reduction)**")
                pca_res = PCA(n_components=2).fit_transform(X_scaled)
                fig_pca, ax_pca = plt.subplots(facecolor='none')
                sns.scatterplot(x=pca_res[:,0], y=pca_res[:,1], hue=df["cluster"], palette="viridis", ax=ax_pca)
                st.pyplot(fig_pca)
            with c2:
                st.write("**t-SNE (Non-linear Projection)**")
                tsne_res = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto').fit_transform(X_scaled[sample_idx])
                fig_tsne, ax_tsne = plt.subplots(facecolor='none')
                sns.scatterplot(x=tsne_res[:,0], y=tsne_res[:,1], hue=df["cluster"].iloc[sample_idx], palette="magma", ax=ax_tsne)
                st.pyplot(fig_tsne)

        with v_tab2:
            st.subheader("Interactive 4D Visualization")
            fig_4d = px.scatter_3d(df.sample(min(1500, len(df))), x='danceability', y='energy', z='loudness', color='cluster', size='popularity_songs', opacity=0.7, template="plotly_dark")
            st.plotly_chart(fig_4d, use_container_width=True)

        with v_tab3:
            st.subheader("Feature Heatmap & Cluster Means")
            st.bar_chart(c_means.T)
            fig_hm, ax_hm = plt.subplots()
            sns.heatmap(c_means, annot=True, cmap='coolwarm', ax=ax_hm)
            st.pyplot(fig_hm)

        with v_tab4:
            st.subheader("Feature Distributions by Cluster")
            dist_feat = st.selectbox("Select Feature to view Distribution:", selected_features)
            fig_dist, ax_dist = plt.subplots()
            sns.kdeplot(data=df, x=dist_feat, hue='cluster', fill=True, palette='viridis', ax=ax_dist)
            st.pyplot(fig_dist)

        with v_tab5:
            inertia = [KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_scaled).inertia_ for k in range(1, 8)]
            fig_k, ax_k = plt.subplots(); ax_k.plot(range(1, 8), inertia, marker='o'); st.pyplot(fig_k)

        st.download_button("📥 Download Results", df.to_csv(index=False), "clustered_songs.csv")
     
