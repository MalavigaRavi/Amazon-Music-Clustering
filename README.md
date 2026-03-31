🎵 Amazon Music Grouping
📌 Project Synopsis

In this project, songs from Amazon Music are automatically grouped into meaningful clusters according to their audio characteristics using unsupervised machine learning. The model finds natural song groupings by identifying patterns in musical characteristics like energy, danceability, tempo, and acousticness rather than depending on manually assigned genres.

Data cleaning, feature engineering, clustering, evaluation, and visualization are all included in the project's demonstration of a full data science workflow.

## 🚀 Objectives

* Automatically group similar songs without using genre labels
* Discover hidden patterns in music audio features
* Build a foundation for recommendation systems and playlist generation

---

## 🧠 Business Use Cases

* **Personalized Playlists**: Automatically generate playlists with similar-sounding songs
* **Music Recommendation Systems**: Suggest songs based on user listening patterns
* **Artist Analysis**: Help artists identify competing songs with similar sound profiles
* **Market Segmentation**: Understand listener behavior across different music clusters

---

## Dataset

**File:** `single_genre_artists.csv`

### Features Used

* danceability
* energy
* loudness
* speechiness
* acousticness
* instrumentalness
* liveness
* valence
* tempo
* duration_ms

### Reference Columns (not used for clustering)

* track_id
* track_name
* artist_name

---

## 🛠️ Technologies & Libraries

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Streamlit (for interactive app)

---

## 🔍 Project Workflow

### 1. Data Exploration & Cleaning

* Loaded dataset using Pandas
* Removed duplicates and unnecessary columns
* Checked for missing values

### 2. Feature Selection

Selected audio features that describe rhythm, mood, and intensity of songs.

### 3. Data Scaling

Used **StandardScaler** to normalize feature ranges for distance-based clustering.

### 4. Clustering Algorithm

Applied **K-Means Clustering** to group songs into clusters.

### 5. Model Evaluation

Evaluated clustering quality using:

* Silhouette Score
* Davies-Bouldin Index
* Inertia (Elbow Method)

### 6. Dimensionality Reduction

Used **PCA** to visualize clusters in two dimensions.

### 7. Visualization & Interpretation

Generated scatter plots, bar charts, and cluster summaries to interpret cluster characteristics.

---

## 📈 Results

The model successfully grouped songs into distinct clusters such as:

* High-energy dance tracks
* Acoustic and calm songs
* Instrumental tracks
* Balanced pop-style music

These clusters can be used for playlist generation and music discovery.

---

## 🧪 Evaluation Metrics

| Metric               | Purpose                                          |
| -------------------- | ------------------------------------------------ |
| Silhouette Score     | Measures how well songs fit within their cluster |
| Davies-Bouldin Index | Measures cluster separation                      |
| Inertia              | Measures compactness of clusters                 |

---

## Project Structure

```
Amazon-Music-Clustering/
│
├── data/
│   └── single_genre_artists.csv
│
├── notebooks/
│   └── amazon_music_clustering.ipynb
│
├── app/
│   └── streamlit_app.py
│
├── outputs/
│   └── clustered_amazon_music.csv
│
└── README.md
```

---

## ▶️ How to Run the Project

### 1. Clone Repository

```
git clone https://github.com/your-username/amazon-music-clustering.git
cd amazon-music-clustering
```

### 2. Install Dependencies

```
pip install pandas numpy matplotlib seaborn scikit-learn streamlit
```

### 3. Run Jupyter Notebook

```
jupyter notebook
```

Open `amazon_music_clustering.ipynb`

### 4. Run Streamlit App (Optional)

```
streamlit run app/streamlit_app.py
```

---

## 📷 Sample Visualizations

* Elbow Method Plot
* PCA Cluster Scatter Plot
* Cluster Feature Heatmap
