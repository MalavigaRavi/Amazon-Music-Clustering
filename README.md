# 🎵 Amazon Music Clustering using Machine Learning

## 📌 Project Overview

This project analyzes Amazon Music song data using Unsupervised Machine Learning techniques to discover hidden patterns among songs based on their audio features.

The project performs data cleaning, exploratory data analysis, feature engineering, dimensionality reduction, and clustering using multiple clustering algorithms.

---

## 🎯 Objective

To group similar songs into clusters based on musical characteristics such as:

- Danceability
- Energy
- Loudness
- Acousticness
- Speechiness
- Tempo
- Instrumentalness
- Liveness
- Valence
- Popularity

These clusters can be used for:

- Music Recommendation Systems
- Playlist Generation
- Genre Discovery
- User Personalization
- Music Trend Analysis

---

# Dataset

**Dataset Name**

single_genre_artists.csv

Dataset contains approximately:

- 95,836 Songs
- 23 Features

### Features

- Song ID
- Song Name
- Artist ID
- Artist Name
- Popularity
- Duration
- Explicit
- Release Date
- Danceability
- Energy
- Loudness
- Speechiness
- Acousticness
- Instrumentalness
- Liveness
- Valence
- Tempo
- Followers
- Genre
- Artist Popularity

---

# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy
- Visual Studio

---

# Machine Learning Algorithms

### K-Means Clustering

Used to create clusters of similar songs.

---

### DBSCAN

Density-based clustering to identify outliers and dense regions.

---

### Hierarchical Clustering

Used to understand relationships between clusters using dendrograms.

---

### PCA (Principal Component Analysis)

Used to reduce dimensions and visualize clusters.

---

# Project Workflow

1. Import Libraries
2. Load Dataset
3. Data Cleaning
4. Missing Value Analysis
5. Duplicate Removal
6. Feature Engineering
7. Data Scaling
8. Exploratory Data Analysis
9. K-Means Clustering
10. Elbow Method
11. Silhouette Score Evaluation
12. PCA Visualization
13. DBSCAN
14. Hierarchical Clustering
15. Cluster Interpretation

---

# Exploratory Data Analysis

Performed:

- Dataset Information
- Statistical Summary
- Missing Values
- Duplicate Records
- Data Types
- Correlation Analysis
- Feature Distribution
- Outlier Detection

---

# Feature Engineering

Converted:

Release Date

↓

Year

Used only numerical audio features for clustering.

---

# Evaluation

Cluster quality evaluated using:

- Elbow Method
- Silhouette Score

---
# Installation

```bash
git clone https://github.com/yourusername/Amazon-Music-Clustering.git
```

```bash
cd Amazon-Music-Clustering
```

```bash
pip install -r requirements.txt
```

---

# Run

```bash
jupyter notebook
```

Open

Amazon_Music_Clustering.ipynb

Run all cells.

---

# Libraries

```
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
```

---

# Future Improvements

- Spotify API Integration
- Streamlit Dashboard
- Real-time Song Recommendation
- Deep Learning Autoencoders
- Recommendation Engine
- Genre Prediction

---

# Applications

- Amazon Music
- Spotify
- YouTube Music
- Apple Music
- Personalized Playlist Generation
- Music Recommendation Systems

---

# Author

**Malaviga Ravi**

MBA Business Analytics

Python | SQL | Power BI | Machine Learning | Data Science
