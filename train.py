import os
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering,
    DBSCAN
)

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

# =========================
# 1. Load Dataset
# =========================

df = pd.read_excel(
    "dataset/cleaned_output.xlsx",
    nrows=1000,
    engine="openpyxl"
)

# =========================
# 2. Clean Data
# =========================

df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# =========================
# 3. Feature Engineering
# =========================

df["Revenue"] = (
    df["Quantity"] *
    df["UnitPrice"]
)

# =========================
# 4. Label Encoding
# =========================

country_encoder = LabelEncoder()

df["Country_Label"] = country_encoder.fit_transform(
    df["Country"]
)

# =========================
# 5. Select Features
# =========================

X = df[
    [
        "Quantity",
        "UnitPrice",
        "Revenue",
        "Country_Label"
    ]
]

# =========================
# 6. Save Columns
# =========================

os.makedirs("models", exist_ok=True)

joblib.dump(
    X.columns,
    "models/columns.pkl"
)

joblib.dump(
    country_encoder,
    "models/country_encoder.pkl"
)

# =========================
# 7. Feature Scaling
# =========================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

# =========================
# 8. Models
# =========================

models = {
    "kmeans": KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10
    ),

    "hierarchical":
    AgglomerativeClustering(
        n_clusters=4,
        linkage="ward"
    ),

    "dbscan":
    DBSCAN(
        eps=0.5,
        min_samples=10
    )
}

# =========================
# 9. Train Models
# =========================

results = {}

# KMeans
kmeans = models["kmeans"]

kmeans.fit(X_scaled)

kmeans_labels = kmeans.labels_

results["KMeans"] = {
    "Silhouette":
    silhouette_score(
        X_scaled,
        kmeans_labels
    ),

    "Davies-Bouldin":
    davies_bouldin_score(
        X_scaled,
        kmeans_labels
    ),

    "Calinski-Harabasz":
    calinski_harabasz_score(
        X_scaled,
        kmeans_labels
    )
}

# Hierarchical
hierarchical = models["hierarchical"]

hierarchical_labels = hierarchical.fit_predict(
    X_scaled
)

results["Hierarchical"] = {
    "Silhouette":
    silhouette_score(
        X_scaled,
        hierarchical_labels
    ),

    "Davies-Bouldin":
    davies_bouldin_score(
        X_scaled,
        hierarchical_labels
    ),

    "Calinski-Harabasz":
    calinski_harabasz_score(
        X_scaled,
        hierarchical_labels
    )
}

# DBSCAN
dbscan = models["dbscan"]

dbscan_labels = dbscan.fit_predict(
    X_scaled
)

mask = dbscan_labels != -1

if len(set(dbscan_labels[mask])) > 1:

    results["DBSCAN"] = {
        "Silhouette":
        silhouette_score(
            X_scaled[mask],
            dbscan_labels[mask]
        ),

        "Davies-Bouldin":
        davies_bouldin_score(
            X_scaled[mask],
            dbscan_labels[mask]
        ),

        "Calinski-Harabasz":
        calinski_harabasz_score(
            X_scaled[mask],
            dbscan_labels[mask]
        )
    }

# =========================
# 10. Save Models
# =========================

joblib.dump(
    kmeans,
    "models/kmeans_model.pkl"
)

joblib.dump(
    hierarchical,
    "models/hierarchical_model.pkl"
)

joblib.dump(
    dbscan,
    "models/dbscan_model.pkl"
)

# =========================
# 11. Results
# =========================

results_df = pd.DataFrame(results).T

print("\nMODEL PERFORMANCE\n")

print(results_df)

best_model = results_df[
    "Silhouette"
].idxmax()

print(
    f"\nBest Model: {best_model}"
)

print(
    "\nTraining Completed Successfully!"
)