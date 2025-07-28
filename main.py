# main.py

import pandas as pd

# Import your own modules
from analysis.eda import perform_eda
from utils.preprocessing import scale_features
from analysis.clustering import find_optimal_clusters, apply_kmeans
from analysis.visualization import plot_pca, plot_tsne

# Step 1: Load the dataset
print("📂 Loading dataset...")
df = pd.read_csv("data/Mall_Customers.csv", sep="\t")


# Only rename columns if they are correct
expected_cols = ['CustomerID', 'Gender', 'Age', 'AnnualIncome', 'SpendingScore']
if len(df.columns) == 5:
    df.columns = expected_cols
else:
    print("⚠️ Column mismatch. Please check your CSV formatting.")
    print("Loaded columns:", df.columns)
    exit()

# Step 2: Perform EDA
print("\n🔍 Performing EDA...")
perform_eda(df)

# Step 3: Preprocess the data
print("\n⚙️ Scaling features...")
X_scaled = scale_features(df)

# Step 4: Elbow Method to find optimal clusters
print("\n📊 Finding optimal number of clusters...")
find_optimal_clusters(X_scaled)

# Step 5: Apply KMeans Clustering
print("\n📌 Applying K-Means clustering (k=5)...")
df = apply_kmeans(X_scaled, df, n_clusters=5)

# Step 6: PCA Visualization
print("\n🎨 PCA visualization...")
plot_pca(X_scaled, df)

# Step 7: t-SNE Visualization
print("\n🎨 t-SNE visualization...")
plot_tsne(X_scaled, df)

# Step 8: Cluster Summary
print("\n📄 Cluster summary:")
summary = df.groupby('Cluster')[['Age', 'AnnualIncome', 'SpendingScore']].mean()
print(summary)

# Save to file
summary.to_csv("report/summary.csv", index=True)
print("\n✅ Cluster summary saved to 'report/summary.csv'")
