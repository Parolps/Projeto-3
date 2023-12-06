import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

pd.options.display.max_columns = 999

# Read data
df = pd.read_csv("data/drd2_data.csv")

# Get X and y
# X = df.iloc[:, :-1]
X = df.iloc[:, :-1][[i for i in df.columns if "D" in i]]
y = df[["target"]]

# PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
scatter = ax.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    X_pca[:, 2],
    c=y.values.ravel(),
    cmap="viridis",
    edgecolor="k",
    s=20,
)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
# title with explained variance
ax.set_title(f"Explained Variance: {pca.explained_variance_ratio_.mean():.3f}")
fig.colorbar(scatter)
plt.show()
