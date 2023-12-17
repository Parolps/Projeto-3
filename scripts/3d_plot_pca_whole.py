import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

pd.options.display.max_columns = 999

# Read data
df = pd.read_csv("data/drd2_data.csv")
FP_cols = [col for col in df.columns if "F" in col]
D_cols = [col for col in df.columns if "D" in col]
D_data = df[D_cols].copy()
FP_data = df[FP_cols].copy()

# Get X and y

# X = FP_data
# X = D_data
X = df.drop(["target"], axis=1)
y = df[["target"]]


# pipeline to only scale D_cols

scaler = StandardScaler()
# scaler = StandardScaler()

ct = ColumnTransformer(
    [("scaler", scaler, D_cols)],
    remainder="passthrough",
    verbose_feature_names_out=True,
)

pipe = Pipeline(
    [
        ("ct", ct),
        ("pca", PCA(n_components=0.95)),
    ]
)

# PCA
# pca = PCA(n_components=3)
X_pca = pipe.fit_transform(X)

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
pca = pipe.named_steps["pca"]
ax.set_title(
    f"Explained Variance: {pca.explained_variance_ratio_.mean():.5f} \n scaler: {scaler.__class__.__name__}"
)
fig.colorbar(scatter)
plt.show()
