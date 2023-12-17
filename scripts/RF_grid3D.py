import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


RF_grid = pd.read_csv("../data/RF_grid.csv")
RF_grid_more = pd.read_csv("../data/RF_grid_more.csv")

RF_grid_all = pd.concat([RF_grid, RF_grid_more])

# Plotting the 3D plot
# EVS

fig = plt.figure()
ax1 = fig.add_subplot(111, projection="3d")

ax1.scatter(
    RF_grid_all["param_n_estimators"],
    RF_grid_all["param_max_features"],
    RF_grid_all["mean_test_evs"],
    c=RF_grid_all["mean_test_evs"],
    cmap="jet",
    edgecolor="k",
)
ax1.plot_trisurf(
    RF_grid_all["param_n_estimators"],
    RF_grid_all["param_max_features"],
    RF_grid_all["mean_test_evs"],
    cmap=plt.cm.jet,
    linewidth=0.8,
    alpha=0.7,
)

ax1.set_xlabel("n_estimators")
ax1.set_ylabel("max_features")
ax1.set_zlabel("mean_test_evs")

# add colorbar
cbar = plt.colorbar(ax1.collections[0])

# RMSE

# ax2 = fig.add_subplot(122, projection="3d")

# ax2.scatter(
#     RF_grid_all["param_n_estimators"],
#     RF_grid_all["param_max_features"],
#     RF_grid_all["mean_test_rmse"],
#     c=RF_grid_all["mean_test_rmse"],
#     cmap="jet",
#     edgecolor="k",
# )
# ax2.plot_trisurf(
#     RF_grid_all["param_n_estimators"],
#     RF_grid_all["param_max_features"],
#     RF_grid_all["mean_test_rmse"],
#     cmap=plt.cm.jet,
#     linewidth=0.8,
#     alpha=0.7,
# )

# ax2.set_xlabel("n_estimators")
# ax2.set_ylabel("max_features")
# ax2.set_zlabel("mean_test_rmse")

# # add colorbar
# cbar = plt.colorbar(ax2.collections[0])

plt.show()
