import torch
import numpy as np
from matplotlib import rcParams, gridspec
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib

# Set the font to Times New Roman
rcParams["font.family"] = "Times New Roman"
SMALL_SIZE = 8
MEDIUM_SIZE = 9
BIGGER_SIZE = 9

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Set the figure size
fig = plt.figure(figsize=(3.5, 3.5 / 3.0))

# spec = gridspec.GridSpec(ncols=4, nrows=2, figure=fig)
# ax1 = fig.add_subplot(spec[0, 1:3])
# ax2 = fig.add_subplot(spec[1, 0:2])
# ax3 = fig.add_subplot(spec[1, 2:])

spec = gridspec.GridSpec(ncols=3, nrows=1, figure=fig)
ax1 = fig.add_subplot(spec[0, 0])
ax2 = fig.add_subplot(spec[0, 1])
ax3 = fig.add_subplot(spec[0, 2])


# Load the confusion matrices from files
epoch = 18
# root_path = Path("/media/jft/diskstation/results_trav/2024.09.14/125823")
root_path = Path("/media/jft/diskstation/results_trav/2024.09.14/162814")
matrixT = torch.load(root_path / f"conf_train_{epoch}.pth").numpy()
matrixA = torch.load(root_path / f"conf_testA_{epoch}.pth").numpy()
matrixB = torch.load(root_path / f"conf_testB_{epoch}.pth").numpy()

matrices = [matrixT, matrixA, matrixB]


# Define class labels
class_labels = ["Trav.", "Untrav."]
titles = ["Training", "Test A", "Test B"]

# for i in range(3):
for matrix, ax, title in zip(matrices, [ax1, ax2, ax3], titles):
    ax.imshow(matrix, cmap="Blues")
    ax.title.set_text(title)
    matrix_max = np.max(matrix)

    # Display the numbers in the confusion matrix
    for j in range(len(class_labels)):
        for k in range(len(class_labels)):
            color = 3 * [matrix[j, k] / matrix_max]
            ax.text(k, j, matrix[j, k], ha="center", va="center", color=color)

    # if i_row == 0:
    #     ax.set_xticks([])
    # else:
    ax.set_xticks(range(len(class_labels)))
    ax.set_xticklabels(class_labels)
    # if i_col == 0:
    ax.set_yticks(range(len(class_labels)))
    ax.set_yticklabels(class_labels)
    # else:
    #     ax.set_yticks([])

# fig.supxlabel("Predicted")
# fig.supylabel("True")
# Adjust the spacing between subplots
plt.tight_layout()

# Show the plot
plt.savefig("confusion_matrices.pdf")
plt.show()
