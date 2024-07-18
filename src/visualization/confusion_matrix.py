
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

ROOT_DIR = '/Users/tousif/LightHART-tf/'

# Confusion matrix values
true_positive = 25
true_negative = 5
false_positive = 30
false_negative = 5


# Create confusion matrix
confusion_matrix = np.array([[true_positive, false_negative],
                             [false_positive, true_negative]])

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.4)
ax = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, annot_kws={"size": 18, "weight": "bold"})

# Set labels, title, and ticks with specified font properties
font = {'family': 'DejaVu Serif', 'weight': 'bold', 'size': 18}

ax.set_xlabel('Predicted Labels', fontsize=18, fontname='DejaVu Serif', fontweight='bold')
ax.set_ylabel('True Labels', fontsize=18, fontname='DejaVu Serif', fontweight='bold')
ax.set_xticklabels(['Fall', 'ADL'], fontdict=font)
ax.set_yticklabels(['Fall', 'ADL'], fontdict=font)

# Show the plot
plt.savefig(os.path.join(ROOT_DIR, 'reports/figures/cm_scratch.png'), dpi = 300)