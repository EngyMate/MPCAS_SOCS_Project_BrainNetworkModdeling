from matplotlib import pyplot as plt
import numpy as np
import data_management as dm
import Evaluation.evaluation_tools as et

DTI = dm.load_DTI_data(0) #246 * 246
fMRI = dm.load_fMRI_data(0)
FC = et.compute_fc(fMRI) #246 * 246
np.fill_diagonal(FC, 0)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot DTI (SC)
im0 = axes[0].imshow(DTI, cmap='viridis')
axes[0].set_title("Structural Connectivity (SC)")
axes[0].axis('off')
fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

# Plot FC
im1 = axes[1].imshow(
    FC,
    cmap='RdBu_r',
    vmin=-0.5,
    vmax=0.5
)
axes[1].set_title("Functional Connectivity (FC)")
axes[1].axis('off')
fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()