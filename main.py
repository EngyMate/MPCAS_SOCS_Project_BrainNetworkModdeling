from matplotlib import pyplot as plt
import data_management as dm
import numpy as np
from scipy.stats import pearsonr, linregress
import nonLinear_model as nlm
import stochastic_linear_model as slm
import diffusion_model as dfm

# -----------------------------------------------------
# Helper function: Pearson + regression columnwise
# -----------------------------------------------------
def compute_columnwise_pearson_and_fit(sim_data, real_data):
    pearsons = []
    slopes = []
    intercepts = []
    fits = []

    for col in range(real_data.shape[1]):
        x = sim_data[:, col]
        y = real_data[:, col]

        # Pearson
        r, _ = pearsonr(x, y)
        pearsons.append(r)

        # Linear regression
        slope, intercept, _, _, _ = linregress(x, y)
        slopes.append(slope)
        intercepts.append(intercept)

        # Fitted line for plotting
        fits.append(slope * x + intercept)

    return (
        np.array(pearsons),
        np.array(slopes),
        np.array(intercepts),
        fits
    )

# -----------------------------------------------------
# Plot function for one model
# -----------------------------------------------------
def plot_model(sim_data, real_data, fits, title):
    plt.figure(figsize=(10, 10))

    for col in range(sim_data.shape[1]):
        x = sim_data[:, col]
        y = real_data[:, col]

        # Scatter
        plt.scatter(x, y, s=2, alpha=0.03, color="gray")

        # Regression line
        plt.plot(x, fits[col], alpha=0.2)

    plt.title(title, fontsize=16)
    plt.xlabel("Simulated fMRI", fontsize=14)
    plt.ylabel("Actual fMRI", fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    DTI_1 = dm.load_DTI_data(0)
    fMRI_1 = dm.load_fMRI_data(0)

    alpha_range = np.arange(-3, 3, 0.1)
    beta_range  = np.arange(0, 6, 0.1)

    # ==== NON-LINEAR MODEL ====
    print("==== NON-LINEAR MODEL ====")
    BOLD_z_all = nlm.non_linear_bold_z_model(nlm.params, DTI_1, out_length=4800)

    # ==== STOCHASTIC LINEAR MODEL ====
    print("==== STOCHASTIC LINEAR MODEL ====")

    u_t_slm = slm.simulate_linear_model(DTI_1, best_alpha, best_beta, n_steps=4800)

    # ==== DIFFUSION MODEL ====
    print("==== DIFFUSION MODEL ====")
    beta_range_dfm = np.arange(0, 6, 0.1)
    times = np.arange(0, 500, 500/4800)

    best_beta_dfm, best_error_dfm, best_sim_dfm = dfm.compute_best_beta_fast(DTI_1, fMRI_1, beta_range_dfm, times)
    evals, evecs= dfm.prepare_laplacian_eigendecomposition(DTI_1)
    sim_dfm = dfm.nodewise_diffusion_timeseries_fast(evals, evecs, best_beta_dfm, times)
    print(best_beta_dfm)
    # -----------------------------------------------------
    # Compute stats for all 3 models
    # -----------------------------------------------------
    print("Computing all stats")
    results = {}

    results["Non-Linear"] = compute_columnwise_pearson_and_fit(BOLD_z_all, fMRI_1)
    results["SLM"]        = compute_columnwise_pearson_and_fit(u_t_slm,  fMRI_1)
    results["DFM"]        = compute_columnwise_pearson_and_fit(sim_dfm,   fMRI_1)

    # -----------------------------------------------------
    # Generate plots for all 3 models
    # -----------------------------------------------------
    print("Plotting")
    plot_model(BOLD_z_all, fMRI_1, results["Non-Linear"][3], "Non-Linear Model: Simulated vs Actual fMRI")
    plot_model(u_t_slm,    fMRI_1, results["SLM"][3],        "Stochastic Linear Model: Simulated vs Actual fMRI")
    plot_model(sim_dfm,    fMRI_1, results["DFM"][3],        "Diffusion Model: Simulated vs Actual fMRI")

"""
Brainnetome Atlas
BNA Label
BNA Notes
BrainMesh_ICBM152.nv
"""
#BNA_atlas = dm.load_mixed_excel(dm.PATH_excel_BNA_atlas)
#print(BNA_atlas[0][:])


