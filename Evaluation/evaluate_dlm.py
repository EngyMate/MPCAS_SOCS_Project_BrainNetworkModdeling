import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, spearmanr

import data_management as dm
import diffusion_model as dlm


# -------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------

def compute_fc(timeseries):
    """
    timeseries shape: (timepoints, nodes)
    returns FC: (nodes x nodes)
    """
    ts = (timeseries - timeseries.mean(axis=0)) / timeseries.std(axis=0)
    return np.corrcoef(ts, rowvar=False)


def sc_fc_corr(SC, FC):
    mask = np.triu(np.ones_like(SC), k=1).astype(bool)

    sc_vals = SC[mask]
    fc_vals = FC[mask]

    r_p, p_p = pearsonr(sc_vals, fc_vals)
    r_s, p_s = spearmanr(sc_vals, fc_vals)

    return r_p, p_p, r_s, p_s, sc_vals, fc_vals


# -------------------------------------------------------------
# MAIN SCRIPT
# -------------------------------------------------------------

if __name__ == "__main__":

    individuals = 100

    best_fitness, best_beta = dlm.get_best()

    # Initialize arrays for storing results per individual
    pear_emp = np.zeros(individuals)
    p_emp = np.zeros(individuals)
    spear_emp = np.zeros(individuals)
    sp_emp = np.zeros(individuals)

    pear_sim = np.zeros(individuals)
    p_sim = np.zeros(individuals)
    spear_sim = np.zeros(individuals)
    sp_sim = np.zeros(individuals)

    # Optional: store all SC-FC values per individual
    sc_vals_all = []
    fc_emp_vals_all = []
    fc_sim_vals_all = []

    # Load structural and functional data
    for i in range(individuals):
        SC = dm.load_DTI_data(i)
        fMRI = dm.load_fMRI_data(i)

        num_nodes = SC.shape[0]

        # Time indices
        times = np.arange(0, 4800 * 1.2, 4800)

        # ---------------------------------------------------------------
        # Simulate diffusion model timeseries: shape = (T, nodes)
        # ---------------------------------------------------------------
        FC_sim = dlm.nodewise_diffusion(SC,1, 2)

        # ------------------------------------------------------------------
        # Compute FC: empirical & simulated
        # ------------------------------------------------------------------
        FC_emp = compute_fc(fMRI)

        # ------------------------------------------------------------------
        # Compute SC–FC correlations
        # ------------------------------------------------------------------
        pear_emp[i], p_emp[i], spear_emp[i], sp_emp[i], sc_vals, fc_emp_vals = sc_fc_corr(SC, FC_emp)
        pear_sim[i], p_sim[i], spear_sim[i], sp_sim[i], sc_vals_sim, fc_sim_vals = sc_fc_corr(SC, FC_sim)

        # Store SC-FC values for plotting (optional: concatenate across individuals)
        sc_vals_all.extend(sc_vals)
        fc_emp_vals_all.extend(fc_emp_vals)
        fc_sim_vals_all.extend(fc_sim_vals)

    # Compute average correlations across individuals
    print("\n===== Empirical SC–FC =====")
    print("Pearson:  mean r =", pear_emp.mean(), " mean p =", p_emp.mean())
    print("Spearman: mean r =", spear_emp.mean(), " mean p =", sp_emp.mean())

    print("\n===== Simulated SC–FC =====")
    print("Pearson:  mean r =", pear_sim.mean(), " mean p =", p_sim.mean())
    print("Spearman: mean r =", spear_sim.mean(), " mean p =", sp_sim.mean())

    # ------------------------------------------------------------------
    # Scatterplot: Simulated FC vs Empirical FC
    # ------------------------------------------------------------------
    plt.figure(figsize=(6, 6))

    # Scatter: each point is a connection (across all individuals)
    plt.scatter(fc_sim_vals_all, fc_emp_vals_all, s=3, alpha=0.3, color='blue')

    # Linear fit
    coeff = np.polyfit(fc_sim_vals_all, fc_emp_vals_all, 1)  # y = m*x + b
    fit_line = np.polyval(coeff, fc_sim_vals_all)
    plt.plot(fc_sim_vals_all, fit_line, color='red', linewidth=2, label=f'Slope: {coeff[0]:.8f}')

    plt.title("Diffusion model FC vs Empirical FC ")
    plt.xlabel("Simulated Functional Connectivity")
    plt.ylabel("Empirical Functional Connectivity")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
