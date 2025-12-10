import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import data_management as dm
import nonLinear_model as nlm
import random

# ---------------------------------------------------------
# --- FUNCTIONAL CONNECTIVITY COMPUTATION -----------------
# ---------------------------------------------------------

def compute_fc(timeseries):
    """
    timeseries shape = (timepoints, nodes)
    returns FC matrix (nodes x nodes)
    """
    # z-score for safety
    ts = (timeseries - timeseries.mean(axis=0)) / timeseries.std(axis=0)
    FC = np.corrcoef(ts, rowvar=False)
    return FC


def sc_fc_correlation(SC, FC):
    mask = np.triu(np.ones(SC.shape), k=1).astype(bool)
    sc_vals = SC[mask]
    fc_vals = FC[mask]

    # Fisher z-transform (optional)
    # fc_vals = np.arctanh(fc_vals)

    pear_r, pear_p = pearsonr(sc_vals, fc_vals)
    spear_r, spear_p = spearmanr(sc_vals, fc_vals)

    return pear_r, pear_p, spear_r, spear_p, sc_vals, fc_vals


# ---------------------------------------------------------
# --- CONTINUE FROM YOUR SCRIPT ---------------------------
# ---------------------------------------------------------

if __name__ == "__main__":
    individuals = 100
    DTI_data = []
    fMRI_data = []

    for i in range(individuals):
        DTI_data.append(dm.load_DTI_data(i))     # SC matrix
        fMRI_data.append(dm.load_fMRI_data(i))   # fMRI timeseries

    best_fitness, best_params = nlm.get_best()

    t_span = (0, 4800 * 1.2)
    t_eval = np.linspace(t_span[1] - 1000*1.2, t_span[1], 1000)

    r = random.randint(0, 99)
    SC = DTI_data[r]
    fMRI = fMRI_data[r][-1000:,:]   # shape (4800, 246)

    num_nodes = SC.shape[0]

    V, W, Z, t_eval = nlm.simulate_network(
        num_nodes,
        t_span,
        t_eval,
        connectome_matrix=SC,
        noise_level=0.1
    )

    # Choose one simulated variable as simulated fMRI (e.g., V)
    sim_fMRI = V.T   # ensure shape (timepoints, nodes)

    # -------------------------------
    # Compute FC (empirical + simulated)
    # -------------------------------

    FC_empirical = compute_fc(fMRI)
    FC_simulated = compute_fc(sim_fMRI)

    # -------------------------------
    # Compute SC–FC correlation
    # -------------------------------

    pear_r_emp, pear_p_emp, spear_r_emp, spear_p_emp, sc_vals, fc_emp_vals = sc_fc_correlation(SC, FC_empirical)
    pear_r_sim, pear_p_sim, spear_r_sim, spear_p_sim, _, fc_sim_vals = sc_fc_correlation(SC, FC_simulated)

    print("\n===== Empirical SC–FC =====")
    print("Pearson r =", pear_r_emp, "   p =", pear_p_emp)
    print("Spearman r =", spear_r_emp, "  p =", spear_p_emp)

    print("\n===== Simulated SC–FC =====")
    print("Pearson r =", pear_r_sim, "   p =", pear_p_sim)
    print("Spearman r =", spear_r_sim, "  p =", spear_p_sim)

    # ------------------------------------------------------------------
    # Scatterplot: Simulated FC vs Empirical FC
    # ------------------------------------------------------------------
    plt.figure(figsize=(6, 6))

    # x = simulated FC, y = empirical FC
    plt.scatter(fc_sim_vals, fc_emp_vals, s=3, alpha=0.3, color='blue')

    # Linear fit
    coeff = np.polyfit(fc_sim_vals, fc_emp_vals, 1)  # y = m*x + b
    fit_line = np.polyval(coeff, fc_sim_vals)
    plt.plot(fc_sim_vals, fit_line, color='red', linewidth=2, label=f'Slope: {coeff[0]:.8f}')

    plt.title("Simulated FC vs Empirical FC")
    plt.xlabel("Simulated Functional Connectivity")
    plt.ylabel("Empirical Functional Connectivity")
    plt.xlim(-0.2, 1)
    plt.ylim(-0.2, 1)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()