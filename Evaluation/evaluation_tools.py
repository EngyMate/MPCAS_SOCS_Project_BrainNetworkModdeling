import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, spearmanr

import data_management as dm


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

def fc_fc_corr(FC_emp, FC_sim):
    mask = np.triu(np.ones_like(FC_emp), k=1).astype(bool)

    emp_vals = FC_emp[mask]
    sim_vals = FC_sim[mask]

    # Remove NaNs / infs
    m = np.isfinite(emp_vals) & np.isfinite(sim_vals)
    emp_vals, sim_vals = emp_vals[m], sim_vals[m]

    r_p, p_p = pearsonr(emp_vals, sim_vals)
    r_s, p_s = spearmanr(emp_vals, sim_vals)

    return r_p, p_p, r_s, p_s



# -------------------------------------------------------------
# MAIN SCRIPT
# -------------------------------------------------------------

if __name__ == "__main__":

    individuals = 100

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

        FC_empirical = compute_fc(fMRI)
        # ------------------------------------------------------------------
        # Compute SC–FC correlations
        # ------------------------------------------------------------------
        pear_emp[i], p_emp[i], spear_emp[i], sp_emp[i], sc_vals, fc_emp_vals = sc_fc_corr(SC, FC_empirical)

        # Store SC-FC values for plotting (optional: concatenate across individuals)
        sc_vals_all.extend(sc_vals)
        fc_emp_vals_all.extend(fc_emp_vals)

    # Compute average correlations across individuals
    print("\n===== Empirical SC–FC =====")
    print("Pearson:  mean r =", pear_emp.mean(), " mean p =", p_emp.mean())
    print("Spearman: mean r =", spear_emp.mean(), " mean p =", sp_emp.mean())

    # ------------------------------------------------------------------
    # Scatterplots: SC vs FC with linear fit (all individuals combined)
    # ------------------------------------------------------------------
    plt.figure(figsize=(12, 5))

    plt.scatter(sc_vals_all, fc_emp_vals_all, s=3, alpha=0.3)
    coeff_emp = np.polyfit(sc_vals_all, fc_emp_vals_all, 1)
    fit_emp = np.polyval(coeff_emp, sc_vals_all)
    plt.plot(sc_vals_all, fit_emp, color='red', linewidth=2, label=f'Slope: {coeff_emp[0]:.8f}')
    plt.title("Empirical SC–FC")
    plt.xlabel("Structural Connectivity")
    plt.ylabel("Functional Connectivity")
    plt.xlim(0, 0.8)
    plt.ylim(-0.2, 1)
    plt.show()
