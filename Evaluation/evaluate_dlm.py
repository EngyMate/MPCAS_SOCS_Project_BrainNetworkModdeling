from os.path import split

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, spearmanr

import data_management as dm
import diffusion_model as dlm
import evaluation_tools as et

# -------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------



# -------------------------------------------------------------
# MAIN SCRIPT
# -------------------------------------------------------------

if __name__ == "__main__":
    BNA_atlas = dm.load_mixed_excel(dm.PATH_excel_BNA_atlas)
    node_labels = [a[0].split("_")[0] for a in BNA_atlas]

    label_to_color = {
        'SFG': '#1f77b4',
        'MFG': '#aec7e8',
        'IFG': '#ff7f0e',
        'OrG': '#ffbb78',
        'PrG': '#2ca02c',
        'PCL': '#98df8a',
        'STG': '#d62728',
        'MTG': '#ff9896',
        'ITG': '#9467bd',
        'FuG': '#c5b0d5',
        'PhG': '#8c564b',
        'pSTS': '#c49c94',
        'SPL': '#e377c2',
        'IPL': '#f7b6d2',
        'PCun': '#7f7f7f',
        'PoG': '#c7c7c7',
        'INS': '#bcbd22',
        'CG': '#dbdb8d',
        'MVOcC': '#17becf',
        'LOcC': '#9edae5',
        'Amyg': '#393b79',
        'Hipp': '#637939',
        'BG': '#8c6d31',
        'Tha': '#843c39'
    }

    colors = [label_to_color[l] for l in node_labels]
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
    sc_vals_all = np.zeros(246*246)
    fc_emp_vals_all = np.zeros(246*246)
    fc_sim_vals_all = np.zeros(246*246)

    pear_fc = np.zeros(individuals)
    spear_fc = np.zeros(individuals)
    # Load structural and functional data
    for i in range(individuals):
        print(i)
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
        FC_emp = et.compute_fc(fMRI)

        # ------------------------------------------------------------------
        # Compute SC–FC correlations
        # ------------------------------------------------------------------
        pear_emp[i], p_emp[i], spear_emp[i], sp_emp[i], sc_vals, fc_emp_vals = et.sc_fc_corr(SC, FC_emp)
        pear_sim[i], p_sim[i], spear_sim[i], sp_sim[i], sc_vals_sim, fc_sim_vals = et.sc_fc_corr(SC, FC_sim)

        r_p, p_p, r_s, p_s = et.fc_fc_corr(FC_emp, FC_sim)

        pear_fc[i] = r_p
        spear_fc[i] = r_s

        # Store SC-FC values for plotting (optional: concatenate across individuals)
        sc_vals_all += SC.flatten()
        fc_emp_vals_all += FC_emp.flatten()
        fc_sim_vals_all += FC_sim.flatten()

    sc_vals_all /= individuals
    fc_emp_vals_all /= individuals
    fc_sim_vals_all /= individuals

    # Compute average correlations across individuals
    print("\n===== Empirical SC–FC =====")
    print("Pearson:  mean r =", pear_emp.mean(), " mean p =", p_emp.mean())
    print("Spearman: mean r =", spear_emp.mean(), " mean p =", sp_emp.mean())

    print("\n===== Simulated SC–FC =====")
    print("Pearson:  mean r =", pear_sim.mean(), " mean p =", p_sim.mean())
    print("Spearman: mean r =", spear_sim.mean(), " mean p =", sp_sim.mean())

    print("\n===== SC–SC =====")
    print("Pearson:  mean r =", pear_fc.mean(), " mean p =", p_sim.mean())
    print("Spearman: mean r =", spear_fc.mean(), " mean p =", sp_sim.mean())

    # ------------------------------------------------------------------
    # Scatterplot: Simulated FC vs Empirical FC
    # ------------------------------------------------------------------

    p = pear_fc.mean()
    s = spear_fc.mean()

    plt.figure(figsize=(6, 6))
    mask = (fc_sim_vals_all < 1) & (fc_emp_vals_all < 1)

    plt.scatter(fc_sim_vals_all[mask], fc_emp_vals_all[mask], s=1)
    x = np.array([0, 0.1])
    #line = p * x + 0.4
    #plt.plot(x, line, 'k--', alpha=0.7)
    plt.text(0.05, 0.95, f"Pearson r = {p:.3f}",
             transform=plt.gca().transAxes, va='top')
    plt.text(0.05, 0.90, f"Spearman r = {s:.3f}",
             transform=plt.gca().transAxes, va='top')
    plt.xlabel("Simulated FC")
    plt.ylabel("Empirical FC")
    plt.title("Diffusion FC-FC similarity")
    plt.tight_layout()
    plt.show()
