import math
import time

import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import data_management as dm
import nonLinear_model as nlm
import random
import evaluation_tools as et



# ---------------------------------------------------------
# --- CONTINUE FROM YOUR SCRIPT ---------------------------
# ---------------------------------------------------------

def sim():
    individuals = 100
    DTI_data = []
    fMRI_data = []

    for i in range(individuals):
        DTI_data.append(dm.load_DTI_data(i))  # SC matrix
        fMRI_data.append(dm.load_fMRI_data(i))  # fMRI timeseries

    #best_fitness, best_params = nlm.get_best()

    t_span = (0, 4800 * 1.2+1000)
    t_eval = np.linspace(t_span[0], t_span[1], 4800+1000)

    #simulated files 0 to 51
    files_to_use = range(99)

    for r in files_to_use:
        print(r)
        SC = DTI_data[r]

        num_nodes = SC.shape[0]
        start_time = time.time()

        V, W, Z, t_eval = nlm.simulate_network(
            num_nodes,
            t_span,
            t_eval,
            connectome_matrix=SC,
            noise_level=0.1
        )
        end_time = time.time()
        print(f"simmulation running time: {end_time - start_time}")

        # Choose one simulated variable as simulated fMRI (e.g., V)
        sim_fMRI = V.transpose()  # ensure shape (timepoints, nodes)
        sim_fMRI[-4800:, :].tofile(f"nln_sim_{r}.bin")

def eval():
    individuals = 100
    files_to_use = range(52)
    l = len(files_to_use)
    pear_emp = 0
    p_emp = 0
    spear_emp = 0
    sp_emp = 0

    pear_sim = 0
    p_sim =0
    spear_sim = 0
    sp_sim = 0

    # Optional: store all SC-FC values per individual
    sc_vals_all = np.zeros(246 * 246)
    fc_emp_vals_all = np.zeros(246 * 246)
    fc_sim_vals_all = np.zeros(246 * 246)

    pear_fc = 0
    spear_fc = 0

    for r in files_to_use:
        print(r)
        sim_fMRI = np.fromfile(f"nln_sim_{r}.bin", dtype=float).reshape((4800, 246))

        SC = dm.load_DTI_data(r)
        fMRI = dm.load_fMRI_data(r)
        # -------------------------------
        # Compute FC (empirical + simulated)
        # -------------------------------

        FC_empirical = et.compute_fc(fMRI)
        FC_simulated = et.compute_fc(sim_fMRI)

        # ------------------------------------------------------------------
        # Compute SC–FC correlations
        # ------------------------------------------------------------------
        pr, pb, sr, pb, sc, fc = et.sc_fc_corr(SC, FC_empirical)
        pear_emp += pr
        p_emp+= pb
        spear_emp += sr
        sp_emp += pb

        pr, pb, sr, pb, sc, fc = et.sc_fc_corr(SC, FC_simulated)
        pear_sim += pr
        p_sim += pb
        spear_sim += sr
        sp_sim += pb

        r_p, p_p, r_s, p_s = et.fc_fc_corr(FC_empirical, FC_simulated)

        pear_fc += r_p
        spear_fc += r_s
        # Store SC-FC values for plotting (optional: concatenate across individuals)
        sc_vals_all += SC.flatten()
        fc_emp_vals_all += FC_empirical.flatten()
        fc_sim_vals_all += FC_simulated.flatten()

    sc_vals_all /= l
    fc_emp_vals_all /= l
    fc_sim_vals_all /= l

    pear_emp /= l
    p_emp /= l
    spear_emp /= l
    sp_emp /= l
    pear_sim /= l
    p_sim /= l
    spear_sim /= l
    sp_sim /= l

    pear_fc /= l
    spear_fc /= l

    # Compute average correlations across individuals
    print("\n===== Empirical SC–FC =====")
    print("Pearson:  mean r =", pear_emp, " mean p =", p_emp)
    print("Spearman: mean r =", spear_emp, " mean p =", sp_emp)

    print("\n===== Simulated SC–FC =====")
    print("Pearson:  mean r =", pear_sim, " mean p =", p_sim)
    print("Spearman: mean r =", spear_sim, " mean p =", sp_sim)

    # ------------------------------------------------------------------
    # Scatterplot: Simulated FC vs Empirical FC
    # ------------------------------------------------------------------

    p = pear_fc.mean()
    s = spear_fc.mean()

    plt.figure(figsize=(6, 6))

    mask = (fc_sim_vals_all < 1) & (fc_emp_vals_all < 1)

    plt.scatter(fc_sim_vals_all[mask], fc_emp_vals_all[mask], s=1)
    x = np.array([0, 0.1])
    # line = p * x + 0.4
    # plt.plot(x, line, 'k--', alpha=0.7)
    plt.text(0.05, 0.95, f"Pearson r = {p:.3f}",
             transform=plt.gca().transAxes, va='top')
    plt.text(0.05, 0.90, f"Spearman r = {s:.3f}",
             transform=plt.gca().transAxes, va='top')
    plt.xlabel("Simulated FC")
    plt.ylabel("Empirical FC")
    plt.title("Stochastic linear FC-FC similarity")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    sim()
    """sim_fMRI = np.fromfile(f"nln_sim_{99}.bin", dtype=float).reshape((4800, 246))
    fMRI = dm.load_fMRI_data(99)
    plt.plot(sim_fMRI[:,0])
    plt.plot(fMRI[:,0])
    plt.show()"""
    #eval()
