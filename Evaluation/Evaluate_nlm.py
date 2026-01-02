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
MODIFIED = True

RUN_SIMULATION = False
SHOW_TIME_SERIES = False
EVALUATE = True
PLOT = True


def sim():
    if MODIFIED:
        individuals = 100
    else:
        individuals = 100

    DTI_data = []
    fMRI_data = []

    for i in range(individuals):
        if not MODIFIED:
            DTI_data.append(dm.load_DTI_data(i))  # SC matrix
        else:
            DTI_data.append(dm.load_mDTI_data(i))  # SC matrix
        fMRI_data.append(dm.load_fMRI_data(i))  # fMRI timeseries

    files_to_use = range(individuals)

    for r in files_to_use:
        print(r)
        SC = DTI_data[r]

        num_nodes = SC.shape[0]
        start_time = time.time()

        last_BOLD, t_eval = nlm.bold_simulate(SC, nlm.params, time_max = 4800, initial = 100, noise_level= 0.01)
        end_time = time.time()
        print(f"simmulation running time: {end_time - start_time}")

        # Choose one simulated variable as simulated fMRI (e.g., V)
        if not MODIFIED:
            last_BOLD[-4800:, :].tofile(f"nln_sim_{r}.bin")
        else:
            last_BOLD[-4800:, :].tofile(f"modified_nln_sim_{r}.bin")

def eval():
    if MODIFIED:
        individuals = 100
    else:
        individuals = 100

    m = ""
    if MODIFIED:
        m = "SPI"

    files_to_use = range(individuals)
    l = len(files_to_use)
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
    sc_vals_all = np.zeros(246 * 246)
    fc_emp_vals_all = np.zeros(246 * 246)
    fc_sim_vals_all = np.zeros(246 * 246)
    person_arr = np.zeros(individuals)

    pear_fc = np.zeros(individuals)
    spear_fc = np.zeros(individuals)

    for r in files_to_use:
        print(r)
        if not MODIFIED:
            sim_fMRI = np.fromfile(f"act_sim_nlm/nln_sim_{r}.bin", dtype=float).reshape((4800, 246))
            SC = dm.load_DTI_data(r)
        else:
            sim_fMRI = np.fromfile(f"sip_sim_nlm/modified_nln_sim_{r}.bin", dtype=float).reshape((4800, 246))
            SC = dm.load_mDTI_data(r)


        fMRI = dm.load_fMRI_data(r)
        # -------------------------------
        # Compute FC (empirical + simulated)
        # -------------------------------

        FC_emp = et.compute_fc(fMRI)
        FC_sim = et.compute_fc(sim_fMRI)

        # ------------------------------------------------------------------
        # Compute SC–FC correlations
        # ------------------------------------------------------------------
        pear_emp[r], p_emp[r], spear_emp[r], sp_emp[r], sc_vals, fc_emp_vals = et.sc_fc_corr(dm.load_DTI_data(r),FC_emp)
        pear_sim[r], p_sim[r], spear_sim[r], sp_sim[r], sc_vals_sim, fc_sim_vals = et.sc_fc_corr(SC, FC_sim)

        r_p, p_p, r_s, p_s = et.fc_fc_corr(FC_emp, FC_sim)

        pear_fc[r] = r_p
        spear_fc[r] = r_s
        # Store SC-FC values for plotting (optional: concatenate across individuals)
        sc_vals_all += SC.flatten()
        fc_emp_vals_all += FC_emp.flatten()
        fc_sim_vals_all += FC_sim.flatten()

    sc_vals_all /= individuals
    fc_emp_vals_all /= individuals
    fc_sim_vals_all /= individuals

    pear_emp.tofile(f"Pearson_values/emp.bin")
    pear_fc.tofile(f"Pearson_values/nlm_fc_{m}.bin")
    pear_sim.tofile(f"Pearson_values/nlm_sim_{m}.bin")

    spear_emp.tofile(f"Spearman_values/emp.bin")
    spear_fc.tofile(f"Spearman_values/nlm_fc_{m}.bin")
    spear_sim.tofile(f"Spearman_values/nlm_sim_{m}.bin")

    # Compute average correlations across individuals
    print("\n===== Empirical SC–FC =====")
    print("Pearson:  mean r =", pear_emp.mean(), " mean p =", p_emp.mean())
    print("Spearman_values: mean r =", spear_emp.mean(), " mean p =", sp_emp.mean())

    print("\n===== Simulated SC–FC =====")
    print("Pearson:  mean r =", pear_sim.mean(), " mean p =", p_sim.mean())
    print("Spearman_values: mean r =", spear_sim.mean(), " mean p =", sp_sim.mean())

    print("\n===== SC–SC =====")
    print("Pearson:  mean r =", pear_fc.mean(), " mean p =", p_sim.mean())
    print("Spearman_values: mean r =", spear_fc.mean(), " mean p =", sp_sim.mean())
    # ------------------------------------------------------------------
    # Scatterplot: Simulated FC vs Empirical FC
    # ------------------------------------------------------------------

    p = pear_fc.mean()
    s = spear_fc.mean()


    if PLOT:
        plt.figure(figsize=(6, 6))

        mask = (fc_sim_vals_all < 1) & (fc_emp_vals_all < 1)

        plt.scatter(fc_sim_vals_all[mask], fc_emp_vals_all[mask], s=1)
        x = np.array([0, 0.1])
        # line = p * x + 0.4
        # plt.plot(x, line, 'k--', alpha=0.7)
        plt.text(0.05, 0.95, f"Pearson r = {p:.5f}",
                 transform=plt.gca().transAxes, va='top')
        plt.text(0.05, 0.90, f"Spearman_values r = {s:.5f}",
                 transform=plt.gca().transAxes, va='top')
        plt.xlabel("Simulated FC")
        plt.ylabel("Empirical FC")
        plt.title(f"{m} NLNMM and Empirical FC-FC correlation")
        plt.tight_layout()

        plt.savefig(
            f"C:/Users/Johan/PycharmProjects/MPCAS_SOCS_Project_BrainNetworkModdeling/Figures/{m}nonLinear_FC_FC.png")
        # plt.show()


if __name__ == "__main__":
    if RUN_SIMULATION:
        #29 ? todo = error?
        sim()
    if SHOW_TIME_SERIES:
        sim_fMRI = np.fromfile(f"nln_sim_{99}.bin", dtype=float).reshape((4800, 246))
        fMRI = dm.load_fMRI_data(99)
        plt.plot(sim_fMRI[:,0])
        plt.plot(fMRI[:,0])
        plt.show()
    if EVALUATE:
        eval()

