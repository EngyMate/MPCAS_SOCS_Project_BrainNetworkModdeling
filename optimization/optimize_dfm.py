import numpy as np

import data_management as dm
import diffusion_model as dlm

if __name__ == "__main__":
    individuals = 100
    DTI_data = []
    fMRI_data = []
    for i in range(individuals):
        DTI_data.append(dm.load_DTI_data(i))
        fMRI_data.append(dm.load_fMRI_data(i))

    beta_range = np.arange(0, 6, 1.0)
    times = np.arange(0, 500, 500/4800)
    best_beta, best_fitness, best_sim = dlm.compute_best_beta_fast(individuals,DTI_data,fMRI_data, beta_range, times)