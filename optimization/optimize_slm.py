import numpy as np
import data_management as dm
import stochastic_linear_model as slm

if __name__ == "__main__":
    individuals = 100
    DTI_data = []
    fMRI_data = []
    for i in range(individuals):
        DTI_data.append(dm.load_DTI_data(i))
        fMRI_data.append(dm.load_fMRI_data(i))

    alpha_range = np.arange(-3, 3, 1)
    beta_range = np.arange(0, 6, 1)
    best_alpha, best_beta, best_fitness = slm.find_best_alpha_beta(individuals, DTI_data, fMRI_data, alpha_range, beta_range, n_steps=4800)
    slm.get_best()
    slm.get_best()