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

    alpha_range = np.arange(-3, 3, 0.1)
    beta_range = np.arange(0, 6, 0.1)
    std_range = np.arange(0.1, 2, 0.1)
    best_alpha, best_beta, best_std, best_fitness = slm.find_best_alpha_beta(individuals, DTI_data, fMRI_data, alpha_range, beta_range,std_range, n_steps=100)
    slm.get_best()