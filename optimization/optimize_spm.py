import numpy as np
import data_management as dm
import shortest_path as spm

if __name__ == "__main__":
    individuals = 100
    DTI_data = []
    fMRI_data = []
    for i in range(individuals):
        DTI_data.append(dm.load_DTI_data(i))
        fMRI_data.append(dm.load_fMRI_data(i))

    alpha_range = np.arange(0.1, 2, 0.1)
    best_alpha, best_fitness = spm.find_best_alpha_beta(individuals, DTI_data, fMRI_data, alpha_range)
    spm.get_best()