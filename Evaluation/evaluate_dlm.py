import numpy as np
from matplotlib import pyplot as plt

import data_management as dm
import diffusion_model as dlm

if __name__ == "__main__":
    individuals = 100
    DTI_data = []
    fMRI_data = []
    for i in range(individuals):
        DTI_data.append(dm.load_DTI_data(i))
        fMRI_data.append(dm.load_fMRI_data(i))

    best_fitness, best_beta = dlm.get_best()
    evals, evecs = dlm.prepare_laplacian_eigendecomposition(DTI_data[0])
    times = np.arange(0, 4800)
    u = dlm.nodewise_diffusion_timeseries_fast(evals, evecs, best_beta, times)


    plt.plot(u[-400:, 0])
    plt.plot(fMRI_data[0][-400:, 0])
    plt.show()