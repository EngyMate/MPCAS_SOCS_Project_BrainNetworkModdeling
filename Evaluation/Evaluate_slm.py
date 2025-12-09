import numpy as np
from matplotlib import pyplot as plt

import data_management as dm
import stochastic_linear_model as slm

if __name__ == "__main__":
    individuals = 100
    DTI_data = []
    fMRI_data = []
    for i in range(individuals):
        DTI_data.append(dm.load_DTI_data(i))
        fMRI_data.append(dm.load_fMRI_data(i))

    best_fitness, best_alpha, best_beta = slm.get_best()

    u = slm.simulate_linear_model(DTI_data[0],best_alpha, best_beta)


    plt.plot(u[-400:, 0])
    plt.plot(fMRI_data[0][-400:, 0])
    plt.show()