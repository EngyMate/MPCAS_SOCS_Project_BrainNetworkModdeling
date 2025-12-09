import numpy as np
from matplotlib import pyplot as plt

import data_management as dm
import nonLinear_model as nlm

if __name__ == "__main__":
    individuals = 100
    DTI_data = []
    fMRI_data = []
    for i in range(individuals):
        DTI_data.append(dm.load_DTI_data(i))
        fMRI_data.append(dm.load_fMRI_data(i))

    best_fitness, best_params = nlm.get_best()

    u, time = nlm.non_linear_bold_z_model(best_params, DTI_data[0], time_span = (0, int(4800*1.2)),time_steps = int(4800*1.2))

    #delta_t = 1.2 to 2 # seconds
    #total 5000 seceonds

    plt.plot(u[0, -400:])
    plt.plot(fMRI_data[0][-400:, 0])
    plt.show()