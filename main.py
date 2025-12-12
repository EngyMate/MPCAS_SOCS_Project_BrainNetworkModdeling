from matplotlib import pyplot as plt

import test
import data_management as dm

if __name__ == "__main__":
    print(dm.load_DTI_data(0))
    print(dm.load_DTI_data(1))
    print("dasdasd")
    print(dm.load_fMRI_data(0))
    print(dm.load_fMRI_data(1))
    plt.plot([1, 2])
    plt.title(f"$E_{{initial}}$")
    plt.show()