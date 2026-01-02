import os

import numpy as np
from matplotlib import pyplot as plt

def plot_series_with_mean(x, y, label):
    # Faint line
    line, = plt.plot(
        x, np.abs(y),
        linestyle='--',
        alpha=0.5,
        label = label
    )

    """# Points
    plt.plot(
        x, np.abs(y),
        marker='o',
        linestyle='None',
        color=line.get_color(),
    )"""

    # Mean line
    mean_val = np.mean(np.abs(y))
    plt.hlines(
        mean_val,
        x.min(),
        x.max(),
        colors=line.get_color(),
        linestyles='-'
    )

    return mean_val, line.get_color()

if __name__ == "__main__":
    x = np.arange(100)
    os.chdir("C:/Users/Johan/PycharmProjects/MPCAS_SOCS_Project_BrainNetworkModdeling")
    #mean_val_1, colour1 = plot_series_with_mean(x, stochastic_pearson_,"SLM")
    #mean_val_2, colour2 = plot_series_with_mean(x, stochastic_pearson_SPI, "SPI SLM")

    diff1 = 1#abs(mean_val_1- mean_val_2)

    t = "DLM"
    diffusion_pearson_ = np.fromfile("Evaluation/Pearson_values/dlm_fc_.bin", dtype=float)
    diffusion_pearson_SPI = np.fromfile("Evaluation/Pearson_values/dlm_fc_SPI.bin", dtype=float)
    mean_val_3, colour3 = plot_series_with_mean(x, diffusion_pearson_, "DLM")
    mean_val_4, colour4 = plot_series_with_mean(x, diffusion_pearson_SPI, "SPI DLM")
    diff = mean_val_3 - mean_val_4
    print(f"{t}={diff}")

    plt.xlabel("$I[n]$ Individual")
    plt.ylabel("ρ pearson correlation coefficient")
    plt.title("Pearson correlation difference")
    plt.legend()
    plt.savefig(f"Figures/correlation_diff_{t}.png")

    t = "NLNMM"
    diff2 = mean_val_3 - mean_val_4
    nonLinear_pearson_ = np.fromfile("Evaluation/Pearson_values/nlm_fc_.bin", dtype=float)
    nonLinear_pearson_SPI = np.fromfile("Evaluation/Pearson_values/nlm_fc_SPI.bin", dtype=float)
    mean_val_5, colour5 = plot_series_with_mean(x, nonLinear_pearson_, "NLNMM")
    mean_val_6, colour6 = plot_series_with_mean(x, nonLinear_pearson_SPI, "SPI NLNMM")

    diff = mean_val_3 - mean_val_4
    print(f"{t}={diff}")

    plt.xlabel("$I[n]$ Individual")
    plt.ylabel("ρ pearson correlation coefficient")
    plt.title("Pearson correlation difference")
    plt.legend()
    plt.savefig(f"Figures/correlation_diff_{t}.png")


    t = "RWM"
    rwm_pearson_ = np.fromfile("Evaluation/Pearson_values/rwm_fc_.bin", dtype=float)
    rwm_pearson_SPI = np.fromfile("Evaluation/Pearson_values/rwm_fc_SPI.bin", dtype=float)
    mean_val_7, colour7 = plot_series_with_mean(x, rwm_pearson_, "RWM")
    mean_val_8, colour8 = plot_series_with_mean(x, rwm_pearson_SPI, "SPI RWM")

    diff = mean_val_3 - mean_val_4
    print(f"{t}={diff}")

    plt.xlabel("$I[n]$ Individual")
    plt.ylabel("ρ pearson correlation coefficient")
    plt.title("Pearson correlation difference")
    plt.legend()
    plt.savefig(f"Figures/correlation_diff_{t}.png")

    t = "SPM"
    spm_pearson_ = np.fromfile("Evaluation/Pearson_values/spm_fc_.bin", dtype=float)
    spm_pearson_SPI = np.fromfile("Evaluation/Pearson_values/spm_fc_SPI.bin", dtype=float)
    mean_val_9, colour9 = plot_series_with_mean(x, spm_pearson_, "SPM")
    mean_val_10, colour10 = plot_series_with_mean(x, spm_pearson_SPI, "SPI SPM")

    diff = mean_val_3 - mean_val_4
    print(f"{t}={diff}")

    plt.xlabel("$I[n]$ Individual")
    plt.ylabel("ρ pearson correlation coefficient")
    plt.title("Pearson correlation difference")
    plt.legend()
    plt.savefig(f"Figures/correlation_diff_{t}.png")

    """text = (
        f"$\\Delta SLM={diff1:.5f}$\n"
        f"$\\Delta DLM={diff2:.5f}$\n"
        f"$\\Delta NLNMM={diff3:.5f}$\n"
        f"$\\Delta RWM={diff4:.5f}$\n"
        f"$\\Delta SPM={diff5:.5f}$"
    )
    print(text)"""

    """plt.text(
        0.99, 0.50,
        text,
        transform=plt.gca().transAxes,  # use current axes
        fontsize=15,
        verticalalignment='top',
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            edgecolor="blue"
        )
    )"""


