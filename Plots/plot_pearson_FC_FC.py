import os

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import ttest_rel


def plot_series_with_mean(x, y, label):
    # Faint line
    line, = plt.plot(
        x, np.abs(y),
        linestyle='-',
        alpha=0.9,
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
    """mean_val = np.mean(np.abs(y))
    plt.hlines(
        mean_val,
        x.min(),
        x.max(),
        colors=line.get_color(),
        linestyles='-'
    )"""

def compute_corr_diff_for(tag):
    slm_pearson_ = np.fromfile(f"Evaluation/Pearson_values/{tag.lower()}_fc_.bin", dtype=float)
    slm_pearson_SPI = np.fromfile(f"Evaluation/Pearson_values/{tag.lower()}_fc_SPI.bin", dtype=float)

    if tag.upper() == "NLM":
        tag = "NLNMM"

    plot_series_with_mean(x, slm_pearson_, f"{tag.upper()}")
    plot_series_with_mean(x, slm_pearson_SPI, f"SPI {tag.upper()}")
    diff = slm_pearson_ - slm_pearson_SPI
    t_stat_uncorrected, p_value_uncorrected = ttest_rel(slm_pearson_, slm_pearson_SPI)
    print(f"""
    --- {tag.upper()} Summary ---
    Mean Difference:          {diff.mean():.4f}
    t-statistic (uncorrected):{t_stat_uncorrected:.4f}
    p-value (uncorrected):    {p_value_uncorrected:.2e}
    Standard Deviations:
      Series 1 (Pearson):     {slm_pearson_.std():.4f}
      Series 2 (SPI):         {slm_pearson_SPI.std():.4f}
      Difference (mean_std):  {diff.std():.4f}
    """)
    plt.xlabel("$I[n]$ Individual")
    plt.ylabel("ρ pearson correlation coefficient")
    plt.title(f"{tag.upper()} Pearson correlation difference")
    plt.legend()
    plt.savefig(f"Figures/correlation_diff_{tag.upper()}.png")
    plt.close()

if __name__ == "__main__":
    x = np.arange(100)

    os.chdir("C:/Users/Johan/PycharmProjects/MPCAS_SOCS_Project_BrainNetworkModdeling")
    compute_corr_diff_for("SLM")
    compute_corr_diff_for("DLM")
    compute_corr_diff_for("NLM")
    compute_corr_diff_for("RWM")
    compute_corr_diff_for("SPM")


