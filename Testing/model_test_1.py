import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma

# Time vector
TR = 1.0  # seconds
t = np.arange(0, 100, TR)

# Neural input: delta spike at t=20
V_i = np.zeros_like(t)
V_i[20] = 1.0  # true delta spike

# HRF (SPM-like double gamma)
def spm_hrf(tr, length=32):
    dt = tr
    time = np.arange(0, length, dt)
    # parameters for the canonical HRF
    peak1 = gamma.pdf(time, 6)
    peak2 = gamma.pdf(time, 16)
    hrf = peak1 - 0.35 * peak2
    hrf /= np.sum(hrf)  # normalize area
    return hrf

hrf = spm_hrf(TR)
print(hrf)
BOLD = np.convolve(V_i, hrf)[:len(V_i)]

plt.figure(figsize=(10,4))
plt.plot(t, BOLD)
plt.title('Simulated BOLD signal from delta neural spike')
plt.xlabel('Time (s)')
plt.ylabel('BOLD')
plt.show()
