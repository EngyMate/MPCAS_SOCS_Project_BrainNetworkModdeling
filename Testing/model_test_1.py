import numpy as np
import matplotlib.pyplot as plt

# ===========================================
# 1. Model parameters
# ===========================================

# Ion channel reversal potentials
V_Ca = 120.0
V_Na = 55.0
V_K  = -80.0
V_L  = -60.0

# Conductances
g_Ca = 1.1
g_Na = 6.7
g_K  = 2.0
g_L  = 0.5

# Synaptic gains
a_ee = 0.4     # excitatory → excitatory
a_ie = 0.6     # inhibitory → excitatory

# NMDA factor
r_NMDA = 0.25

# External input
I_delta = 0.4

# Coupling parameter (0 = independent nodes, 1 = fully coupled)
c = 0.2

# ===========================================
# 2. Structural connectivity (example)
# ===========================================
N = 3                       # number of brain regions
C = np.array([
    [0.0, 0.2, 0.1],
    [0.2, 0.0, 0.3],
    [0.1, 0.3, 0.0],
])  # symmetric structural connectivity matrix

# Normalize weights
C = C / C.max()

# ===========================================
# 3. Firing-rate function
# ===========================================
def Q_V(V):
    """Sigmoidal firing rate function."""
    V0 = -40.0
    beta = 0.2
    return 1 / (1 + np.exp(-(V - V0) / beta))

# ===========================================
# 4. Gating-variable dynamics
# ===========================================
def m_inf(V):
    """Fast activation variable (simplified)."""
    return 1 / (1 + np.exp(-(V + 20) / 10))

def tau_m(V):
    """Voltage-dependent time constant."""
    return 5.0

def update_m(m, V):
    return (m_inf(V) - m) / tau_m(V)

# Recovery variable W (slow K activation)
def W_inf(V):
    return 1 / (1 + np.exp(-(V + 40) / 10))

def tau_W(V):
    return 50.0

def update_W(W, V):
    return (W_inf(V) - W) / tau_W(V)

# ===========================================
# 5. Right-hand side of dV/dt
# ===========================================
def dVdt(V, m, W):
    """
    Compute dV/dt for all regions.
    V, m, W are vectors of length N.
    """
    # Local firing
    Q_local = Q_V(V)
    # Long-range structural coupling
    Q_global = C @ Q_local

    # Effective excitation = mixture of local and global
    Exc = (1 - c) * Q_local + c * Q_global

    # Ionic currents
    I_Ca = -(g_Ca + r_NMDA * a_ee * Exc) * m * (V - V_Ca)
    I_Na = -(g_Na * m + a_ee * Exc) * (V - V_Na)
    I_K  = -(g_K * W * (V - V_K))
    I_Lk = -g_L * (V - V_L)

    # Inhibition (local only here)
    I_inh = a_ie * Q_local

    # External input
    I_ext = I_delta

    return I_Ca + I_Na + I_K + I_Lk + I_ext - I_inh

# ===========================================
# 6. RK4 integrator
# ===========================================
def rk4_step(V, m, W, dt):
    k1_V = dVdt(V, m, W)
    k1_m = update_m(m, V)
    k1_W = update_W(W, V)

    k2_V = dVdt(V + dt/2 * k1_V, m + dt/2 * k1_m, W + dt/2 * k1_W)
    k2_m = update_m(m + dt/2 * k1_m, V + dt/2 * k1_V)
    k2_W = update_W(W + dt/2 * k1_W, V + dt/2 * k1_V)

    k3_V = dVdt(V + dt/2 * k2_V, m + dt/2 * k2_m, W + dt/2 * k2_W)
    k3_m = update_m(m + dt/2 * k2_m, V + dt/2 * k2_V)
    k3_W = update_W(W + dt/2 * k2_W, V + dt/2 * k2_V)

    k4_V = dVdt(V + dt * k3_V, m + dt * k3_m, W + dt * k3_W)
    k4_m = update_m(m + dt * k3_m, V + dt * k3_V)
    k4_W = update_W(W + dt * k3_W, V + dt * k3_V)

    V_new = V + dt/6 * (k1_V + 2*k2_V + 2*k3_V + k4_V)
    m_new = m + dt/6 * (k1_m + 2*k2_m + 2*k3_m + k4_m)
    W_new = W + dt/6 * (k1_W + 2*k2_W + 2*k3_W + k4_W)

    return V_new, m_new, W_new

# ===========================================
# 7. Simulation
# ===========================================
T = 5000.0      # total simulation time (ms)
dt = 0.1       # time step (ms)
steps = int(T / dt)

# State variables
V = -65 * np.ones(N)
m = 0.1 * np.ones(N)
W = 0.1 * np.ones(N)

# Record
V_hist = np.zeros((steps, N))

for t in range(steps):
    V, m, W = rk4_step(V, m, W, dt)
    V_hist[t] = V

# ===========================================
# 8. Plot results
# ===========================================
plt.figure(figsize=(10, 5))
for i in range(N):
    plt.plot(V_hist[:, i], label=f'Region {i+1}')
plt.xlabel("Time step")
plt.ylabel("Membrane potential V (mV)")
plt.title("Neural Mass Model Simulation")
plt.legend()
plt.show()
