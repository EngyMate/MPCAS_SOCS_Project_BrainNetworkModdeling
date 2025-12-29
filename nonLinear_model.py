import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.stats import gamma
import Evaluation.evaluation_tools as et
import data_management

# Default parameters (TVB-style)
params = dict(
    gCa=1.1, gK=2.0, gL=0.6, gNa=6.7,
    phi=0.7, tau_K=1.0,
    TK=0.0, TCa=-0.01, TNa=0.3,
    d_K=0.3, d_Ca=0.15, d_Na=0.15,
    VCa=1.0, VK=-0.7, VL=-0.5, VNa=0.53,
    aei=2.0, aie=2.0, aee=0.4, ane=1.0, ani=0.4,
    Iext=0.3, b=0.1, C=0.5, rNMDA=0.25,
    VT=0.0, d_V=0.65, ZT=0.0, d_Z=0.7,
    QV_max=1.0, QZ_max=1.0, t_scale=1.2, kz=0.5
)


def init_param_files():
    # Allocate output array
    init_param = [0.0] * len(params)

    # Fill using a for-loop
    for i, key in enumerate(params):
        init_param[i] = params[key]

    np.array([0]).tofile("optimization/nlm_best_fitness.bin")
    np.array(init_param).tofile("optimization/nlm_best_param.bin")

def get_best():
    best_fitness = np.fromfile(
        "C:\\Users\\Johan\\PycharmProjects\\MPCAS_SOCS_Project_BrainNetworkModdeling\\optimization\\nlm_best_fitness.bin",
        dtype=float)[0]

    path = "C:\\Users\\Johan\\PycharmProjects\\MPCAS_SOCS_Project_BrainNetworkModdeling\\optimization\\nlm_best_param.bin"
    best_params_list = np.fromfile(path, dtype=float)

    params_copy = params.copy()

    # Fill using a for-loop
    for i, key in enumerate(params_copy):
        params_copy[key] = best_params_list[i]

    print(f"best_fitness:{best_fitness}, best_alpha:{params_copy}")

    return best_fitness, params_copy

def dfun(state_variables, num_nodes, i, coupling, local_coupling=0.0, p=params):
    V_all = state_variables.reshape((num_nodes, 3))[:, 0]
    W_all = state_variables.reshape((num_nodes, 3))[:, 1]
    Z_all = state_variables.reshape((num_nodes, 3))[:, 2]
    derivative = np.zeros([3])

    W = W_all[i]
    V = V_all[i]
    Z = Z_all[i]

    # Channel activations
    m_Ca = 0.5 * (1 + np.tanh((V - p['TCa']) / p['d_Ca']))
    m_Na = 0.5 * (1 + np.tanh((V - p['TNa']) / p['d_Na']))
    m_K = 0.5 * (1 + np.tanh((V - p['TK']) / p['d_K']))

    # Firing rates
    QV = 0.5 * p['QV_max'] * (1 + np.tanh((V - p['VT']) / p['d_V']))
    QZ = 0.5 * p['QZ_max'] * (1 + np.tanh((Z - p['ZT']) / p['d_Z']))

    QV_global = 0.5 * params['QV_max'] * (1 + np.tanh((V_all - params['VT']) / params['d_V']))
    c_0 = coupling @ QV_global
    lc_0 = local_coupling * QV

    # Voltage derivative
    derivative[0] = p['t_scale'] * (
            - (p['gCa'] + (1.0 - p['C']) * (p['rNMDA'] * p['aee']) * (QV + lc_0) + p['C'] * p['rNMDA'] * p['aee'] * c_0) * m_Ca * (V - p['VCa'])
            - p['gK'] * W * (V - p['VK'])
            - p['gL'] * (V - p['VL'])
            - (p['gNa'] * m_Na + (1.0 - p['C']) * p['aee'] * (QV + lc_0) + p['C'] * p['aee'] * c_0) * (V - p['VNa'])
            - p['aie'] * Z * QZ
            + p['ane'] * p['Iext']
    )

    # K gating variable
    derivative[1] = p['t_scale'] * p['phi'] * (m_K - W) / p['tau_K']

    # Inhibitory population
    derivative[2] = p['t_scale'] * p['b'] * (p['ani'] * p['Iext'] + p['aei'] * V * QV)

    return derivative


def simulate_network(num_nodes, t_span, t_eval, parameters, connectome_matrix=None, noise_level=0.1):
    if connectome_matrix is None:
        connectome_matrix = np.eye(num_nodes)

    # Generate per-node noisy parameters
    params_per_node = generate_noisy_params_per_node(parameters, num_nodes, noise_level)


    initial_state = np.zeros((num_nodes, 3))
    initial_state[:, 0] = 0.1
    initial_state = initial_state.flatten()

    sol = solve_ivp(
        lambda t, y: np.vstack(
            [dfun(y, num_nodes, i,connectome_matrix[i, :], p=params_per_node[i]) for i in range(num_nodes)]
        ).flatten(),
        t_span,
        initial_state,
        t_eval=t_eval,
        vectorized=False
    )

    V = sol.y.reshape((num_nodes, 3, -1))[:, 0, :]
    W = sol.y.reshape((num_nodes, 3, -1))[:, 1, :]
    Z = sol.y.reshape((num_nodes, 3, -1))[:, 2, :]
    return np.array(V), np.array(W), np.array(Z), np.array(t_eval)



def plot_voltage(V, t_eval):
    plt.figure(figsize=(12, 6))
    for i in range(V.shape[0]):
        plt.plot(t_eval, V[i], label=f'Neuron {i + 1}')
    plt.xlabel('Time')
    plt.ylabel('Membrane Potential V')
    plt.title('Coupled Neurons')
    plt.legend()
    plt.show()


def generate_noisy_params_per_node(params, num_nodes, noise_level=0.1):
    """
    Returns a list of parameter dictionaries, one for each node,
    with each numerical parameter perturbed ±noise_level.
    """
    noisy_params_list = []
    for _ in range(num_nodes):
        noisy_params = {}
        for key, value in params.items():
            if isinstance(value, (int, float)):
                factor = np.random.lognormal(mean=0, sigma=noise_level)
                noisy_params[key] = value * factor
            else:
                noisy_params[key] = value
        noisy_params_list.append(noisy_params)
    return noisy_params_list

def compute_QV(V, params):
    return 0.5 * params['QV_max'] * (1 + np.tanh((V - params['VT']) / params['d_V']))

def canonical_hrf(t, peak1=6, peak2=16, ratio=1/6, scale=1):
    hrf = gamma.pdf(t, peak1) - ratio * gamma.pdf(t, peak2)
    return scale * hrf / np.max(hrf)

def neural_to_bold(neural_signal, hrf):
    bold = np.array([
        np.convolve(neural_signal[i], hrf, mode="full")[:neural_signal.shape[1]]
        for i in range(neural_signal.shape[0])
    ])
    return bold

def bold_simulate(SC,parameters, time_max = 4800, initial = 100, noise_level= 0.1):
    t_span = (0, time_max * 1.2 + initial* 1.2)
    t_eval = np.linspace(t_span[0], t_span[1], time_max + initial)
    num_nodes = SC.shape[0]
    V, W, Z, t_eval = simulate_network(num_nodes, t_span, t_eval,parameters, connectome_matrix=SC, noise_level=noise_level)
    QV = compute_QV(V, params)  # shape: (nodes, time)

    dt = t_eval[1] - t_eval[0]
    hrf_t = np.arange(0, 32, dt)
    hrf = canonical_hrf(hrf_t)

    BOLD = neural_to_bold(QV, hrf)
    BOLD = (BOLD - BOLD.mean(axis=1, keepdims=True)) / BOLD.std(axis=1, keepdims=True)
    BOLD = BOLD.T

    t_eval = t_eval[initial:]
    last_BOLD = BOLD[initial:, :]
    return last_BOLD, t_eval

if __name__ == "__main__":
    #get_best()
    #init_param_files()
    """for key in params:
        mod_param = params.copy()
        mod_param[key] -= 0.0"""

    for i in range(5):
        C = data_management.load_DTI_data(0)
        act = data_management.load_fMRI_data(0)

        last_BOLD, t_eval = bold_simulate(C, params, time_max = 1000, initial = 100, noise_level= 0.01)
        empirical = act[-1000:, :]
        FC_empirical = et.compute_fc(empirical)
        FC_simulated = et.compute_fc(last_BOLD)
        r_p, p_p, r_s, p_s = et.fc_fc_corr(FC_empirical, FC_simulated)

        print(f"No change: Pearson: {r_p} pp:{p_p} ")
        #print(f"{key}:{mod_param[key]}: Pearson: {r_p} pp:{p_p} ")

        """plt.figure(figsize=(12, 6))
        plt.plot(t_eval, last_BOLD[:, 0], label=f'Sim')
        plt.plot(t_eval, empirical[:, 0], label=f'actual')
        plt.xlabel('Time')
        plt.ylabel('BOLD Activity')
        plt.title('Coupled Neurons')
        plt.legend()
        plt.show()

        plt.figure(figsize=(12, 6))
        for i in range(last_BOLD.shape[1]):
            plt.plot(t_eval, last_BOLD[:, i], label=f'Neuron {i + 1}')
        plt.xlabel('Time')
        plt.ylabel('Membrane Potential V')
        plt.title('Coupled Neurons')
        plt.legend()
        plt.show()"""

"""
Key references

Góni et al. (2014) — “Resting-brain functional connectivity predicted by analytic measures of network communication”

This paper shows that shortest paths and related network communication measures derived from structural connectivity can predict resting-state functional connectivity patterns.

It specifically focuses on path length and “search information” (a shortest-path embedded measure) and demonstrates that these structural metrics explain significant FC variance. 
PubMed

Meier et al. (2021) — “Predicting MEG resting-state functional connectivity from microstructural information”

This study directly uses shortest path length based on SC to predict FC components from MEG data and finds that shortest-path algorithms have good predictive accuracy. 
PubMed

Other related work

Chen et al. (2011) — “Negative functional connectivity and its dependence on the shortest path length…”

Shows empirically that shortest path distance in SC correlates with certain FC patterns (including negative FC), supporting the idea that indirect, shortest routes in the anatomy influence functional coupling. 
PubMed

Barton et al. / Drosophila connectome work — “The connectome predicts resting-state functional connectivity…”

In a non-human system, this paper demonstrates that shortest path distances in the structural network help explain functional connectivity beyond direct connections"""


#https://docs.thevirtualbrain.org/_modules/tvb/simulator/models/larter_breakspear.html