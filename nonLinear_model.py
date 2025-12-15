import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.stats import gamma

import data_management

# Default parameters (TVB-style)
params = dict(
    gCa=1.1, gK=2.0, gL=0.5, gNa=6.7,
    phi=0.7, tau_K=1.0,
    TK=0.0, TCa=-0.01, TNa=0.3,
    d_K=0.3, d_Ca=0.15, d_Na=0.15,
    VCa=1.0, VK=-0.7, VL=-0.5, VNa=0.53,
    aei=2.0, aie=2.0, aee=0.4, ane=1.0, ani=0.4,
    Iext=0.3, b=0.1, C=0.1, rNMDA=0.25,
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
    derivative[2] = p['t_scale'] * p['b'] * (p['ani'] * p['Iext'] + p['aei'] * V * QV -p["kz"]*Z)

    if V < -10:
        print(f"V:{V}, W:{W}, Z:{Z}, dV:{derivative[0]}, DW{derivative[1]}, DZ{derivative[2]}")

    return derivative


def simulate_network(num_nodes, t_span, t_eval, connectome_matrix=None, noise_level=0.1):
    if connectome_matrix is None:
        connectome_matrix = np.eye(num_nodes)

    # Generate per-node noisy parameters
    params_per_node = generate_noisy_params_per_node(params, num_nodes, noise_level)


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
                factor = 1 + np.random.uniform(-noise_level, noise_level)
                noisy_params[key] = value * factor
            else:
                noisy_params[key] = value
        noisy_params_list.append(noisy_params)
    return noisy_params_list



if __name__ == "__main__":
    C = data_management.load_DTI_data(0)
    act = data_management.load_fMRI_data(0)

    t_span=(0, 4800 * 1.2)
    t_eval = np.linspace(t_span[1] - 500, t_span[1], 1000)
    num_nodes = C.shape[0]
    V, W, Z, t_eval = simulate_network(num_nodes, t_span, t_eval, connectome_matrix=C, noise_level=0.1)
    plot_voltage(V, t_eval)


#https://docs.thevirtualbrain.org/_modules/tvb/simulator/models/larter_breakspear.html