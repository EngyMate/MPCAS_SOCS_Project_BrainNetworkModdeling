


import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.signal import convolve
from scipy.stats import gamma
import data_management

params = dict(
    g_Ca_max_Ca_conductance=1.2,
    g_Na_max_Na_conductance=1.0,
    g_K_max_K_conductance=1.8,
    g_L_leak_conductance=0.5,

    V_Ca_Nernst_potential_Ca=0.0,    # FIX
    V_Na_Nernst_potential_Na=50.0,   # FIX
    V_K_Nernst_potential_K=-77.0,    # FIX
    V_L_leak_reversal_potential=-10.0,

    V_T_trigger_value=-30,

    a_ee_excitatory_to_excitatory=0.1,
    a_ie_inhibitory_to_excitatory=1.0,
    a_ei_excitatory_to_inhibitory=0.1,

    r_NMDA_ratio_NMDA_to_AMPA=0.3,

    Q_max_excitatory_firing_rate=1.0,
    Q_max_inhibitory_firing_rate=1.0,

    phi_temperature_scaling_W=0.5,
    tau_W_relaxation_time_W=1.0,

    T_ion_variance_channel_sigmoid=15.0,

    T_V_variance_firing_sigmoid=10.0,

    I_external_current=0.3,

    b_inhibitory_scaling_factor=0.1,
    C_intercolumn_coupling=0.1
)

noise_level = 0.10


def apply_noise_to_single_param_dict(params, noise_level=0.10):
    noisy_dict = {}
    for key, val in params.items():
        noisy_dict[key] = val * (1 + noise_level * (2 * np.random.rand() - 1))
    return noisy_dict


def create_node_specific_parameters(base_params, num_nodes, noise_level=0.10):
    node_params = []
    for _ in range(num_nodes):
        node_params.append(apply_noise_to_single_param_dict(base_params, noise_level))
    return node_params


def coupled_neurons_diffusive_loop(
    t,
    state,
    num_neurons,
    connectome_matrix,
    node_parameters
):
    state_reshaped = state.reshape((num_neurons, 3))
    V_i = state_reshaped[:, 0]
    Z_i = state_reshaped[:, 1]
    W_i = state_reshaped[:, 2]

    dV = np.zeros(num_neurons)
    dZ = np.zeros(num_neurons)
    dW = np.zeros(num_neurons)

    Q_local = np.zeros(num_neurons)
    for i in range(num_neurons):

        Q_max = node_parameters[i]["Q_max_excitatory_firing_rate"]
        V_T = node_parameters[i]["V_T_trigger_value"]
        delta_v =  node_parameters[i]["T_V_variance_firing_sigmoid"]

        Q_local[i] = 0.5 * Q_max * (1 + np.tanh((V_i[i] - V_T) / delta_v))

    Q_global = connectome_matrix @ Q_local

    for i in range(num_neurons):
        p = node_parameters[i]
        Q_max = 1.0

        delta_Ca = 0.15
        delta_Na = 0.3
        delta_K = 0.3

        delta_V = p["T_V_variance_firing_sigmoid"]

        V_Ca = 1.0
        V_Na = 0.53
        V_K = -0.7
        V_L = -0.5

        V_T_Ca = -0.01
        V_T_Na = 0.3
        V_T_K = 0.0

        g_Ca = 1.0
        g_Na = 6.7
        g_K = 2.0
        g_L = 0.5

        r_NMDA = 0.25
        a_ee = 0.36
        a_ei = 2.0
        a_ie = 2.0
        a_ne = 1.0
        a_ni = 0.4


        I = 0.3
        b = 0.1
        phi= 0.7
        tau= 1.0

        C = p["C_intercolumn_coupling"]

        Q_exc = Q_local[i]
        Q_inh = 0.5 * Q_max * (1 + np.tanh((V_i[i] - V_T) / delta_V))

        m_Ca = 0.5 * (1 + np.tanh((V_i[i] - V_T_Ca) / delta_Ca))
        m_Na = 0.5 * (1 + np.tanh((V_i[i] - V_T_Na) / delta_Na))
        m_K = 0.5 * (1 + np.tanh((V_i[i] - V_T_K) / delta_K))

        exc_nmda = (g_Ca +  (1 - 0* C)* r_NMDA * a_ee * Q_exc + 0 * C * r_NMDA *a_ee* Q_global[i] ) * m_Ca
        exc_ampa = (g_Na * m_Na +  (1 - 0* C)*a_ee * Q_exc + a_ee*Q_global[i])

        dV[i] = (
            -exc_nmda * (V_i[i] - V_Ca)
            - exc_ampa * (V_i[i] - V_Na)
            - g_K * W_i[i] * (V_i[i] - V_K)
            - g_L * (V_i[i] - V_L)
            + a_ie * Z_i[i] * Q_inh
            + a_ne * I
        )

        dZ[i] =  b * (a_ni * Q_exc + a_ei*V_i[i] * I)

        dW[i] = phi * (m_K - W_i[i]) / tau

    if t > 4000:
        asdad = 10

    return np.column_stack((dV, dZ, dW)).flatten()


def non_linear_model(params, time_span=(0, 500), time_steps=6000, connectome_matrix=None, plot=False):

    if connectome_matrix is None:
        connectome_matrix = np.array([
            [0.0, 1.0, 0.0, 0.2],
            [1.0, 0.0, 0.1, 0.0],
            [0.0, 0.1, 0.0, 1.0],
            [0.2, 0.0, 1.0, 0.0]
        ])

    num_nodes = connectome_matrix.shape[0]

    node_params = create_node_specific_parameters(params, num_nodes, noise_level)

    V0 = params["V_L_leak_reversal_potential"]
    Z0 = 0
    W0 = 0.5 * (1 + np.tanh((V0 - params["V_K_Nernst_potential_K"]) / params["T_ion_variance_channel_sigmoid"]))

    initial_state = np.tile([V0, Z0, W0], (num_nodes, 1)).flatten()
    initial_state = initial_state * (1 + noise_level * (2 * np.random.random(size = 3 * num_nodes) - 1))

    t_eval = np.linspace(time_span[0], time_span[1], time_steps)

    solution = solve_ivp(
        coupled_neurons_diffusive_loop,
        time_span,
        initial_state,
        args=(num_nodes, connectome_matrix, node_params),
        t_eval=t_eval,
        dense_output=True
    )

    V_i = solution.y.reshape((num_nodes, 3, -1))[:, 0, :]
    Z_i = solution.y.reshape((num_nodes, 3, -1))[:, 1, :]
    W_i = solution.y.reshape((num_nodes, 3, -1))[:, 2, :]

    if plot:
        plt.figure(figsize=(12, 6))
        for i in range(num_nodes):
            plt.plot(t_eval[-100:], V_i[i, -100:], label=f'Neuron {i+1}')
        plt.xlabel('Time')
        plt.ylabel('Membrane Potential (V)')
        plt.title('Coupled Neurons with Independent Node Parameters')
        plt.legend()
        plt.grid()
        plt.show()

    return V_i, Z_i, W_i, t_eval


def spm_hrf(time):
    peak1 = gamma.pdf(time, 6)
    peak2 = gamma.pdf(time, 16)
    hrf = peak1 - 0.35 * peak2
    hrf /= np.sum(hrf)
    return hrf


def non_linear_bold_z_model(params, connectome_matrix, time_span=(0, 500), time_steps=6000):
    V_i, Z_i, W_i, t_eval = non_linear_model(params, time_span, time_steps, connectome_matrix, plot=False)

    Q_i = 0.5 * params["Q_max_excitatory_firing_rate"] * (1 + np.tanh((V_i - (-70)) / 1.0))

    hrf = spm_hrf(t_eval)

    BOLD_all = np.array([np.convolve(neuron, hrf)[:neuron.shape[0]] for neuron in Q_i])

    BOLD_z_all = []
    for BOLD in BOLD_all:
        BOLD_z = (BOLD - np.mean(BOLD)) / np.std(BOLD)
        BOLD_z_all.append(BOLD_z)

    return np.array(BOLD_z_all), t_eval


def noisy(x):
    mod = (1 + noise_level * np.random.rand())
    temp = x * mod
    return temp


def init_param_files():
    init_param = [0.0] * 22
    init_param[0] = (params["g_Ca_max_Ca_conductance"])
    init_param[1] = (params["g_Na_max_Na_conductance"])
    init_param[2] = (params["g_K_max_K_conductance"])
    init_param[3] = (params["g_L_leak_conductance"])
    init_param[4] = (params["V_Ca_Nernst_potential_Ca"])
    init_param[5] = (params["V_Na_Nernst_potential_Na"])
    init_param[6] = (params["V_K_Nernst_potential_K"])
    init_param[7] = (params["V_L_leak_reversal_potential"])
    init_param[8] = (params["V_T_trigger_value"])
    init_param[9] = (params["a_ee_excitatory_to_excitatory"])
    init_param[10] = (params["a_ie_inhibitory_to_excitatory"])
    init_param[11] = (params["a_ei_excitatory_to_inhibitory"])
    init_param[12] = (params["r_NMDA_ratio_NMDA_to_AMPA"])
    init_param[13] = (params["Q_max_excitatory_firing_rate"])
    init_param[14] = (params["Q_max_inhibitory_firing_rate"])
    init_param[15] = (params["phi_temperature_scaling_W"])
    init_param[16] = (params["tau_W_relaxation_time_W"])
    init_param[17] = (params["T_ion_variance_channel_sigmoid"])
    init_param[18] = (params["T_V_variance_firing_sigmoid"])
    init_param[19] = (params["I_external_current"])
    init_param[20] = (params["b_inhibitory_scaling_factor"])
    init_param[21] = (params["C_intercolumn_coupling"])

    np.array([0]).tofile("optimization/nlm_best_fitness.bin")
    np.array(init_param).tofile("optimization/nlm_best_param.bin")


def get_best():
    best_fitness = np.fromfile(
        "C:\\Users\\Johan\\PycharmProjects\\MPCAS_SOCS_Project_BrainNetworkModdeling\\optimization\\nlm_best_fitness.bin",
        dtype=float)[0]

    path = "C:\\Users\\Johan\\PycharmProjects\\MPCAS_SOCS_Project_BrainNetworkModdeling\\optimization\\nlm_best_param.bin"
    best_params = np.fromfile(path, dtype=float)

    params = dict(
        g_Ca_max_Ca_conductance=best_params[0],
        g_Na_max_Na_conductance=best_params[1],
        g_K_max_K_conductance=best_params[2],
        g_L_leak_conductance=best_params[3],
        V_Ca_Nernst_potential_Ca=best_params[4],
        V_Na_Nernst_potential_Na=best_params[5],
        V_K_Nernst_potential_K=best_params[6],
        V_L_leak_reversal_potential=best_params[7],
        V_T_trigger_value=best_params[8],
        a_ee_excitatory_to_excitatory=best_params[9],
        a_ie_inhibitory_to_excitatory=best_params[10],
        a_ei_excitatory_to_inhibitory=best_params[11],
        r_NMDA_ratio_NMDA_to_AMPA=best_params[12],
        Q_max_excitatory_firing_rate=best_params[13],
        Q_max_inhibitory_firing_rate=best_params[14],
        phi_temperature_scaling_W=best_params[15],
        tau_W_relaxation_time_W=best_params[16],
        T_ion_variance_channel_sigmoid=best_params[17],
        T_V_variance_firing_sigmoid=best_params[18],
        I_external_current=best_params[19],
        b_inhibitory_scaling_factor=best_params[20],
        C_intercolumn_coupling=best_params[21]
    )

    print(f"best_fitness:{best_fitness}, best_alpha:{params}")
    return best_fitness, params


if __name__ == "__main__":

    C = data_management.load_DTI_data(0)
    act = data_management.load_fMRI_data(0)

    BOLD_z_all, t_eval = non_linear_bold_z_model(params, C,
                                                 time_span=(0, int(4800 * 1.2)),
                                                 time_steps=4800)

    plt.figure(figsize=(10, 4))
    for n, BOLD_z in enumerate(BOLD_z_all):
        plt.plot(t_eval, BOLD_z, label=f"node {n}")
    plt.title('Simulated resting-state BOLD Z-scored')
    plt.xlabel('Time (s)')
    plt.ylabel('BOLD signal')
    plt.legend()
    plt.show()


#https://docs.thevirtualbrain.org/_modules/tvb/simulator/models/larter_breakspear.html?utm_source=chatgpt.com