import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# --- Coupled neuron dynamics with proper independent updates ---
def coupled_neurons_diffusive_loop(
    t,
    state,
    num_neurons,
    g_Ca_max_Ca_conductance,
    g_Na_max_Na_conductance,
    g_K_max_K_conductance,
    g_L_leak_conductance,
    V_Ca_Nernst_potential_Ca,
    V_Na_Nernst_potential_Na,
    V_K_Nernst_potential_K,
    V_L_leak_reversal_potential,
    V_T_trigger_value,
    a_ee_excitatory_to_excitatory,
    a_ie_inhibitory_to_excitatory,
    a_ei_excitatory_to_inhibitory,
    r_NMDA_ratio_NMDA_to_AMPA,
    Q_max_excitatory_firing_rate,
    Q_max_inhibitory_firing_rate,
    phi_temperature_scaling_W,
    tau_W_relaxation_time_W,
    T_ion_variance_channel_sigmoid,
    T_V_variance_firing_sigmoid,
    I_external_current,
    b_inhibitory_scaling_factor,
    connectome_matrix,
    C_intercolumn_coupling
):
    # Reshape state vector: [V_i, Z_i, W_i] per neuron
    state_reshaped = state.reshape((num_neurons, 3))
    V_i_membrane_potential_exc = state_reshaped[:, 0]
    Z_i_membrane_potential_inh = state_reshaped[:, 1]
    W_i_open_K_fraction = state_reshaped[:, 2]

    # Initialize derivatives
    dVdt_membrane_exc = np.zeros(num_neurons)
    dZdt_membrane_inh = np.zeros(num_neurons)
    dWdt_K_recovery = np.zeros(num_neurons)

    # --- Compute firing rates for all neurons first ---
    Q_i_V_local_exc_firing_rate = 0.5 * Q_max_excitatory_firing_rate * (1 + np.tanh((V_i_membrane_potential_exc - V_T_trigger_value) / T_V_variance_firing_sigmoid))
    Q_i_V_local_inhibitary_firing_rate =  0.5 * Q_max_inhibitory_firing_rate * (1 + np.tanh((V_i_membrane_potential_exc - V_T_trigger_value) / T_V_variance_firing_sigmoid))

    # --- Loop over neurons ---
    for i in range(num_neurons):
        # Global firing rate for neuron i
        Q_V_global_exc_firing_rate = np.sum(connectome_matrix[i, :] * Q_i_V_local_exc_firing_rate)

        # Ion channel fractions
        m_Ca = 0.5 * (1 + np.tanh((V_i_membrane_potential_exc[i] - V_Ca_Nernst_potential_Ca) / T_ion_variance_channel_sigmoid))
        m_Na = 0.5 * (1 + np.tanh((V_i_membrane_potential_exc[i] - V_Na_Nernst_potential_Na) / T_ion_variance_channel_sigmoid))
        m_K  = 0.5 * (1 + np.tanh((V_i_membrane_potential_exc[i] - V_K_Nernst_potential_K) / T_ion_variance_channel_sigmoid))

        # Excitatory currents (NMDA + AMPA)
        exc_nmda = (g_Ca_max_Ca_conductance + r_NMDA_ratio_NMDA_to_AMPA * a_ee_excitatory_to_excitatory *
                    ((1 - C_intercolumn_coupling) * Q_i_V_local_exc_firing_rate[i] + C_intercolumn_coupling * Q_V_global_exc_firing_rate)) * m_Ca

        exc_ampa = (g_Na_max_Na_conductance + a_ee_excitatory_to_excitatory *
                    ((1 - C_intercolumn_coupling) * Q_i_V_local_exc_firing_rate[i] + C_intercolumn_coupling * Q_V_global_exc_firing_rate)) * m_Na

        # Differential equations
        dVdt_membrane_exc[i] = (
            -exc_nmda * (V_i_membrane_potential_exc[i] - V_Ca_Nernst_potential_Ca)
            -exc_ampa * (V_i_membrane_potential_exc[i] - V_Na_Nernst_potential_Na)
            -g_K_max_K_conductance * W_i_open_K_fraction[i] * (V_i_membrane_potential_exc[i] - V_K_Nernst_potential_K)
            -g_L_leak_conductance * (V_i_membrane_potential_exc[i] - V_L_leak_reversal_potential)
            + a_ei_excitatory_to_inhibitory * Z_i_membrane_potential_inh[i] * Q_i_V_local_inhibitary_firing_rate[i]
            + a_ie_inhibitory_to_excitatory * I_external_current
        )

        dZdt_membrane_inh[i] = b_inhibitory_scaling_factor * \
            (a_ei_excitatory_to_inhibitory * Q_i_V_local_exc_firing_rate[i] + a_ie_inhibitory_to_excitatory * I_external_current)

        dWdt_K_recovery[i] = phi_temperature_scaling_W * (m_K - W_i_open_K_fraction[i]) / tau_W_relaxation_time_W

    # Return flattened derivatives
    return np.column_stack((dVdt_membrane_exc, dZdt_membrane_inh, dWdt_K_recovery)).flatten()


params = dict(
    g_Ca_max_Ca_conductance = 1.2,
    g_Na_max_Na_conductance = 1.0,
    g_K_max_K_conductance = 1.8,
    g_L_leak_conductance = 0.5,
    V_Ca_Nernst_potential_Ca = 100.0,
    V_Na_Nernst_potential_Na = 30.0,
    V_K_Nernst_potential_K = -77.0,
    V_L_leak_reversal_potential = -50.0,
    V_T_trigger_value=30,
    a_ee_excitatory_to_excitatory = 0.1,
    a_ie_inhibitory_to_excitatory = 0.9,
    a_ei_excitatory_to_inhibitory = 0.1,
    r_NMDA_ratio_NMDA_to_AMPA = 0.3,
    Q_max_excitatory_firing_rate = 1.0,
    Q_max_inhibitory_firing_rate = 1.0,
    phi_temperature_scaling_W = 0.5,
    tau_W_relaxation_time_W = 1.0,
    T_ion_variance_channel_sigmoid = 0.001,
    T_V_variance_firing_sigmoid = 1.0,
    I_external_current = 0.3,
    b_inhibitory_scaling_factor = 0.1,
    C_intercolumn_coupling = 0.3
)

def non_linear_model(params, num_neuronal_nodes=4, connectome_matrix=None, plot=False):
    # --- Parameters ---

    if connectome_matrix is None:
        connectome_matrix = np.array([
            [0.0, 1.0, 0.0, 0.2],
            [1.0, 0.0, 0.1, 0.0],
            [0.0, 0.1, 0.0, 1.0],
            [0.2, 0.0, 1.0, 0.0]
        ])
        num_neuronal_nodes = 4

    W_i_initial = 0.5 * (1 + np.tanh((-71 - params["V_K_Nernst_potential_K"]) / params["T_ion_variance_channel_sigmoid"]))
    # --- Initial conditions with larger random differences ---
    initial_state = np.tile([-71.0, 0, W_i_initial], (num_neuronal_nodes, 1)).flatten() + np.random.normal(0, 1.0, 3*num_neuronal_nodes)

    # --- Simulation ---
    time_span = (0, 500)
    t_eval = np.linspace(time_span[0], time_span[1], 5000)

    solution = solve_ivp(
        lambda t, y: coupled_neurons_diffusive_loop(
            t, y,
            num_neuronal_nodes,
            *[
                params["g_Ca_max_Ca_conductance"],
                params["g_Na_max_Na_conductance"],
                params["g_K_max_K_conductance"],
                params["g_L_leak_conductance"],
                params["V_Ca_Nernst_potential_Ca"],
                params["V_Na_Nernst_potential_Na"],
                params["V_K_Nernst_potential_K"],
                params["V_L_leak_reversal_potential"],
                params["V_T_trigger_value"],
                params["a_ee_excitatory_to_excitatory"],
                params["a_ie_inhibitory_to_excitatory"],
                params["a_ei_excitatory_to_inhibitory"],
                params["r_NMDA_ratio_NMDA_to_AMPA"],
                params["Q_max_excitatory_firing_rate"],
                params["Q_max_inhibitory_firing_rate"],
                params["phi_temperature_scaling_W"],
                params["tau_W_relaxation_time_W"],
                params["T_ion_variance_channel_sigmoid"],
                params["T_V_variance_firing_sigmoid"],
                params["I_external_current"],
                params["b_inhibitory_scaling_factor"]
            ],
            connectome_matrix,
            params["C_intercolumn_coupling"]  # now only passed once
        ),
        time_span,
        initial_state,
        t_eval=t_eval
    )

    V_i_membrane_potential_exc = solution.y.reshape((num_neuronal_nodes, 3, -1))[:, 0, :]
    Z_i_membrane_potential_inh = solution.y.reshape((num_neuronal_nodes, 3, -1))[:, 1, :]
    W_i_open_K_fraction = solution.y.reshape((num_neuronal_nodes, 3, -1))[:, 2, :]

    if plot:
        # --- Plot ---
        plt.figure(figsize=(12, 6))
        for i in range(num_neuronal_nodes):
            plt.plot(t_eval[-100:], V_i_membrane_potential_exc[i, -100:], label=f'Neuron {i+1}')
        plt.xlabel('Time')
        plt.ylabel('Membrane Potential (V)')
        plt.title('Coupled Neurons with Loop-Based Independent Dynamics')
        plt.legend()
        plt.grid()
        plt.show()

    return V_i_membrane_potential_exc, Z_i_membrane_potential_inh, W_i_open_K_fraction, t_eval


from scipy.signal import convolve
from scipy.stats import gamma

# HRF (SPM-like double gamma)
def spm_hrf(time):

    # parameters for the canonical HRF
    peak1 = gamma.pdf(time, 6)
    peak2 = gamma.pdf(time, 16)
    hrf = peak1 - 0.35 * peak2
    hrf /= np.sum(hrf)  # normalize area
    return hrf


if __name__ == "__main__":
    V_i, Z_i, W_i, t_eval = non_linear_model(params, plot=False)

    # Convert to firing rate
    Q_i = 0.5 * params["Q_max_excitatory_firing_rate"] * (1 + np.tanh((V_i - (-70)) / 1.0))

    # HRF
    hrf = spm_hrf(1.0)  # TR = 1s

    # Convolve each neuron/population
    BOLD_all = np.array([np.convolve(neuron, hrf)[:neuron.shape[0]] for neuron in Q_i])
    BOLD_z_all = []

    for BOLD in BOLD_all:

        BOLD_z = (BOLD - np.mean(BOLD, axis=0)) / np.std(BOLD, axis=0)
        BOLD_fc_matrix = np.corrcoef(BOLD_z.T)  # transpose so correlation is computed between columns (regions)
        BOLD_z_all.append(BOLD_z)

    # Plot a single neuron
    plt.figure(figsize=(10, 4))
    for BOLD_z in BOLD_z_all:
        plt.plot(t_eval[-200:], BOLD_z[ -200:])
    plt.title('Simulated resting-state BOLD Z-scored')
    plt.xlabel('Time (s)')
    plt.ylabel('BOLD signal')
    plt.show()



