import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def coupled_neurons_diffusive(
    t,
    state,
    num_neurons,
    calcium_conductance,
    sodium_conductance,
    potassium_conductance,
    leak_conductance,
    calcium_potential,
    sodium_potential,
    potassium_potential,
    leak_potential,
    excitatory_synapse_strength,
    inhibitory_synapse_strength_to_inhibitory,
    inhibitory_synapse_strength_to_excitatory,
    excitatory_synapse_strength_to_inhibitory,
    nmda_ratio,
    max_firing_rate,
    max_inhibitory_firing_rate,
    temperature_scaling,
    relaxation_time_constant,
    ion_channel_variance,
    firing_rate_variance,
    external_current,
    scaling_factor,
    connectome_matrix,
    coupling_strength
):
    state = state.reshape((num_neurons, 3))
    V, Z, W = state[:, 0], state[:, 1], state[:, 2]

    # Firing rates
    firing_rate = 0.5 * max_firing_rate * (1 + np.tanh((V - 30) / firing_rate_variance))

    # Ion channel activations
    Ca_act = 0.5 * (1 + np.tanh((V - calcium_potential) / ion_channel_variance))
    Na_act = 0.5 * (1 + np.tanh((V - sodium_potential) / ion_channel_variance))
    K_act  = 0.5 * (1 + np.tanh((V - potassium_potential) / ion_channel_variance))

    # Diffusive coupling (pull towards neighbors)
    diffusive_input = connectome_matrix @ V - np.sum(connectome_matrix, axis=1) * V

    # Differential equations
    dVdt = (
        -(calcium_conductance * Ca_act + nmda_ratio * excitatory_synapse_strength * firing_rate) * (V - calcium_potential)
        - (sodium_conductance * Na_act + excitatory_synapse_strength * firing_rate) * (V - sodium_potential)
        - potassium_conductance * W * (V - potassium_potential)
        - leak_conductance * (V - leak_potential)
        + excitatory_synapse_strength_to_inhibitory * Z * firing_rate
        + inhibitory_synapse_strength_to_excitatory * external_current
        + coupling_strength * diffusive_input
    )

    dZdt = scaling_factor * (inhibitory_synapse_strength_to_inhibitory * external_current + excitatory_synapse_strength_to_inhibitory * firing_rate)
    dWdt = temperature_scaling * (K_act - W) / relaxation_time_constant

    return np.column_stack((dVdt, dZdt, dWdt)).flatten()


# --- Parameters ---
num_neurons = 4
coupling_strength = 0.5  # tuned for network oscillations

params = dict(
    calcium_conductance = 1.2,
    sodium_conductance = 1.0,
    potassium_conductance = 1.8,
    leak_conductance = 0.5,
    calcium_potential = 100.0,
    sodium_potential = 50.0,
    potassium_potential = -77.0,
    leak_potential = -50.0,
    excitatory_synapse_strength = 0.5,
    inhibitory_synapse_strength_to_inhibitory = 0.1,
    inhibitory_synapse_strength_to_excitatory = 0.1,
    excitatory_synapse_strength_to_inhibitory = 0.1,
    nmda_ratio = 0.3,
    max_firing_rate = 1.0,
    max_inhibitory_firing_rate = 1.0,
    temperature_scaling = 0.5,
    relaxation_time_constant = 15.0,
    ion_channel_variance = 1.0,
    firing_rate_variance = 1.0,
    external_current = 0.5,
    scaling_factor = 0.1
)

connectome_matrix = np.array([
    [0.0, 0.1, 0.0, 0.2],  # Neuron 1 receives input from neurons 2 and 4
    [0.1, 0.0, 0.1, 0.0],  # Neuron 2 receives input from neurons 1 and 3
    [0.0, 0.1, 0.0, 1],  # Neuron 3 receives input from neurons 2 and 4
    [0.2, 0.0, 1, 0.0]   # Neuron 4 receives input from neurons 1 and 3
])


# Initial conditions
initial_state = np.tile([-70.0, 0, 0], (num_neurons, 1)).flatten() + np.random.normal(0, 1, 3*num_neurons)
print(initial_state)

# Time span
time_windows = [(0, 20), (100, 110), (2000, 2010)]
time_span = (min(min(time_windows)), max(max(time_windows)))

for time_window in time_windows:

    t_eval = np.linspace(time_window[0], time_window[1], 1000)

    # Solve the system
    solution = solve_ivp(
        coupled_neurons_diffusive,
        time_span,
        initial_state,
        t_eval=t_eval,
        args=(num_neurons, *params.values(), connectome_matrix, coupling_strength)
    )

    # Reshape solution
    V = solution.y.reshape((num_neurons, 3, -1))[:, 0, :]

    # Plot all 6 neurons
    plt.figure(figsize=(12, 6))
    for i in range(num_neurons):
        l = len(t_eval)
        plt.plot(t_eval, V[i, :], label=f'Neuron {i+1}')  # skip transient
    plt.xlabel('Time')
    plt.ylabel('Membrane Potential (V)')
    plt.title('6-Node Coupled Neurons with Diffusive Coupling')
    plt.legend()
    plt.grid()
    plt.show()
