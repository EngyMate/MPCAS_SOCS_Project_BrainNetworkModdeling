import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def neural_model_connectome(
    time,
    state,
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
    coupling_constant,
    connectome_matrix,
    num_nodes
):
    # Reshape state to separate variables for each node
    state = state.reshape((num_nodes, 3))
    membrane_potential = state[:, 0]
    inhibitory_variable = state[:, 1]
    potassium_channel_variable = state[:, 2]

    # Compute firing rate for each node
    firing_rate = 0.5 * max_firing_rate * (1 + np.tanh((membrane_potential - 30) / firing_rate_variance))

    # Compute weighted sum of firing rates using the connectome matrix
    weighted_firing_rate = np.dot(connectome_matrix, firing_rate)

    # Fraction of open ion channels for each node
    calcium_channel_activation = 0.5 * (1 + np.tanh((membrane_potential - calcium_potential) / ion_channel_variance))
    sodium_channel_activation = 0.5 * (1 + np.tanh((membrane_potential - sodium_potential) / ion_channel_variance))
    potassium_channel_activation = 0.5 * (1 + np.tanh((membrane_potential - potassium_potential) / ion_channel_variance))

    # Differential equations for each node
    dVdt = (
        -(calcium_conductance * calcium_channel_activation + (1 - coupling_constant) * nmda_ratio * excitatory_synapse_strength * firing_rate + coupling_constant * nmda_ratio * excitatory_synapse_strength * weighted_firing_rate) * (membrane_potential - calcium_potential)
        - (sodium_conductance * sodium_channel_activation + (1 - coupling_constant) * excitatory_synapse_strength * firing_rate + coupling_constant * excitatory_synapse_strength * weighted_firing_rate) * (membrane_potential - sodium_potential)
        - potassium_conductance * potassium_channel_variable * (membrane_potential - potassium_potential)
        - leak_conductance * (membrane_potential - leak_potential)
        + excitatory_synapse_strength_to_inhibitory * inhibitory_variable * firing_rate
        + inhibitory_synapse_strength_to_excitatory * external_current
    )

    dZdt = scaling_factor * (inhibitory_synapse_strength_to_inhibitory * external_current + excitatory_synapse_strength_to_inhibitory * firing_rate)
    dWdt = temperature_scaling * (potassium_channel_activation - potassium_channel_variable) / relaxation_time_constant

    # Flatten the derivatives for solve_ivp
    return np.column_stack((dVdt, dZdt, dWdt)).flatten()

# Parameters
calcium_conductance = 1.2
sodium_conductance = 1.0
potassium_conductance = 1.8
leak_conductance = 0.5

calcium_potential = 100.0
sodium_potential = 50.0
potassium_potential = -77.0
leak_potential = -50.0

excitatory_synapse_strength = 0.5
inhibitory_synapse_strength_to_inhibitory = 0.1
inhibitory_synapse_strength_to_excitatory = 0.1
excitatory_synapse_strength_to_inhibitory = 0.1

nmda_ratio = 0.3
max_firing_rate = 1.0
max_inhibitory_firing_rate = 1.0
temperature_scaling = 0.5
relaxation_time_constant = 15.0
ion_channel_variance = 1.0
firing_rate_variance = 1.0
external_current = 0.5
scaling_factor = 0.1

# Coupling parameters
coupling_constant = 1
num_nodes = 3  # Example: 5 nodes

# Example connectome matrix (random for demonstration)
connectome_matrix = np.array([[0.0,0.1,0.3],
                              [0.1,0.0,0.1],
                            [0.3,0.1,0.0]])

# Normalize the connectome matrix (optional)
connectome_matrix /= np.sum(connectome_matrix, axis=1, keepdims=True)

# Initial conditions for all nodes
initial_state = np.tile([-70.0, 0.0, 0.0], (num_nodes, 1)).flatten() + np.random.normal(0, 0.1, size=3*num_nodes)
print(initial_state)

# Time span
time_span = (0, 250)

# Solve the system
solution = solve_ivp(
    neural_model_connectome,
    time_span,
    initial_state,
    args=(
        calcium_conductance, sodium_conductance, potassium_conductance, leak_conductance,
        calcium_potential, sodium_potential, potassium_potential, leak_potential,
        excitatory_synapse_strength, inhibitory_synapse_strength_to_inhibitory,
        inhibitory_synapse_strength_to_excitatory, excitatory_synapse_strength_to_inhibitory,
        nmda_ratio, max_firing_rate, max_inhibitory_firing_rate, temperature_scaling,
        relaxation_time_constant, ion_channel_variance, firing_rate_variance, external_current,
        scaling_factor, coupling_constant, connectome_matrix, num_nodes
    ),
    dense_output=True
)

# Plot the results for each node
time_points = np.linspace(200, 250, 1000)
state_trajectories = solution.sol(time_points)  # Shape: (num_nodes * 3, len(time_points))

# Reshape to (num_nodes, 3, len(time_points)) for easier indexing
state_trajectories = state_trajectories.reshape((num_nodes, 3, -1))

# Extract membrane potential for each node
membrane_potential = state_trajectories[:, 0, :]  # Shape: (num_nodes, len(time_points))

plt.figure(figsize=(12, 8))
for i in range(num_nodes):
    plt.plot(time_points, membrane_potential[i, :], label=f'Node {i+1} Membrane Potential')
plt.title('Simulation of Coupled Neural Dynamics with Connectome Matrix')
plt.xlabel('Time')
plt.ylabel('Membrane Potential (V)')
plt.legend()
plt.grid()
plt.show()

