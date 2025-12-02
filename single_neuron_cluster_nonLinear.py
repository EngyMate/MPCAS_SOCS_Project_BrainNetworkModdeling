import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def neural_model(
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
    scaling_factor
):
    membrane_potential, inhibitory_variable, potassium_channel_variable = state

    # Fraction of open ion channels
    calcium_channel_activation = 0.5 * (1 + np.tanh((membrane_potential - calcium_potential) / ion_channel_variance))
    sodium_channel_activation = 0.5 * (1 + np.tanh((membrane_potential - sodium_potential) / ion_channel_variance))
    potassium_channel_activation = 0.5 * (1 + np.tanh((membrane_potential - potassium_potential) / ion_channel_variance))

    # Firing rate
    firing_rate = 0.5 * max_firing_rate * (1 + np.tanh((membrane_potential - 30) / firing_rate_variance))

    # Differential equations
    dVdt = -(calcium_conductance * calcium_channel_activation + nmda_ratio * excitatory_synapse_strength * firing_rate) * (membrane_potential - calcium_potential) \
          - (sodium_conductance * sodium_channel_activation + excitatory_synapse_strength * firing_rate) * (membrane_potential - sodium_potential) \
          - potassium_conductance * potassium_channel_variable * (membrane_potential - potassium_potential) \
          - leak_conductance * (membrane_potential - leak_potential) \
          + excitatory_synapse_strength_to_inhibitory * inhibitory_variable * firing_rate + inhibitory_synapse_strength_to_excitatory * external_current

    dZdt = scaling_factor * (inhibitory_synapse_strength_to_inhibitory * external_current + excitatory_synapse_strength_to_inhibitory * firing_rate)
    dWdt = temperature_scaling * (potassium_channel_activation - potassium_channel_variable) / relaxation_time_constant

    return [dVdt, dZdt, dWdt]

# Parameters with understandable names
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

# Initial conditions
initial_state = [-70.0, 0.0, 0.0]  # membrane_potential, inhibitory_variable, potassium_channel_variable

# Time span
time_span = (0, 100)

# Solve the system
solution = solve_ivp(
    neural_model,
    time_span,
    initial_state,
    args=(
        calcium_conductance, sodium_conductance, potassium_conductance, leak_conductance,
        calcium_potential, sodium_potential, potassium_potential, leak_potential,
        excitatory_synapse_strength, inhibitory_synapse_strength_to_inhibitory,
        inhibitory_synapse_strength_to_excitatory, excitatory_synapse_strength_to_inhibitory,
        nmda_ratio, max_firing_rate, max_inhibitory_firing_rate, temperature_scaling,
        relaxation_time_constant, ion_channel_variance, firing_rate_variance, external_current,
        scaling_factor
    ),
    dense_output=True
)

# Plot the results
time_points = np.linspace(0, 100, 1000)
membrane_potential, inhibitory_variable, potassium_channel_variable = solution.sol(time_points)

plt.figure(figsize=(12, 8))
plt.plot(time_points, membrane_potential.T, label='Membrane Potential (V)')
plt.title('Simulation of Neural Dynamics')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid()
plt.show()
