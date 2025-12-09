import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def single_neuron_model(
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
    V, Z, W = state

    # Firing rate
    firing_rate = 0.5 * max_firing_rate * (1 + np.tanh((V - 30) / firing_rate_variance))

    # Ion channel activations
    Ca_act = 0.5 * (1 + np.tanh((V - calcium_potential) / ion_channel_variance))
    Na_act = 0.5 * (1 + np.tanh((V - sodium_potential) / ion_channel_variance))
    K_act  = 0.5 * (1 + np.tanh((V - potassium_potential) / ion_channel_variance))

    # Differential equations
    dVdt = (
        -(calcium_conductance * Ca_act + nmda_ratio * excitatory_synapse_strength * firing_rate) * (V - calcium_potential)
        - (sodium_conductance * Na_act + excitatory_synapse_strength * firing_rate) * (V - sodium_potential)
        - potassium_conductance * W * (V - potassium_potential)
        - leak_conductance * (V - leak_potential)
        + excitatory_synapse_strength_to_inhibitory * Z * firing_rate
        + inhibitory_synapse_strength_to_excitatory * external_current
    )
    dZdt = scaling_factor * (inhibitory_synapse_strength_to_inhibitory * external_current + excitatory_synapse_strength_to_inhibitory * firing_rate)
    dWdt = temperature_scaling * (K_act - W) / relaxation_time_constant

    return [dVdt, dZdt, dWdt]

# Function to run n independent simulations
def run_multiple_neurons(
    n_runs=5,
    time_span=(0, 2000),
    time_points=4000,
    noise_std=1.0
):
    # Parameters
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

    t_eval = np.linspace(time_span[1]-200, time_span[1], time_points)
    plt.figure(figsize=(12, 6))

    for run in range(n_runs):
        # Add random noise to initial conditions
        initial_state = [-70.0, 0.0, 0.0] + np.random.normal(0, noise_std, size=3)
        sol = solve_ivp(
            single_neuron_model,
            time_span,
            initial_state,
            args=tuple(params.values()),
            t_eval=t_eval,
            dense_output=True
        )
        V = sol.sol(t_eval)[0]
        l = len(t_eval)
        plt.plot(t_eval[l-400:], V[l-400:], label=f'Run {run+1}')  # skip transient

    plt.title(f'Single Neuron: {n_runs} Independent Runs')
    plt.xlabel('Time')
    plt.ylabel('Membrane Potential (V)')
    plt.grid()
    plt.legend()
    plt.show()

# Example: run 5 independent simulations
run_multiple_neurons(n_runs=1)
