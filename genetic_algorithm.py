import random
import math


number_of_genes = 22
creep_rate = 0.05

# Initialize population:
# Uncomment the line below and implement the function
def initialize_population(population_size, maximum_variable_value):
  population = []

  for i in range(population_size):
    if i==0:
      params = dict(
        g_Ca_max_Ca_conductance=1.2,
        g_Na_max_Na_conductance=1.0,
        g_K_max_K_conductance=1.8,
        g_L_leak_conductance=0.5,
        V_Ca_Nernst_potential_Ca=100.0,
        V_Na_Nernst_potential_Na=30.0,
        V_K_Nernst_potential_K=-77.0,
        V_L_leak_reversal_potential=-50.0,
        V_T_trigger_value=30,
        a_ee_excitatory_to_excitatory=0.1,
        a_ie_inhibitory_to_excitatory=0.9,
        a_ei_excitatory_to_inhibitory=0.1,
        r_NMDA_ratio_NMDA_to_AMPA=0.3,
        Q_max_excitatory_firing_rate=1.0,
        Q_max_inhibitory_firing_rate=1.0,
        phi_temperature_scaling_W=0.5,
        tau_W_relaxation_time_W=1.0,
        T_ion_variance_channel_sigmoid=0.001,
        T_V_variance_firing_sigmoid=1.0,
        I_external_current=0.3,
        b_inhibitory_scaling_factor=0.1,
        C_intercolumn_coupling=0.3
      )
      chromosome = [0.0] * 22
      chromosome[0] = (params["g_Ca_max_Ca_conductance"] + maximum_variable_value) / (2 * maximum_variable_value)
      chromosome[1] = (params["g_Na_max_Na_conductance"] + maximum_variable_value) / (2 * maximum_variable_value)
      chromosome[2] = (params["g_K_max_K_conductance"] + maximum_variable_value) / (2 * maximum_variable_value)
      chromosome[3] = (params["g_L_leak_conductance"] + maximum_variable_value) / (2 * maximum_variable_value)
      chromosome[4] = (params["V_Ca_Nernst_potential_Ca"] + maximum_variable_value) / (2 * maximum_variable_value)
      chromosome[5] = (params["V_Na_Nernst_potential_Na"] + maximum_variable_value) / (2 * maximum_variable_value)
      chromosome[6] = (params["V_K_Nernst_potential_K"] + maximum_variable_value) / (2 * maximum_variable_value)
      chromosome[7] = (params["V_L_leak_reversal_potential"] + maximum_variable_value) / (2 * maximum_variable_value)
      chromosome[8] = (params["V_T_trigger_value"] + maximum_variable_value) / (2 * maximum_variable_value)
      chromosome[9] = (params["a_ee_excitatory_to_excitatory"] + maximum_variable_value) / (2 * maximum_variable_value)
      chromosome[10] = (params["a_ie_inhibitory_to_excitatory"] + maximum_variable_value) / (2 * maximum_variable_value)
      chromosome[11] = (params["a_ei_excitatory_to_inhibitory"] + maximum_variable_value) / (2 * maximum_variable_value)
      chromosome[12] = (params["r_NMDA_ratio_NMDA_to_AMPA"] + maximum_variable_value) / (2 * maximum_variable_value)
      chromosome[13] = (params["Q_max_excitatory_firing_rate"] + maximum_variable_value) / (2 * maximum_variable_value)
      chromosome[14] = (params["Q_max_inhibitory_firing_rate"] + maximum_variable_value) / (2 * maximum_variable_value)
      chromosome[15] = (params["phi_temperature_scaling_W"] + maximum_variable_value) / (2 * maximum_variable_value)
      chromosome[16] = (params["tau_W_relaxation_time_W"] + maximum_variable_value) / (2 * maximum_variable_value)
      chromosome[17] = (params["T_ion_variance_channel_sigmoid"] + maximum_variable_value) / (2 * maximum_variable_value)
      chromosome[18] = (params["T_V_variance_firing_sigmoid"] + maximum_variable_value) / (2 * maximum_variable_value)
      chromosome[19] = (params["I_external_current"] + maximum_variable_value) / (2 * maximum_variable_value)
      chromosome[20] = (params["b_inhibitory_scaling_factor"] + maximum_variable_value) / (2 * maximum_variable_value)
      chromosome[21] = (params["C_intercolumn_coupling"] + maximum_variable_value) / (2 * maximum_variable_value)

    else:
      chromosome = [random.random() for _ in range(number_of_genes)]
    population.append(chromosome)
  return population

# Decode chromosome:
def decode_chromosome(chromosome, maximum_variable_value):

  params = dict(
    g_Ca_max_Ca_conductance=-maximum_variable_value + 2*maximum_variable_value*chromosome[0],
    g_Na_max_Na_conductance=-maximum_variable_value + 2*maximum_variable_value*chromosome[1],
    g_K_max_K_conductance=-maximum_variable_value + 2*maximum_variable_value*chromosome[2],
    g_L_leak_conductance=-maximum_variable_value + 2*maximum_variable_value*chromosome[3],
    V_Ca_Nernst_potential_Ca=-maximum_variable_value + 2*maximum_variable_value*chromosome[4],
    V_Na_Nernst_potential_Na=-maximum_variable_value + 2*maximum_variable_value*chromosome[5],
    V_K_Nernst_potential_K=-maximum_variable_value + 2*maximum_variable_value*chromosome[6],
    V_L_leak_reversal_potential=-maximum_variable_value + 2*maximum_variable_value*chromosome[7],
    V_T_trigger_value=-maximum_variable_value + 2*maximum_variable_value*chromosome[8],
    a_ee_excitatory_to_excitatory=-maximum_variable_value + 2*maximum_variable_value*chromosome[9],
    a_ie_inhibitory_to_excitatory=-maximum_variable_value + 2*maximum_variable_value*chromosome[10],
    a_ei_excitatory_to_inhibitory=-maximum_variable_value + 2*maximum_variable_value*chromosome[11],
    r_NMDA_ratio_NMDA_to_AMPA=-maximum_variable_value + 2*maximum_variable_value*chromosome[12],
    Q_max_excitatory_firing_rate=-maximum_variable_value + 2*maximum_variable_value*chromosome[13],
    Q_max_inhibitory_firing_rate=-maximum_variable_value + 2*maximum_variable_value*chromosome[14],
    phi_temperature_scaling_W=-maximum_variable_value + 2*maximum_variable_value*chromosome[15],
    tau_W_relaxation_time_W=-maximum_variable_value + 2*maximum_variable_value*chromosome[16],
    T_ion_variance_channel_sigmoid=-maximum_variable_value + 2*maximum_variable_value*chromosome[17],
    T_V_variance_firing_sigmoid=-maximum_variable_value + 2*maximum_variable_value*chromosome[18],
    I_external_current=-maximum_variable_value + 2*maximum_variable_value*chromosome[19],
    b_inhibitory_scaling_factor=-maximum_variable_value + 2*maximum_variable_value*chromosome[20],
    C_intercolumn_coupling=-maximum_variable_value + 2*maximum_variable_value*chromosome[21]
  )

  return params

# Evaluate indviduals:
def evaluate_individual(x):
  #g(x1, x2) = (1.5 − x1 + x1x2)2 + (2.25 − x1 + x1x22)2 + (2.625 − x1 + x1x32)2
  x1 = x[0]
  x2 = x[1]
  g_x = pow((1.5 - x1 + x1*x2), 2)  \
  + pow((2.25 - x1 + x1*pow(x2,2)),2) \
  + pow((2.625 - x1 + x1*pow(x2, 3)),2)

  fitness = pow(g_x+1, -1)
  return fitness

# Select individuals:
def tournament_select(fitness_list, tournament_probability, tournament_size):
  selected_individuals = []
  number_of_individuals = len(fitness_list)

  for i in range(tournament_size):
      individual_nr = random.randint(0,number_of_individuals-1)
      selected_individuals.append([individual_nr, fitness_list[individual_nr]])

  selected_individuals.sort(key = lambda x:x[1], reverse=True)

  for trial in range(tournament_size - 1):
    r = random.random()
    if r<tournament_probability:
      return selected_individuals[0][0]
    else:
      selected_individuals.pop(0)

  return selected_individuals[0][0]


# Carry out crossover:
def cross(chromosome1, chromosome2):
  length_of_chromosome = len(chromosome1)
  crossover_index = random.randint(1, length_of_chromosome-2)

  chromosome1_new = chromosome1[:crossover_index] + chromosome2[crossover_index:]
  chromosome2_new = chromosome2[:crossover_index] + chromosome1[crossover_index:]

  return chromosome1_new, chromosome2_new

# Mutate individuals:
def mutate(chromosome, mutation_probability):
  new_chromosome = chromosome.copy()

  number_of_genes = len(chromosome)
  for gene_index in range(number_of_genes):
    r = random.random()
    if r < mutation_probability:
      mutated_value = random.gauss(new_chromosome[gene_index], creep_rate)
      if mutated_value > 1:
          mutated_value = 1
      elif mutated_value <0:
          mutated_value = 0
      new_chromosome[gene_index] = mutated_value

  return new_chromosome

# Genetic algorithm

def run_function_optimization(population_size, number_of_genes, number_of_variables, maximum_variable_value, \
                              tournament_size, tournament_probability, crossover_probability,\
                              mutation_probability, number_of_generations):
 
 # This function should return the maximum fitness and the best individual (i.e., a vector with
 # two elements (x1,x2) containing the values corresponding to the maximum fitness found.
 
 # Note that some parameters have different names compared to the programming introduction

  population = initialize_population(population_size,number_of_genes)

  for generation_index in range(number_of_generations):

    maximum_fitness = 0

    best_chromosome = []
    best_individual = []
    fitness_list = []
    for chromosome in population:
      individual = decode_chromosome(chromosome,maximum_variable_value)
      fitness = evaluate_individual(individual)
      if (fitness > maximum_fitness):
        maximum_fitness = fitness
        best_chromosome = chromosome.copy()  
        best_individual = individual.copy()
      fitness_list.append(fitness)

    temp_population = []
    for i in range(0,population_size,2):
      index_1 = tournament_select(fitness_list, tournament_probability, tournament_size)
      index_2 = tournament_select(fitness_list, tournament_probability, tournament_size)
      chromosome1 = population[index_1].copy()
      chromosome2 = population[index_2].copy()
      r = random.random()
      if r < crossover_probability:
        [new_chromosome_1, new_chromosome_2] = cross(chromosome1,chromosome2)
        temp_population.append(new_chromosome_1)
        temp_population.append(new_chromosome_2) 
      else:
        temp_population.append(chromosome1)
        temp_population.append(chromosome2)

    for i in range(population_size):
      original_chromosome = temp_population[i]

      mutated_chromosome = mutate(original_chromosome, mutation_probability)
      temp_population[i] = mutated_chromosome

    temp_population[0] = best_chromosome
    population = temp_population.copy()

  return [maximum_fitness, best_individual]
 

