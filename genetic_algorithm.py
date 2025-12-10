import random
import math

import numpy as np

import data_management as dm
import nonLinear_model as nlm
import test
import multiprocessing as mp


creep_rate = 0.05

# Initialize population:
# Uncomment the line below and implement the function
def initialize_population(population_size, maximum_variable_value):
    population = []

    number_of_genes = len(nlm.params)

    for i in range(population_size):
        if i == 0:
            chromosome = encode_individual(nlm.params, maximum_variable_value)
        elif i == 1 or i == 2:
            chromosome = [min(max(np.random.normal(0, 0.1), -1), 1) for _ in range(number_of_genes)]
        else:
            mold = encode_individual(nlm.params, maximum_variable_value)
            chromosome = [min(max(mold[k] + np.random.normal(0, 0.01), -1), 1) for k in range(number_of_genes)]

    population.append(chromosome)
    return population

def encode_individual(individual, maximum_variable_value):

    chromosome = [0.0] * len(individual)
    # Fill using a for-loop
    for i, key in enumerate(individual):
        chromosome[i] = (individual[key] + maximum_variable_value[i])/ (2 * maximum_variable_value[i])

    return chromosome
# Decode chromosome:
def decode_chromosome(chromosome, maximum_variable_value):
    param_copy = test.params.copy()

    for i, key in enumerate(param_copy):
        param_copy[key] = -maximum_variable_value[i] + 2*maximum_variable_value*chromosome[i]

    return param_copy

# Evaluate indviduals:
def evaluate_individual(params):
    individuals = 100
    DTI_data = []
    fMRI_data = []

    for i in range(individuals):
        DTI_data.append(dm.load_DTI_data(i))  # SC matrix
        fMRI_data.append(dm.load_fMRI_data(i))  # fMRI timeseries

    try:
          t_span = (0, 4800 * 1.2)
          t_eval = np.linspace(t_span[1] - 1000 * 1.2, t_span[1], 1000)

          r = random.randint(0, 99)
          SC = DTI_data[r]


          num_nodes = SC.shape[0]

          V, W, Z, t_eval = nlm.simulate_network(
              num_nodes,
              t_span,
              t_eval,
              connectome_matrix=SC,
              noise_level=0.1
          )
          sim_fMRI = V.T

    except Exception:
        return -1  # Solver failure = zero fitness

    fMRI = fMRI_data[r][-1000:, :]  # shape (4800, 246)

    # not np.isfinite(sim_fMRI).all():
        #return -2

    if not fMRI.shape == sim_fMRI.shape:
        return -3

    err = np.sum((fMRI - sim_fMRI) ** 2)
    if err <= 0:
        return -2

    return 1.0 / err


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

def run_function_optimization(population_size, maximum_variable_value,
                              tournament_size, tournament_probability, crossover_probability,
                              mutation_probability, number_of_generations):

    population = initialize_population(population_size, maximum_variable_value)

    for generation_index in range(number_of_generations):
        if generation_index % 10 == 0:
            print(f"Generation: {generation_index}")

        # Load current maximum fitness
        maximum_fitness = np.fromfile("nlm_best_fitness.bin", dtype=float)[0]
        best_chromosome = []
        best_individual = []

        # === Parallel evaluation ===
        with mp.Pool(mp.cpu_count()) as pool:
            individuals_decoded = [decode_chromosome(chrom, maximum_variable_value) for chrom in population]
            fitness_list = pool.map(evaluate_individual, individuals_decoded)
        # ===========================

        # Update best individual
        for i, fitness in enumerate(fitness_list):
            if fitness > maximum_fitness:
                maximum_fitness = fitness
                np.array([maximum_fitness]).tofile("nlm_best_fitness.bin")
                best_chromosome = population[i].copy()
                best_individual = individuals_decoded[i].copy()
                np.array(best_chromosome).tofile("nlm_best_param.bin")

        print(f"Fitness list: {fitness_list}")

        # Selection and crossover
        temp_population = []
        for i in range(0, population_size, 2):
            index_1 = tournament_select(fitness_list, tournament_probability, tournament_size)
            index_2 = tournament_select(fitness_list, tournament_probability, tournament_size)
            chromosome1 = population[index_1].copy()
            chromosome2 = population[index_2].copy()
            if random.random() < crossover_probability:
                chromosome1, chromosome2 = cross(chromosome1, chromosome2)
            temp_population.append(chromosome1)
            temp_population.append(chromosome2)

        # Mutation
        for i in range(population_size):
            temp_population[i] = mutate(temp_population[i], mutation_probability)

        # Elitism: keep the best chromosome
        if len(best_chromosome) > 0:
            temp_population[0] = best_chromosome

        population = temp_population.copy()

    return [maximum_fitness, best_individual]
 

