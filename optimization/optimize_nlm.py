import genetic_algorithm as ga
import data_management as dm

if __name__ == "__main__":
    individuals = 100
    DTI_data = []
    fMRI_data = []
    for i in range(individuals):
        DTI_data.append(dm.load_DTI_data(i))
        fMRI_data.append(dm.load_fMRI_data(i))

    population_size = 100
    maximum_variable_value = 200
    tournament_size = 3
    tournament_probability = 0.7
    crossover_probability = 0.5
    mutation_probability = 0.3
    number_of_generations = 100

    [maximum_fitness, best_individual] = ga.run_function_optimization(population_size, maximum_variable_value,
                                          tournament_size, tournament_probability, crossover_probability,
                                          mutation_probability, number_of_generations)