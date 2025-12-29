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
    maximum_variable_value = [2, 2, 2, 10, 2,
                              2, 2, 2, 2, 2,
                              2, 2, 2, 2, 2,
                              2, 2, 2, 2, 2,
                              2, 2, 2, 2, 2,
                              2, 2, 2, 2, 2,
                              2, 2, 2]
    tournament_size = 3
    tournament_probability = 0.7
    crossover_probability = 0.5
    mutation_probability = 0.3
    number_of_generations = 100

    population = ga.initialize_population(population_size, maximum_variable_value)

    for generation_index in range(number_of_generations):
        print(f"Generation: {generation_index}")

        fitness_list, individuals_decoded = ga.evaluate_population(population, maximum_variable_value)
        population, maximum_fitness, best_individual= ga.one_pop_ga(population, population_size,
                                                tournament_size, tournament_probability, crossover_probability,
                                                mutation_probability, fitness_list, individuals_decoded)