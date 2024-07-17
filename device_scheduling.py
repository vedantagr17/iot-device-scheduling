
import random
import numpy as np
import math

# Sample IoT device data
device_data = {
    'device1': {'usage': 5, 'priority': 4},
    'device2': {'usage': 3, 'priority': 3},
    'device3': {'usage': 8, 'priority': 2},
    'device4': {'usage': 4, 'priority': 1}
    
}

# Placeholder for AI-driven schedules
ai_schedules = []

# Genetic Algorithm (GA)
def genetic_algorithm(population_size=10, generations=100):
    # Initial population
    population = [[random.randint(0, 1) for _ in range(len(device_data))] for _ in range(population_size)]

    for generation in range(generations):
        # Evaluate fitness
        fitness_scores = [evaluate_fitness(schedule) for schedule in population]

        # Select parents based on fitness
        parents = select_parents(population, fitness_scores)

        # Crossover and mutation
        population = crossover_and_mutate(parents)

    # Return the best schedule
    best_schedule = population[np.argmax(fitness_scores)]
    ai_schedules.append(('Genetic Algorithm', best_schedule))
    return best_schedule

# Particle Swarm Optimization (PSO)
def particle_swarm_optimization(iterations=100, swarm_size=10):
    # Initialize particles
    particles = initialize_particles(swarm_size, len(device_data))

    # Initialize velocities
    velocities = np.zeros_like(particles)

    # Initialize personal and global bests
    personal_best = particles.copy()
    global_best_index = np.argmax([evaluate_fitness(p) for p in particles])
    global_best = personal_best[global_best_index]

    for x in range(iterations):
        # Update velocities and positions
        velocities = update_velocities(velocities, particles, personal_best, global_best)
        particles = update_positions(particles, velocities)

        # Update personal and global bests
        new_fitness = np.array([evaluate_fitness(p) for p in particles])
        personal_best = update_personal_best(particles, personal_best, new_fitness)

        # Update global best
        current_best_index = np.argmax(new_fitness)
        if new_fitness[current_best_index] > evaluate_fitness(global_best):
            global_best = particles[current_best_index]

    ai_schedules.append(('Particle Swarm Optimization', global_best))
    return global_best

# Simulated Annealing (SA)
def simulated_annealing(iterations=100, initial_temperature=100.0, cooling_rate=0.95):
    current_solution = [random.randint(0, 1) for _ in range(len(device_data))]
    best_solution = current_solution

    for _ in range(iterations):
        new_solution = get_neighbor(current_solution)
        
        current_energy = evaluate_fitness(current_solution)
        new_energy = evaluate_fitness(new_solution)

        if accept_solution(current_energy, new_energy, initial_temperature):
            current_solution = new_solution

        if evaluate_fitness(current_solution) > evaluate_fitness(best_solution):
            best_solution = current_solution

        initial_temperature *= cooling_rate

    ai_schedules.append(('Simulated Annealing', best_solution))
    return best_solution

# Evaluation function (fitness function)
def evaluate_fitness(schedule):
    total_energy = sum(device_data[f'device{i+1}']['usage'] * schedule[i] for i in range(1,len(device_data)))
    return -total_energy  # Negative because we want to minimize energy usage

# Genetic Algorithm helper functions
def select_parents(population, fitness_scores):
    selected_indices = np.random.choice(len(population), size=len(population), p=fitness_scores/np.sum(fitness_scores))
    return [population[i] for i in selected_indices]

def crossover_and_mutate(parents, mutation_rate=0.01):
    children = []

    for i in range(0, len(parents), 2):
        parent1, parent2 = parents[i], parents[i + 1]
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]

        # Mutation
        child1 = [bit ^ 1 if random.random() < mutation_rate else bit for bit in child1]
        child2 = [bit ^ 1 if random.random() < mutation_rate else bit for bit in child2]

        children.extend([child1, child2])

    return children

# Particle Swarm Optimization helper functions
def update_velocities(velocities, particles, personal_best, global_best, inertia=0.5, phi_p=0.5, phi_g=0.5):
    return inertia * velocities + phi_p * np.random.rand(*velocities.shape) * (personal_best - particles) + phi_g * np.random.rand(*velocities.shape) * (global_best - particles)

def update_positions(particles, velocities):
    return np.clip(particles + velocities, 0, 1)

def initialize_particles(swarm_size, schedule_length):
    return np.random.randint(0, 2, size=(swarm_size, schedule_length))

def update_personal_best(particles, personal_best, fitness_values):
    greater_fitness = fitness_values > np.array([evaluate_fitness(p) for p in personal_best])
    updated_personal_best = personal_best.copy()
    for i in range(len(personal_best)):
        if greater_fitness[i]:
            updated_personal_best[i] = particles[i]
    return updated_personal_best.astype(int)
# Simulated Annealing helper functions
def get_neighbor(solution):
    neighbor = solution.copy()
    index_to_mutate = random.randint(0, len(solution) - 1)
    neighbor[index_to_mutate] = 1 - neighbor[index_to_mutate]  # Flip the bit
    return neighbor

def accept_solution(current_energy, new_energy, temperature):
    if new_energy > current_energy:
        return True
    return random.random() < math.exp((new_energy - current_energy) / temperature)

def hill_climbing(max_iterations=100):
    current_solution = [random.randint(0, 1) for _ in range(len(device_data))]
    current_energy = evaluate_fitness(current_solution)

    for _ in range(max_iterations):
        neighbor = get_neighbor(current_solution)
        neighbor_energy = evaluate_fitness(neighbor)

        if neighbor_energy > current_energy:
            current_solution = neighbor
            current_energy = neighbor_energy

    ai_schedules.append(('Hill Climbing', current_solution))
    return current_solution

# Ant Colony Optimization (ACO)
def ant_colony_optimization(iterations=100, ants=10, alpha=1.0, beta=2.0, evaporation_rate=0.5, pheromone_init=1.0):
    num_devices = len(device_data)
    pheromones = np.ones((num_devices, num_devices)) * pheromone_init

    best_solution = None
    best_energy = float('-inf')

    for _ in range(iterations):
        solutions = []

        for ant in range(ants):
            solution = construct_solution(pheromones, alpha, beta)
            solutions.append((solution, evaluate_fitness(solution)))

        # Update pheromones
        update_pheromones(pheromones, solutions, evaporation_rate)

        # Update best solution
        for solution, energy in solutions:
            if energy > best_energy:
                best_solution = solution
                best_energy = energy

    ai_schedules.append(('Ant Colony Optimization', best_solution))
    return best_solution

def construct_solution(pheromones, alpha, beta):
    num_devices = len(device_data)
    solution = [0] * num_devices
    visited = set()

    for _ in range(num_devices):
        current_device = choose_device(pheromones, solution, visited, alpha, beta)
        solution[current_device] = 1
        visited.add(current_device)

    return solution

def choose_device(pheromones, solution, visited, alpha, beta):
    num_devices = len(device_data)
    available_devices = [d for d in range(num_devices) if d not in visited]

    if not available_devices:
        # If all devices are visited, choose a random device
        chosen_device = np.random.choice(num_devices)
    else:
        probabilities = np.array([pheromones[solution[-1]][device] ** alpha * (1 / max(1, evaluate_fitness([device] + solution))) ** beta for device in available_devices])
        total_probability = np.sum(probabilities)

        if total_probability == 0:
            # If all probabilities are zero, choose a random device
            chosen_device = np.random.choice(available_devices)
        else:
            probabilities /= total_probability
            chosen_device = np.random.choice(available_devices, p=probabilities)

    return chosen_device

def update_pheromones(pheromones, solutions, evaporation_rate):
    pheromones *= evaporation_rate

    for solution, energy in solutions:
        for i in range(len(solution) - 1):
            pheromones[i][i + 1] += 1.0 / energy  # Deposit pheromones based on energy

# Example usage
genetic_algorithm_schedule = genetic_algorithm()
particle_swarm_optimization_schedule = particle_swarm_optimization()
simulated_annealing_schedule = simulated_annealing()
hill_climbing_schedule = hill_climbing()
ant_colony_optimization_schedule = ant_colony_optimization()

# Print the resulting schedules
print("Genetic Algorithm Schedule:", genetic_algorithm_schedule)
print("Particle Swarm Optimization Schedule:", particle_swarm_optimization_schedule)
print("Simulated Annealing Schedule:", simulated_annealing_schedule)
print("Hill Climbing Schedule:", hill_climbing_schedule)
print("Ant Colony Optimization Schedule:", ant_colony_optimization_schedule)
