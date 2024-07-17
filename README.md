# Energy-efficient IoT Device Scheduling

This repository contains implementations of various optimization algorithms to enhance the scheduling of IoT devices for improved energy efficiency. The algorithms implemented are Genetic Algorithm (GA), Particle Swarm Optimization (PSO), Simulated Annealing (SA), Hill Climbing, and Ant Colony Optimization (ACO).

## Overview

The project aims to optimize the scheduling of IoT devices to reduce energy consumption and improve device utilization. Each algorithm is designed to find the best schedule for turning devices on or off based on their usage and priority. The provided code evaluates different schedules using a custom fitness function and compares the results of each optimization technique.

## Algorithms Implemented

- **Genetic Algorithm (GA)**: Uses crossover, mutation, and selection to evolve a population of schedules over generations.
- **Particle Swarm Optimization (PSO)**: Optimizes schedules by simulating a swarm of particles that adjust their velocities and positions based on personal and global best solutions.
- **Simulated Annealing (SA)**: Employs a probabilistic approach to explore and exploit solutions, accepting worse solutions with decreasing probability over time.
- **Hill Climbing**: Iteratively improves a solution by moving to neighboring solutions that have higher fitness.
- **Ant Colony Optimization (ACO)**: Mimics the behavior of ants searching for food to construct solutions and update pheromone trails based on the quality of solutions.

## Code Structure

- **Optimization Algorithms**:
  - `genetic_algorithm()`: Performs scheduling optimization using Genetic Algorithm.
  - `particle_swarm_optimization()`: Applies Particle Swarm Optimization for scheduling.
  - `simulated_annealing()`: Implements Simulated Annealing for optimizing schedules.
  - `hill_climbing()`: Uses Hill Climbing to find better schedules iteratively.
  - `ant_colony_optimization()`: Optimizes scheduling using Ant Colony Optimization.

- **Helper Functions**:
  - `evaluate_fitness(schedule)`: Evaluates the fitness of a schedule based on energy consumption.
  - `select_parents(population, fitness_scores)`: Selects parents for crossover in GA.
  - `crossover_and_mutate(parents)`: Applies crossover and mutation to generate new schedules in GA.
  - `update_velocities()`, `update_positions()`, `initialize_particles()`, `update_personal_best()`: Functions for PSO.
  - `get_neighbor()`, `accept_solution()`: Functions for Simulated Annealing.
  - `construct_solution()`, `choose_device()`, `update_pheromones()`: Functions for ACO.
