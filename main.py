#NaghamMassis - 1220149
#Al=zaharaNassif - 1220168
import math
import random
import copy
import matplotlib.pyplot as plt
# -------------------------- Variables --------------------------
vehiclesWeight=[]
numV=0
numP=0
packages={}
shop_location=(0, 0)
# -------------------------- Read from file --------------------------
def readfile():
    global numV, numP
    with open("input.txt", 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            if not parts[0]:
                continue
            if parts[0].startswith('v'):
                vechileCapacity=int(parts[1])
                vehiclesWeight.append(vechileCapacity)
                numV+=1
            if parts[0].startswith('p'):
                arr = []
                xcoor=int(parts[1])
                arr.append(xcoor)
                ycoor=int(parts[2])
                arr.append(ycoor)
                waight=int(parts[3])
                arr.append(waight)
                prior=int(parts[4])
                arr.append(prior)
                packages[numP]=arr
                numP+=1
    print("Vehicles:", vehiclesWeight)
    print("Number of vehicles is:", numV)
    print("Packages:",packages)
    print("Number of packages is:", numP)
# -------------------------- Distance Calculation --------------------------
def euclidean_distance(point1, point2):
    delta_x = point2[0] - point1[0]
    delta_y = point2[1] - point1[1]
    distance = math.sqrt(delta_x ** 2 + delta_y ** 2)
    return distance
# -------------------------- Total Route Distance --------------------------
def total_distance(state):
    total = 0
    for v_id in state:
        route = [shop_location] + [tuple(packages[p_id][0:2]) for p_id in state[v_id]] + [shop_location]
        for i in range(len(route)-1):
            total += euclidean_distance(route[i], route[i+1])
    return total
# -------------------------- Priority Reward --------------------------
def priority_score(state):
    score = 0
    for v in state:
        route = state[v]
        n = len(route)
        for i in range(n):
            p_id = route[i]
            priority = packages[p_id][3]
            score += (n - i) * (6 - priority)
    return score
# -------------------------- Objective Function --------------------------
def objective_function(state):
    dist = total_distance(state)
    score = priority_score(state)
    return dist - score
# -------------------------- State Validation --------------------------
def is_valid(state):
    for v_id in state:
        total_weight=0
        for p in state[v_id]:
            total_weight += packages[p][2]
        if total_weight > vehiclesWeight[v_id]:
            return False
    return True
# -------------------------- Generate Initial State --------------------------
def generate_initial_state():
    state = {}
    for v_id in range(numV):
        state[v_id] = []
    pkg_ids = list(packages.keys())

    random.shuffle(pkg_ids)

    for pkg_id in pkg_ids:
        attempts = 0
        placed = False
        while not placed and attempts < 100:
            v_id = random.randint(0, numV - 1)#get random vehicle
            current_weight = 0
            attempts += 1
            if not state[v_id] and packages[pkg_id][2] <= vehiclesWeight[v_id]:
                state[v_id].append(pkg_id)
                placed = True
            else:
                for p in state[v_id]:
                    current_weight +=packages[p][2]
                if current_weight + packages[pkg_id][2] <= vehiclesWeight[v_id]:
                    state[v_id].append(pkg_id)
                    placed = True
        if not placed:
            print(f" Could not place package {pkg_id} due to weight constraints.Dropping package with lowest priority.")
            current_priority= packages[pkg_id][3]
            pkg_weight = packages[pkg_id][2]
            swapped = False
            for v_id in range(numV):
                for existing_pkg_id in state[v_id]:
                    current_total_weight=0
                    existing_priority = packages[existing_pkg_id][3]
                    existing_weight = packages[existing_pkg_id][2]
                    for p in state[v_id]:
                        current_total_weight += packages[p][2]
                    if (existing_priority > current_priority) and \
                            (current_total_weight - existing_weight + pkg_weight <= vehiclesWeight[v_id]):
                        state[v_id].remove(existing_pkg_id)
                        state[v_id].append(pkg_id)
                        swapped = True
                        print(f"Swapped out package {existing_pkg_id} (priority {existing_priority}) "
                              f"to place {pkg_id} (priority {current_priority}) in vehicle {v_id}")
                        packages.pop(existing_pkg_id)
                        break
                if swapped:
                    break
            if not swapped:
                print(f"Could not place package {pkg_id} (priority {current_priority}) even after attempting swaps. Dropping it.")
                packages.pop(pkg_id)
                if not packages:
                    print("no more package to deleveir! exit")
                    exit(0)
    print("The initial state is ",state)
    print(f"Total Distance for initial state: {total_distance(state):.2f}")
    return state
# -------------------------- Generate Next State --------------------------
def generate_next(state):
    new_state = copy.deepcopy(state)
    if numV == 1:
        vehicle_id = 0
        route = new_state[vehicle_id]
        if len(route) >= 2:  # need at least 2 packages to swap
            i, j = random.sample(range(len(route)), 2)
            route[i], route[j] = route[j], route[i]
    else:
        from_v, to_v = random.sample(range(numV), 2)
        if new_state[from_v]:
            p_idx = random.randint(0, len(new_state[from_v]) - 1)
            pkg = new_state[from_v].pop(p_idx)
            to_weight = 0
            for p in new_state[to_v]:
                to_weight += packages[p][2]
            if to_weight + packages[pkg][2] <= vehiclesWeight[to_v]:
                new_state[to_v].append(pkg)
            else:
                new_state[from_v].append(pkg)
    return new_state
# -------------------------- Simulated annealing --------------------------
def simulated_annealing():
    T = 1000
    cooling_rate = 0.9
    stopping_temp = 1
    iterations_per_temp = 100
    current = generate_initial_state()
    best = copy.deepcopy(current)
    while T >= stopping_temp:
        for _ in range(iterations_per_temp):
            next = generate_next(current)
            if not is_valid(next):
                continue
            current_cost = objective_function(current)
            next_cost = objective_function(next)
            delta = next_cost - current_cost
            if delta < 0 or random.random() < math.exp(-delta / T):
                current = next
                if objective_function(current) < objective_function(best):
                    best = copy.deepcopy(current)
        T *= cooling_rate
    return best
# -------------------------- Genome Initialization --------------------------
def generate_genome(num_pkgs, num_vehicles):
    genome = [[] for _ in range(num_vehicles)]
    for pkg_id in range(num_pkgs):
        assigned_vehicle = random.randint(0, num_vehicles - 1)
        genome[assigned_vehicle].append(pkg_id)
    return genome
# -------------------------- Population Initialization --------------------------
def generate_initial_solutions(num_solutions, packages, vehicle_capacities):
    population = []  # Final list of valid initial solutions
    warning_printed = False  # Only print warning once if any packages get dropped

    # Sort packages by priority (lowest priority first)
    sorted_packages = sorted(packages.items(), key=lambda item: item[1][3])

    dropped_packages_overall = set()  # Track all dropped packages across all solutions

    for _ in range(num_solutions):
        # Initialize an empty solution: each vehicle has an empty list of packages
        solution = {v_id: [] for v_id in range(len(vehicle_capacities))}

        # Copy of vehicle capacities to track usage as we assign packages
        remaining_capacity = vehicle_capacities.copy()

        dropped_packages = set()  # Track packages that couldn't be assigned in this solution

        # Attempt to assign each package to a vehicle
        for pkg_id, (x, y, weight, priority) in sorted_packages:
            assigned = False

            # Shuffle vehicle order for randomness
            vehicle_ids = list(range(len(vehicle_capacities)))
            random.shuffle(vehicle_ids)

            # Try to assign the package to any vehicle with enough remaining capacity
            for v_id in vehicle_ids:
                if remaining_capacity[v_id] >= weight:
                    solution[v_id].append(pkg_id)
                    remaining_capacity[v_id] -= weight
                    assigned = True
                    break

            # If no vehicle could take the package, mark it as dropped
            if not assigned:
                dropped_packages.add(pkg_id)

        # Print warning only once if any packages are dropped due to capacity limits
        if dropped_packages and not warning_printed:
            print("Warning: Some packages were dropped due to capacity limits.")
            for pkg_id in dropped_packages:
                print(f" - Dropped package {pkg_id} (priority {packages[pkg_id][3]}, weight {packages[pkg_id][2]})")
            warning_printed = True

        # Update global dropped package tracker
        dropped_packages_overall.update(dropped_packages)

        # Only keep the solution if at least one package is assigned
        if any(solution[v] for v in solution):
            population.append(solution)

    return population  # Return list of initial valid solutions
# -------------------------- Selection --------------------------
def selection_pair(population):
    # Step 1: Evaluate the objective (cost) of each individual in the population
    fitness_raw = [objective_function(ind) for ind in population]

    # Step 2: Convert objective values into fitness scores
    # Lower objective function values are better, so we invert them
    # Add a small constant (1e-6) to avoid division by zero
    fitness = [1 / (f + 1e-6) for f in fitness_raw]

    # Step 3: Normalize fitness scores into probabilities for selection
    total_fitness = sum(fitness)
    probabilities = [f / total_fitness for f in fitness]

    # Step 4: Randomly select two parents using roulette wheel selection
    # Individuals with higher fitness (lower objective cost) are more likely to be selected
    parents = random.choices(population=population, weights=probabilities, k=2)

    return parents
# -------------------------- Crossover --------------------------
def crossover(parent1, parent2, vehicle_capacities, packages, max_retries=10):
    # Helper: Flatten a solution into a package-to-vehicle mapping
    def flatten(state):
        return {pkg_id: v_id for v_id, pkgs in state.items() for pkg_id in pkgs}

    # Helper: Rebuild a solution (dictionary) from package-to-vehicle mapping
    def rebuild(mapping):
        state = {}
        for pkg_id, v_id in mapping.items():
            state.setdefault(v_id, []).append(pkg_id)
        return state

    # Helper: Check if all vehicles in a state stay within their weight capacities
    def is_within_capacity(state):
        for v_id, pkg_ids in state.items():
            total_weight = sum(packages[pkg_id][2] for pkg_id in pkg_ids)
            if total_weight > vehicle_capacities[v_id]:
                return False
        return True

    # Convert each parent into a package-to-vehicle mapping
    p1_map = flatten(parent1)
    p2_map = flatten(parent2)

    # Identify common packages that exist in both parents (safe to swap)
    common_pkgs = list(set(p1_map.keys()) & set(p2_map.keys()))

    # Retry crossover multiple times to increase the chance of generating valid children
    for _ in range(max_retries):
        # Need at least 2 common packages to perform a meaningful swap
        if len(common_pkgs) > 1:
            # Randomly decide how many common packages to swap (at least one)
            num_to_swap = random.randint(1, len(common_pkgs) - 1)

            # Randomly choose which packages to swap
            pkgs_to_swap = random.sample(common_pkgs, num_to_swap)

            # Copy parents' mappings to create temporary child mappings
            temp_p1 = p1_map.copy()
            temp_p2 = p2_map.copy()

            # Swap the vehicle assignments for the selected packages
            for pkg_id in pkgs_to_swap:
                temp_p1[pkg_id], temp_p2[pkg_id] = temp_p2[pkg_id], temp_p1[pkg_id]

            # Rebuild the full solution (vehicle -> list of packages)
            child1 = rebuild(temp_p1)
            child2 = rebuild(temp_p2)

            # Only return the children if both are valid (within capacity constraints)
            if is_within_capacity(child1) and is_within_capacity(child2):
                return child1, child2
        else:
            # If not enough common packages, fallback to returning parents
            return parent1, parent2

    # If no valid crossover is found within retries, return None
    return None
# -------------------------- Mutation --------------------------
def mutate(state, vehicle_capacities, packages, mutation_rate=0.1):
    # Make a deep copy of the current solution so we don’t change the original
    mutated = copy.deepcopy(state)

    # Create a list of all package IDs currently assigned in the solution
    all_pkg_ids = [pkg for pkgs in mutated.values() for pkg in pkgs]

    # Iterate over every package to decide if it should be mutated (i.e., reassigned to another vehicle)
    for pkg_id in all_pkg_ids:
        # Apply mutation probabilistically based on mutation rate
        if random.random() < mutation_rate:
            # Find the vehicle that currently carries this package
            current_vehicle = next(v_id for v_id, pkgs in mutated.items() if pkg_id in pkgs)

            # Temporarily remove the package from its current vehicle
            mutated[current_vehicle].remove(pkg_id)

            # Find vehicles that can accommodate the package based on remaining capacity
            eligible = [
                v_id for v_id in mutated
                if sum(packages[p][2] for p in mutated[v_id]) + packages[pkg_id][2] <= vehicle_capacities[v_id]
            ]

            # If there’s at least one vehicle that can take the package
            if eligible:
                # Randomly choose one of the eligible vehicles and assign the package to it
                new_vehicle = random.choice(eligible)
                mutated[new_vehicle].append(pkg_id)
            else:
                # If no eligible vehicle is found, return the package to its original vehicle
                mutated[current_vehicle].append(pkg_id)

    # Return the mutated solution (could be the same as original if no mutation happened)
    return mutated
# -------------------------- Evolution Process --------------------------
def genetic(population, vehicle_capacities, packages, generations=500, mutation_rate=0.1):
    # If no valid initial solutions were generated, skip evolution
    if not population:
        print("No valid initial population generated. Skipping evolution.")
        return None, None

    # Initialize tracking for the best solution found during evolution
    best_solution = None
    best_generation = 0
    best_objective_value = float('inf')  # Smaller objective value is assumed to be better

    # Main evolution loop, runs for a specified number of generations
    for generation in range(generations):
        next_population = []  # Will store the next generation of solutions

        # Check again in case population was emptied due to some issue
        if not population:
            print("Population is empty, skipping evolution.")
            break

        # Create next generation by repeatedly selecting parents, crossing over, and mutating
        while len(next_population) < len(population):
            # Select two parent solutions from the current population
            parent1, parent2 = selection_pair(population)

            # Perform crossover to produce offspring (2 children)
            offspring = crossover(parent1, parent2, vehicle_capacities, packages)

            if offspring:
                child1, child2 = offspring

                # Mutate the offspring with a given mutation rate
                child1 = mutate(child1, vehicle_capacities, packages, mutation_rate)
                child2 = mutate(child2, vehicle_capacities, packages, mutation_rate)

                # Add mutated offspring to the next generation
                next_population.extend([child1, child2])

        # Replace current population with the newly created one (trimming if too many)
        population = next_population[:len(population)]

        # Find the best solution in the current generation
        generation_best = min(population, key=objective_function)
        generation_best_objective = objective_function(generation_best)

        # If this generation produced a better solution than before, update tracking variables
        if generation_best_objective < best_objective_value:
            best_solution = generation_best
            best_generation = generation + 1  # +1 because generation is 0-indexed
            best_objective_value = generation_best_objective

    # After all generations, return the best solution and the generation it was found
    return best_solution, best_generation
# -------------------------- Plot --------------------------
def simple_vehicle_plot(best_state):
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    depot = (0, 0)
    total_distance = 0
    total_load = 0
    plt.figure(figsize=(9, 7))
    routes = list(best_state.values())
    for i, route in enumerate(routes):
        color = colors[i % len(colors)]
        x_points = [depot[0]]
        y_points = [depot[1]]
        load = 0
        dist = 0
        path_labels = []
        for pkg_id in route:
            x, y, weight, _ = packages[pkg_id]
            x_points.append(x)
            y_points.append(y)
            path_labels.append(pkg_id)
            load += weight
        x_points.append(depot[0])
        y_points.append(depot[1])
        for j in range(len(x_points) - 1):
            plt.arrow(x_points[j], y_points[j],
                      x_points[j + 1] - x_points[j], y_points[j + 1] - y_points[j],
                      color=color, length_includes_head=True, head_width=1.5, alpha=0.6)
            dist += euclidean_distance((x_points[j], y_points[j]), (x_points[j + 1], y_points[j + 1]))
        total_distance += dist
        total_load += load
        for step, pkg_id in enumerate(route):
            x, y, _, _ = packages[pkg_id]
            plt.text(x + 1, y + 1, f"{step + 1}. P{pkg_id}", fontsize=9, color=color)
        plt.plot([], [], color=color, label=f"Vehicle {i} | Load: {load}kg | Distance: {dist:.1f}")
    plt.title("Vehicle Routes and Delivery Order")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.legend()
    summary = f"Vehicles: {len(routes)}\n Total Load: {total_load}kg\nTotal Distance: {total_distance:.1f}"
    plt.gcf().text(0.7, 0.15, summary, bbox=dict(facecolor='white', alpha=0.7))
    plt.show()
# -------------------------- Main --------------------------
def main():
    choice = 0
    readfile()
    while True:
        print("Which algorithm to use to generate the solution:")
        print("1- Simulated annealing algorithm.")
        print("2- Genetic algorithm.")
        print("3- Exit.")
        try:
            choice = int(input())
        except ValueError:
            print("Invalid input. Please enter 1 or 2.")
            continue
        if choice == 1:
            best_state = simulated_annealing()

            print("\nBest vehicle assignments:")
            for v_id in best_state:
                assigned_p = best_state[v_id]
                total_w = sum(packages[p][2] for p in assigned_p)
                print(f"Vehicle {v_id} (Capacity: {vehiclesWeight[v_id]}, Load: {total_w}): Packages {assigned_p}")

            print(f"\nTotal Distance: {total_distance(best_state):.2f}")
            simple_vehicle_plot(best_state)
            continue
        elif choice == 2:
            initial_population = generate_initial_solutions(50, packages, vehiclesWeight)
            best_solution, best_generation = genetic(initial_population, vehiclesWeight, packages, generations=500)
            if best_solution:
                print("\nBest vehicle assignments:")
                for v_id in best_solution:
                    assigned_p = best_solution[v_id]
                    total_w = sum(packages[p][2] for p in assigned_p)
                    print(f"Vehicle {v_id} (Capacity: {vehiclesWeight[v_id]}, Load: {total_w}): Packages {assigned_p}")
                total_dist = total_distance(best_solution)
                print(f"Total Distance: {total_dist:.2f}")
                print(f"Best solution found in Generation {best_generation}")
                simple_vehicle_plot(best_solution)
            else:
                print("No valid solution found.")
            continue
        elif choice == 3:
            exit()


if __name__ == "__main__":
    main()