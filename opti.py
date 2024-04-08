from pymoo.core.problem import Problem
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.util.ref_dirs.energy_layer import LayerwiseRieszEnergyReferenceDirectionFactory
from pymoo.indicators.hv import HV
from pymoo.indicators.gd import GD
from pymoo.indicators.igd import IGD
# from pymoo.operators.crossover.pntx import TwoPointCrossover
# from pymoo.operators.mutation.bitflip import BitflipMutation
# from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
import numpy as np
import matplotlib.pyplot as plt
import os

# Define the directory for saving plots
PLOTS_DIR = "/home/mikealpharomeo/Downloads"

# Ensure the directory exists
os.makedirs(PLOTS_DIR, exist_ok=True)

#TODO: redefine the problem's class TaskOffloadingProblem
class TaskOffloadingProblem(Problem):
    """
    Classe du problème pour le délestage de tâches dans un environnement informatique distribué.
    L'objectif est d'optimiser le coût et le temps d'exécution des tâches sur différents nœuds.
    """
    def __init__(self, C, B):
        self.C = C  # Cost of executing tasks on nodes
        self.B = B  # Execution time of tasks on nodes
        n_tasks, n_nodes = C.shape
        super().__init__(n_var=n_tasks * n_nodes, n_obj=2, n_constr=n_tasks, xl=0, xu=1, elementwise_evaluation=False)

    def _evaluate(self, X, out, *args, **kwargs):
        n_tasks, n_nodes = self.C.shape

        # Initialize objective values
        f1, f2 = np.zeros(X.shape[0]), np.zeros(X.shape[0])

        # Calculate objectives
        for i in range(n_tasks):
            for j in range(n_nodes):
                idx = i * n_nodes + j
                f1 += X[:, idx] * self.B[i, j]
                f2 += X[:, idx] * self.C[i, j] * self.B[i, j]

        # Assign objectives to the output dictionary
        out["F"] = np.column_stack([f1, f2])

        # Initialize and calculate constraints
        G = np.zeros((X.shape[0], n_tasks))
        for i in range(n_tasks):
            for j in range(n_nodes):
                idx = i * n_nodes + j
                G[:, i] += X[:, idx]
        out["G"] = G - 1


# Define your problem instance with the given C and B matrices
n_tasks = 100
n_nodes = 10
C = np.random.rand(n_tasks, n_nodes)
B = np.random.rand(n_tasks, n_nodes)

problem = TaskOffloadingProblem(C, B)

# Population size and number of generations
POP_SIZE = 300
N_GEN = 500

# Mutation and crossover probabilities
crossover_prob = 0.9  # Crossover probability
mutation_prob = 0.1  # Mutation probability


# Calcul de la frontière de Pareto

def pareto_front():
    # Define the function for the second objective based on the first objective
    # This should be adapted based on your problem's objectives
    f2 = lambda f1: - ((f1/100) ** 0.5 - 1)**2
    F1_a, F1_b = np.linspace(1, 16, 300), np.linspace(36, 81, 300)
    F2_a, F2_b = f2(F1_a), f2(F1_b)

    # Combine the objective values into a 2D array
    return np.column_stack([np.concatenate([F1_a, F1_b]), np.concatenate([F2_a, F2_b])])

def save_plot(figure, title, pareto_front, directory=PLOTS_DIR):
    # Add the Pareto front to the plot
    figure.add(pareto_front, label="Pareto-front", color="red")

    # Save the plot to a file
    filename = f"{title.replace(' ', '_').lower()}.png"
    filepath = os.path.join(directory, filename)
    figure.save(filepath)
    print(f"Plot saved to {filepath}")

pf = pareto_front()

# Riesz s-Energy
ref_dirs_energy = get_reference_directions("energy", 2, 90, seed=1)
algorithm_energy = RVEA(ref_dirs_energy)
res_energy = minimize(
    problem,
    algorithm_energy,
    ('n_gen', N_GEN),
    seed=1,
    verbose=False
)
scatter_energy = Scatter(title="Riesz s-Energy")
scatter_energy.add(ref_dirs_energy)
save_plot(scatter_energy, "Riesz s-Energy", pf)

# Calculate the performance indicator
hv_energy = HV(ref_point=np.array([1.1, 1.1]))
hv_value_energy = hv_energy.do(res_energy.F)
print("Riesz s-Energy Hypervolume: ", hv_value_energy)


# Das-Dennis
ref_dirs_uniform = get_reference_directions("uniform", 2, n_partitions=12)
algorithm_uniform = RVEA(ref_dirs_uniform)
res_uniform = minimize(
    problem,
    algorithm_uniform,
    ('n_gen', N_GEN),
    seed=1,
    verbose=False
)
scatter_uniform = Scatter(title="Das-Dennis")
scatter_uniform.add(ref_dirs_uniform)
pf = pareto_front()
save_plot(scatter_uniform, "Das-Dennis")

# Calculate the performance indicator
hv_uniform = HV(ref_point=np.array([1.1, 1.1]))
hv_value_uniform = hv_uniform.do(res_uniform.F)
print("Das-Dennis Hypervolume: ", hv_value_uniform)



# Multi-layer Approach
ref_dirs_multi_layer = get_reference_directions(
    "multi-layer",
    get_reference_directions("das-dennis", 2, n_partitions=12, scaling=1.0),
    get_reference_directions("das-dennis", 2, n_partitions=12, scaling=0.5)
)
algorithm_multi_layer = RVEA(ref_dirs_multi_layer)
res_multi_layer = minimize(
    problem,
    algorithm_multi_layer,
    ('n_gen', N_GEN),
    seed=1,
    verbose=False
)
scatter_multi_layer = Scatter(title="Multi-layer Approach")
scatter_multi_layer.add(ref_dirs_multi_layer)
pf = pareto_front()
save_plot(scatter_multi_layer, "Multi-layer Approach")

# Calculate the performance indicator
hv_multi_layer = HV(ref_point=np.array([1.1, 1.1]))
hv_value_multi_layer = hv_multi_layer.do(res_multi_layer.F)
print("Multi-layer Approach Hypervolume: ", hv_value_multi_layer)



# LayerwiseRieszEnergyReferenceDirectionFactory
fac = LayerwiseRieszEnergyReferenceDirectionFactory(2, [9, 5, 2, 1])
ref_dirs_layer_energy = fac.do()
algorithm_layer_energy = RVEA(ref_dirs_layer_energy)
res_layer_energy = minimize(
    problem,
    algorithm_layer_energy,
    ('n_gen', N_GEN),
    seed=1,
    verbose=False
)
scatter_layer_energy = Scatter(title="LayerwiseRieszEnergyReferenceDirectionFactory")
scatter_layer_energy.add(ref_dirs_layer_energy)
pf = pareto_front()
save_plot(scatter_layer_energy, "LayerwiseRieszEnergyReferenceDirectionFactory")

# Calculate the performance indicator
hv_layer_energy = HV(ref_point=np.array([1.1, 1.1]))
hv_value_layer_energy = hv_layer_energy.do(res_layer_energy.F)
print("LayerwiseRieszEnergyReferenceDirectionFactory Hypervolume: ", hv_value_layer_energy)