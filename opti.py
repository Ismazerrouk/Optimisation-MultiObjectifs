from pymoo.core.problem import Problem
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.util.ref_dirs.energy_layer import LayerwiseRieszEnergyReferenceDirectionFactory
from pymoo.indicators.hv import HV
# from pymoo.operators.crossover.pntx import TwoPointCrossover
# from pymoo.operators.mutation.bitflip import BitflipMutation
# from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
import numpy as np
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
n_tasks = 10
n_nodes = 5
C = np.random.rand(n_tasks, n_nodes)
B = np.random.rand(n_tasks, n_nodes)

problem = TaskOffloadingProblem(C, B)

# Population size and number of generations
POP_SIZE = 100  # Increased from 100
N_GEN = 500  # Increased from 500

# Mutation and crossover probabilities
crossover_prob = 0.9  # Crossover probability
mutation_prob = 0.1  # Mutation probability


# Define the algorithm parameters

ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=12)

# Implementation of the RVEA algorithm
algorithm = RVEA(ref_dirs)

# Execute the optimization with the updated number of generations
res = minimize(
    problem,
    algorithm,
    ('n_gen', N_GEN),
    seed=1,
    verbose=False
)


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
scatter_energy.show()

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
scatter_uniform.show()

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
scatter_multi_layer.show()

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
scatter_layer_energy.show()

# Calculate performance metrics
#ref_point = np.max(problem.pareto_front(), axis=0) * 1.1
ref_point = np.array([1.2, 1.2])
ind = HV(ref_point=ref_point)
print("HV", ind(A))


igd = IGD(problem.pareto_front())
igd_value = igd.calc(res.F)
print(f"Inverted Generational Distance: {igd_value}")

# A function to save the plots to files
def save_plot(figure, title, directory=PLOTS_DIR):
    filename = f"{title.replace(' ', '_').lower()}.png"
    filepath = os.path.join(directory, filename)
    figure.save(filepath)
    print(f"Plot saved to {filepath}")

# Visualize the results
plot = Scatter(title="RVEA Result")
plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
plot.add(res.F, facecolor="none", edgecolor="red")
save_plot(plot, "RVEA Result")