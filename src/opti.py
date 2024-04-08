from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
import numpy as np


#TODO: redefine the problem's class TaskOffloadingProblem
class TaskOffloadingProblem(Problem):
    def __init__(self, C, B):
        self.C = C  # Cost of executing tasks on nodes
        self.B = B  # Execution time of tasks on nodes
        n_tasks, n_nodes = C.shape
        # Define problem properties
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
algorithm = NSGA2(
    pop_size=POP_SIZE,
    n_offsprings=20,  # Consider adjusting this as well
    sampling=BinaryRandomSampling(),
    crossover=TwoPointCrossover(prob=crossover_prob),
    mutation=BitflipMutation(prob=mutation_prob),
    eliminate_duplicates=True
)

# Execute the optimization with the updated number of generations
res = minimize(
    problem,
    algorithm,
    ('n_gen', N_GEN),
    seed=1,
    verbose=False
)

# Visualize the results
plot = Scatter()
plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
plot.add(res.F, facecolor="none", edgecolor="red")
plot.show()
