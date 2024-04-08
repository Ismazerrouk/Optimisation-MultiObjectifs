from pymoo.core.problem import Problem
from pymoo.factory get_sampling, get_crossover, get_mutation
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
import numpy as np


class TaskOffloadingProblem(Problem):
    def __init__(self, C, B):
        self.C = C  # Coûts d'exécution des tâches sur les nœuds
        self.B = B  # Temps d'exécution des tâches sur les nœuds
        n_tasks, n_nodes = C.shape
        super().__init__(n_var=n_tasks*n_nodes, n_obj=2, n_constr=n_tasks, xl=0, xu=1, elementwise_evaluation=True)

    def _evaluate(self, X, out, *args, **kwargs):
        X = X.reshape(self.C.shape)

        # Objectif 1: Minimiser le temps d'exécution total
        f1 = np.sum(X * self.B)

        # Objectif 2: Minimiser le coût total
        f2 = np.sum(X * self.C * self.B)

        out["F"] = np.column_stack([f1, f2])

        # Contraintes: Chaque tâche doit être assignée à exactement un nœud
        out["G"] = np.sum(X, axis=1) - 1
        
        
# Define your problem parameters
n_tasks = 10
n_nodes = 5
C = np.random.rand(n_tasks, n_nodes)  # Coûts d'exécution des tâches sur les nœuds
B = np.random.rand(n_tasks, n_nodes)  # Temps d'exécution des tâches sur les nœuds

problem = TaskOffloadingProblem(C, B)

algorithm = NSGA2(
    pop_size=40,
    n_offsprings=10,
    eliminate_duplicates=True
)

res = minimize(
    problem,
    algorithm,
    ("n_gen", 100),
    verbose=True,
    seed=1
)

plot = Scatter()
plot.add(res.F, color="red")
plot.show()
        
        