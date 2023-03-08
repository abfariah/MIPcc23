import sys
import pyscipopt as scip
from scipy import sparse
import numpy as np
from typing import Tuple
from InfluenceBranching import InfluenceBranching

sys.path.append("../")


class SparsInfluenceBranching(InfluenceBranching):
    """
    Influence branching rule implementation for
    high dimensionnal sparse problems (rhs_obj 1
    for example), and other sparse hiddens series
    """

    def __init__(self, graph: str, max_depth: int) -> None:
        super(SparsInfluenceBranching, self).__init__(graph, max_depth)
        self.is_sparse = True
        self.never_normalize = True

    def get_true_A(self) -> np.array:
        """
        Partially vectorized version
        Retrieves matrix A
        """
        A_lines = []
        self.vars_dict = self.get_vars_dict()
        for cons in self.linear_constraints:
            constraint_dict = self.model.getValsLinear(cons)
            keys = np.array(list(constraint_dict.keys()))
            values = np.array(list(constraint_dict.values()))
            idx_var = [self.vars_dict[f"{key}"] for key in keys]
            idx_row = [0] * len(idx_var)
            A_line = sparse.csr_matrix((values, (idx_row, idx_var)), shape=(1, self.n))
            A_lines += [A_line]
        if len(A_lines) > 0:
            A = sparse.vstack(A_lines)
        else:
            A = sparse.csr_matrix((0, self.n))
        return A

    def get_matrices(self) -> Tuple[np.array, np.array, np.array]:
        """
        From a pyscipopt model, get the matrices c, A and b
        such that the program is written max cx s.t. Ax <= b
        """
        ## Global features
        self.n, self.m = self.model.getNVars(), self.model.getNConss()
        self.variables = self.model.getVars(True)
        variable_types = np.array([var.vtype() for var in self.variables])
        binary_condition = variable_types == "BINARY"
        integer_condition = variable_types == "INTEGER"
        self.idx_integer = np.where(binary_condition | integer_condition)[0]
        self.constraints = self.model.getConss()
        self.linear_constraints = [cons for cons in self.constraints if cons.isLinear()]
        self.m_linear = len(self.linear_constraints)
        assert self.n == len(self.variables)
        assert self.m == len(self.constraints)
        self.test_series(binary_condition)

        ## A, b, c
        c = sparse.csr_matrix([var.getObj() for var in self.variables])
        ## Modifs - Normalize c
        self.c_ = np.abs(c.toarray())
        self.c_ = 1 + (self.c_ / np.std(self.c_))

        A = self.get_true_A()
        return A, c

    def get_W(self) -> np.array:
        A_bool = self.A.copy() * 1.0
        A_bool[A_bool != 0] = 1
        if self.graph == "count":
            W = A_bool.transpose() @ A_bool

        elif self.graph == "base":  ## named binary_graph in Marc's manuscript
            W = A_bool.transpose() @ A_bool
            W[W != 0] = 1

        elif self.graph == "countdual":
            _, y_dual = self.get_primal_dual_values()
            y = sparse.csr_matrix(((np.abs(y_dual) > 1e-3) * 1.0).reshape(-1, 1))
            A_bool = A_bool.multiply(y)
            W = A_bool.T @ A_bool * 1.0

        elif self.graph == "bindual":
            _, y_dual = self.get_primal_dual_values()
            y = sparse.csr_matrix(((np.abs(y_dual) > 1e-3) * 1.0).reshape(-1, 1))
            A_bool = A_bool.multiply(y)
            W = A_bool.T @ A_bool * 1.0
            W[W != 0] = 1

        elif self.graph == "dual":
            _, y_dual = self.get_primal_dual_values()
            y = sparse.csr_matrix(np.sqrt(np.abs(y_dual)).reshape(-1, 1))
            A_bool = A_bool.multiply(y)
            W = A_bool.T @ A_bool * 1.0

        elif self.graph in ["auxiliary", "adversarial"]:
            x_primal, y_dual = self.get_primal_dual_values()
            lb = np.array([var.getLbLocal() for var in self.variables])
            ub = np.array([var.getUbLocal() for var in self.variables])
            bound_idx = [i for i in range(self.n) if i not in self.idx_integer]
            lb[bound_idx] = x_primal[bound_idx]
            ub[bound_idx] = x_primal[bound_idx]

            y = sparse.csr_matrix((np.abs(y_dual) * 1.0).reshape(-1, 1))
            ix = np.where(ub == lb)[0]
            denom = ub - lb
            denom[ix] = 1
            slack = 0.25 + (x_primal - lb) * (ub - x_primal) / denom
            slack[ix] = 0.25
            slack = sparse.csr_matrix(slack.reshape(1, -1))

            Ay = abs(self.A).multiply(y)
            sAy = slack.multiply(Ay)
            W = sAy.transpose() @ A_bool

        else:
            raise ValueError("graph unknown")

        W = sparse.csr_matrix(W)

        return W

    def get_best_candidate(self, action_set: np.array) -> np.array:
        """
        Get best action in action set in terms of influence
        taking into account previous branching descisions
        in parents nodes.
        """
        branched_vars = []
        current_node = self.node
        W_n = sparse.lil_matrix(self.W)
        while current_node.getDepth() > 0:
            parent_branching_var = current_node.getParentBranchings()[0]
            branched_vars += [self.vars_dict[f"{parent_branching_var[0]}"]]
            current_node = current_node.getParent()
        if len(branched_vars) > 0:
            W_n[branched_vars, :] = 0
        idx_actions = np.intersect1d(action_set, self.idx_integer, assume_unique=True)
        variables_influence = np.array(W_n[idx_actions, :].sum(axis=1)).reshape(-1)
        if True:
            variables_influence *= self.c_.reshape(-1)[idx_actions]
        idx_best_candidate = np.argmax(variables_influence)
        best_candidate = action_set[idx_best_candidate].item()
        del W_n
        return best_candidate

    def get_influence_graph(self):
        self.A, self.b, self.c = self.std_model()
        self.W = self.get_W()
        self.branching_restart = False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clusters", default=1, type=int)
    parser.add_argument("--graph", default="base", type=str)
    parser.add_argument("--ibra", default=1, type=int)
    parser.add_argument("--verbose", default=0, type=int)
    parser.add_argument("--time", default=300, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--no_clusters", default=1, type=int)
    model = scip.Model()
    args = parser.parse_args()

    model.readProblem("../../datasets/vary_obj/series_2/obj_s2_i01.mps.gz")
    model.setRealParam("limits/time", args.time)

    if args.ibra == 1:
        branchrule = SparsInfluenceBranching(
            args.n_clusters,
            args.graph,
            args.seed,
            verbose=args.verbose,
            no_clusters=args.no_clusters,
        )
        model.includeBranchrule(
            branchrule=branchrule,
            name="Influence Branching",  # name of the branching rule
            desc="",  # description of the branching rule
            priority=10001,  # priority: set to this to make it default
            maxdepth=args.n_clusters - 1,  # max depth
            maxbounddist=1,
        )
    # model.hideOutput()
    model.optimize()
    if args.ibra == 1:
        print("Nombre de restarts : ", branchrule.n_run)
        print("Bilan branchements : ", branchrule.summary)
        print("Nombre d'appels : ", branchrule.n_call)
