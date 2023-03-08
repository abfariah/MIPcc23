import sys
import pyscipopt as scip
import numpy as np
from typing import Tuple

sys.path.append("../")


class InfluenceBranching(scip.Branchrule):
    """
    Original implementation of influence branching rule
    using SCIP solver and pySCIPopt python interface
    """

    def __init__(self, graph: str, max_depth: int) -> None:
        super(scip.Branchrule).__init__()
        self.max_depth = max_depth
        self.W = None
        self.graph = graph
        self.branching_restart = True
        self.depth = 0
        self.n_call = 0
        self.n_run = 0
        self.obj_1 = False
        self.sparse = False
        self.normalize = True
        self.never_normalize = False  # To be adjusted
        self.symmetrize = False  # To be adjusted

    def get_vars_dict(self):
        """
        Builds dictionnary associating id to every
        pySCIPopt variable object
        """
        vars_dict = {}
        for idx, var in enumerate(self.variables):
            vars_dict[f"{var}"] = idx
        return vars_dict

    def get_true_A(self) -> np.array:
        """
        Partially vectorized version
        Retrieves matrix A
        """
        A_lines = []
        self.vars_dict = self.get_vars_dict()
        for cons in self.linear_constraints:
            A_line = np.zeros(self.n)
            constraint_dict = self.model.getValsLinear(cons)
            keys = np.array(list(constraint_dict.keys()))
            values = np.array(list(constraint_dict.values()))
            idx_var = [self.vars_dict[f"{key}"] for key in keys]
            A_line[idx_var] = values
            A_lines += [A_line]
        if len(A_lines) > 0:
            A_true = np.vstack(A_lines)
        else:
            A_true = np.empty((0, self.n))
        return A_true

    def get_matrices(self) -> Tuple[np.array, np.array]:
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
        self.test_series(binary_condition)

        ## A, b, c
        c = np.array([var.getObj() for var in self.variables])
        A = self.get_true_A()
        return A, c

    def get_b(self) -> np.array:
        """
        Compute the vector b used for normalization from linear constraints
        """
        rhs = np.array(
            [[self.model.getRhs(cons) for cons in self.linear_constraints]]
        ).reshape(-1, 1)
        lhs = np.array(
            [[self.model.getLhs(cons) for cons in self.linear_constraints]]
        ).reshape(-1, 1)

        if len(rhs) > 0:
            rhs_idx = np.where(rhs < 1e10)
            lhs_idx = np.where(lhs > -1e10)
            b = np.zeros_like(lhs)
            b[lhs_idx] -= lhs[lhs_idx]
            b[rhs_idx] += rhs[rhs_idx]
        else:
            b = np.empty((0, 1))
            lhs_idx = np.empty((0, 1))
            rhs_idx = np.empty((0, 1))

        return b, lhs_idx, rhs_idx

    def std_matrices(
        self,
        A: np.array,
        c: np.array,
    ) -> Tuple[np.array, np.array, np.array]:
        """
        Standardize the problem data.
        """
        # compute b
        b, lhs_idx, rhs_idx = self.get_b()
        # objective
        denom = np.std(c)
        if denom > 1e-5:
            c /= denom

        # constraint matrix
        ix_0, ix_1 = np.where(b == 0)[0], np.where(b != 0)[0]
        A[ix_1, :] = A[ix_1, :] / b[ix_1]
        denom = np.std(A[ix_0, :], axis=1)
        denom[denom <= 1e-5] = 1.0
        A[ix_0, :] = A[ix_0, :] / denom.reshape(-1, 1)

        # constraint rhs
        b[lhs_idx] = -1.0
        b[rhs_idx] = 1.0
        b[ix_0] = 0

        return A, b, c

    def test_normalize(self) -> bool:
        """
        Skip normalization if there is no linear constraints
        """
        if self.never_normalize:
            self.normalize = False
        elif self.m_linear == 0:
            self.normalize = False
        else:
            self.normalize = True
        return self.normalize

    def std_model(self) -> Tuple[np.array, np.array, np.array]:
        """
        Standardize pb and return the standardized data
        """
        A, c = self.get_matrices()
        if self.test_normalize():
            A, b, c = self.std_matrices(A, c)
            return A, b, c
        else:
            return A, None, c

    def symmetrizeGraph(self, W: np.array):
        return (W + W.T) * 0.5

    def get_primal_dual_values(self):
        x_primal = np.array([var.getLPSol() for var in self.variables])
        y_dual = np.array(
            [self.model.getDualSolVal(cons) for cons in self.linear_constraints]
        )
        return x_primal, y_dual

    def get_W(self) -> np.array:
        A_bool = (self.A != 0) * 1.0
        if self.graph == "count":
            W = np.matmul(A_bool.T, A_bool) * 1.0

        elif self.graph == "base":  ## named binary_graph in Marc's manuscript
            W = np.dot(A_bool.T, A_bool) * 1.0
            W[W != 0] = 1

        elif self.graph == "countdual":
            x_primal, y_dual = self.get_primal_dual_values()
            y = (np.abs(y_dual) > 1e-3) * 1.0
            A_bool = A_bool * y.reshape(-1, 1)
            W = np.dot(A_bool.T, A_bool) * 1.0

        elif self.graph == "bindual":
            x_primal, y_dual = self.get_primal_dual_values()
            y = (np.abs(y_dual) > 1e-3) * 1.0
            A_bool = A_bool * y.reshape(-1, 1)
            W = np.dot(A_bool.T, A_bool) * 1.0
            W[W != 0] = 1

        elif self.graph == "dual":
            x_primal, y_dual = self.get_primal_dual_values()
            y = np.sqrt(np.abs(y_dual))
            A_bool = A_bool * y.reshape(-1, 1)
            W = np.dot(A_bool.T, A_bool) * 1.0

        elif self.graph in ["auxiliary", "adversarial"]:
            x_primal, y_dual = self.get_primal_dual_values()
            lb = np.array([var.getLbLocal() for var in self.variables])
            ub = np.array([var.getUbLocal() for var in self.variables])
            bound_idx = [i for i in range(self.n) if i not in self.idx_integer]
            lb[bound_idx] = x_primal[bound_idx]
            ub[bound_idx] = x_primal[bound_idx]

            y = np.abs(y_dual)
            ix = np.where(ub == lb)[0]
            denom = ub - lb
            denom[ix] = 1
            slack = 0.25 + (x_primal - lb) * (ub - x_primal) / denom
            slack[ix] = 0.25

            Ay = y.reshape(-1, 1) * np.abs(self.A)
            sAy = slack.reshape(1, -1) * Ay
            W = np.dot(sAy.T, A_bool)

            if self.graph == "adversarial":
                A_ = self.A.copy() * 1
                A_[A_ == 0] = 1
                y = (y != 0) * 1

                invA = 1.0 / np.abs(A_)
                Ay = np.reshape(y, (-1, 1)) * np.abs(self.A)
                sAy = np.reshape(slack, (1, -1)) * Ay
                W = np.dot(sAy.T, invA)

        else:
            raise ValueError("graph unknown")

        is_nul = W == 0
        W[is_nul] = 1e-3
        W -= np.diag(np.diagonal(W))
        if self.symmetrize:
            W = self.symmetrizeGraph(W)

        return W

    def test_series(self, binary_variables):
        if len(self.variables) == sum(binary_variables):
            self.obj_1 = True

    def get_best_candidate(self, action_set: np.array) -> np.array:
        """
        Get best action in action set in terms of influence
        taking into account previous branching descisions
        in parents nodes.
        """
        branched_vars = []
        current_node = self.node
        W_n = self.W.copy()
        while current_node.getDepth() > 0:
            parent_branching_var = current_node.getParentBranchings()[0]
            branched_vars += [self.vars_dict[f"{parent_branching_var[0]}"]]
            current_node = current_node.getParent()
        if len(branched_vars) > 0:
            W_n[branched_vars, :] = 0
        idx_actions = np.intersect1d(action_set, self.idx_integer, assume_unique=True)
        variables_influence = np.sum(W_n[idx_actions, :], axis=1)
        if self.obj_1:
            variables_influence *= 1 + np.abs(self.c[idx_actions])
        else:
            variables_influence *= np.sqrt(1 + self.c[idx_actions])
        idx_best_candidate = action_set[np.argmax(variables_influence)]
        del W_n
        return idx_best_candidate

    def get_influence_graph(self):
        self.A, self.b, self.c = self.std_model()
        self.W = self.get_W()
        self.branching_restart = False

    def branchexeclp(self, allowaddcons):  ### MAIN
        """
        Influence Branching rule
        """
        self.node = self.model.getCurrentNode()
        self.depth = self.node.getDepth()
        if self.depth == 0:
            self.n_run += 1
            self.branching_restart = True
        assert self.depth <= self.max_depth
        if self.branching_restart:
            self.get_influence_graph()

        candidate_vars, *_ = self.model.getLPBranchCands()
        action_set = np.array([self.vars_dict[f"{var}"] for var in candidate_vars])
        idx_branch_var = self.get_best_candidate(action_set)
        branch_var = self.variables[idx_branch_var]

        self.model.branchVar(branch_var)
        self.n_call += 1
        result = scip.SCIP_RESULT.BRANCHED
        return {"result": result}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--max_depth", default=1, type=int)
    parser.add_argument("--graph", default="base", type=str)
    parser.add_argument("--ibra", default=1, type=int)
    parser.add_argument("--verbose", default=0, type=int)
    parser.add_argument("--time", default=300, type=int)
    parser.add_argument("--seed", default=0, type=int)
    model = scip.Model()
    args = parser.parse_args()

    model.readProblem("../datasets/vary_obj/series_2/obj_s2_i04.mps.gz")
    model.setRealParam("limits/time", args.time)

    if args.ibra == 1:
        branchrule = InfluenceBranching(
            args.graph,
            args.max_depth,
            args.seed,
            verbose=args.verbose,
        )
        model.includeBranchrule(
            branchrule=branchrule,
            name="Influence Branching",  # name of the branching rule
            desc="",  # description of the branching rule
            priority=10001,  # priority: set to this to make it default
            maxdepth=args.max_depth - 1,  # max depth
            maxbounddist=1,
        )
    # model.hideOutput()
    model.optimize()
    if args.ibra == 1:
        print("Nombre de runs : ", branchrule.n_run)
        print("Nombre d'appels : ", branchrule.n_call)
