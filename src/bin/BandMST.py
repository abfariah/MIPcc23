import pyscipopt as scip
import os
from BandMIP import NormalBandit
from utils import compute_dual_gap
from InfluenceBranching import InfluenceBranching


class MIPBandit(NormalBandit):
    """Train bandit"""

    def __init__(self, params: dict):
        self.solution_folder = params["solution folder"]
        self.base_folder = params["base folder"]
        self.instances = iter(params["instances"])
        self.actions = params["actions"]
        self.time_limit = params["time limit"]
        n = self.actions.__len__()
        mus = [1.0] * n
        sigma = 0.3
        super(MIPBandit, self).__init__(n, mus, sigma)

    def read_instance(self):
        """Read the next instance"""
        self.MIPinstance = next(self.instances)
        self.instance_base = os.path.basename(self.MIPinstance)
        print("[INSTANCE]", self.instance_base)
        instance_path = os.path.join(self.base_folder, self.MIPinstance)
        self.model = scip.Model()
        self.model.readProblem(instance_path)
        self.model.setRealParam("limits/time", self.time_limit)

    def generate_reward(self, i: int):
        """
        Solve current instance for action i and return reward
        """
        ## Set parameters
        action = self.actions[i]
        graph, max_depth = action

        branchrule = InfluenceBranching(
            graph,
            max_depth,
        )

        self.model.includeBranchrule(
            branchrule=branchrule,
            name="Influence Branching",  # name of the branching rule
            desc="",  # description of the branching rule
            priority=10001,  # priority: set to this to make it default
            maxdepth=max_depth - 1,  # max depth
            maxbounddist=1,
        )

        # optimize
        self.model.optimize()

        dual_gap = compute_dual_gap(self.model)
        solving_time = self.model.getSolvingTime()
        nofeas = (self.model.getStatus() == "infeasible") * 1.0
        r = solving_time + dual_gap + nofeas

        return r

    def write_solutions(self):
        """
        Write solution to instance_name.sol in 'solutions' directory
        """
        print("[DUALBOUND]", self.model.getDualbound())
        if self.model.getNSols() > 0:
            sol = self.model.getBestSol()
            with open(
                os.path.join(self.solution_folder, f"{self.instance_base}.sol"), "w"
            ) as f:
                for j in range(self.model.getNVars()):
                    v = self.model.getVars()[j]
                    name = v.name
                    val = sol[v]
                    f.write(name)
                    f.write("    ")
                    f.write(str(val))
                    f.write("\n")
        else:
            print("No solution found")
