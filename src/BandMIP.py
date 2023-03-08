from __future__ import division
import numpy as np
import time
import math
import pyscipopt as scip
import os
from datetime import datetime

from utils import compute_dual_gap
from InfluenceBranching import InfluenceBranching
from SparsInfluenceBranching import SparsInfluenceBranching


def ind_min(x):
    m = min(x)
    return x.index(m)


class Bandit(object):
    def generate_reward(self, i):
        raise NotImplementedError


class NormalBandit(Bandit):
    def __init__(self, n, mus, sigma):
        assert mus is None or len(mus) == n
        self.n = n
        if mus is None:
            np.random.seed(int(time.time()))
            self._mus = [np.random.uniform(-10, 10) for _ in range(self.n)]
            self._sigma = np.random.uniform(-2, 2)
        else:
            self._mus = mus
            self._sigma = sigma

        self.best_mu = min(self._mus)
        self.best_machine = self._mus.index(self.best_mu)

    def generate_reward(self, i):
        # The player selected the i-th machine.
        r = np.random.normal(self._mus[i], self._sigma)
        return r


class MIPBandit(NormalBandit):
    """
    MIPcc23 Bandit
    """

    def __init__(self, params: dict):
        self.solution_folder = params["solution folder"]
        self.base_folder = params["base folder"]
        self.instances = iter(params["instances"])
        self.actions = params["actions"]
        self.time_limit = params["time limit"]
        n = self.actions.__len__()
        mus = [0.9] * n
        sigma = 0.2
        super(MIPBandit, self).__init__(n, mus, sigma)

    def read_instance(self):
        """
        Read the next instance
        """
        self.MIPinstance = next(self.instances)
        self.instance_base = os.path.basename(self.MIPinstance)
        print("[INSTANCE]", self.instance_base)
        instance_path = os.path.join(self.base_folder, self.MIPinstance)
        self.model = scip.Model()
        self.model.readProblem(instance_path)
        self.model.setRealParam("limits/time", self.time_limit - 1)

    def generate_reward(self, i: int):

        ## Set parameters (g, k) for influence branching
        action = self.actions[i]
        graph, max_depth = action
        print("[ACTION] : ", action)

        if self.model.getNVars() < 25000:
            branchrule = InfluenceBranching(graph, max_depth)
        else:
            if max_depth == 5:
                max_depth += 3
            branchrule = SparsInfluenceBranching(graph, max_depth)

        self.model.includeBranchrule(
            branchrule=branchrule,
            name="Influence Branching",  # name of the branching rule
            desc="",  # description of the branching rule
            priority=10001,  # priority: set to this to make it default
            maxdepth=max_depth - 1,  # max depth
            maxbounddist=1,
        )

        # optimize
        self.model.hideOutput()
        self.model.optimize()

        # compute reward
        dual_gap = compute_dual_gap(self.model)
        solving_time = self.model.getSolvingTime()
        nofeas = (self.model.getStatus() == "infeasible") * 1.0
        r = solving_time + dual_gap + nofeas

        return r

    def write_solutions(self):
        """
        write solution to instance_name.sol in 'solutions' directory
        """
        print("[DUALBOUND]", self.model.getDualbound())
        if self.model.getNSols() > 0:
            sol = self.model.getBestSol()
            with open(
                os.path.join(self.solution_folder, f"{self.instance_base[:-7]}.sol"),
                "w",
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


class Solver(object):
    def __init__(self, bandit):
        """
        bandit (Bandit): the target bandit to solve.
        """
        assert isinstance(bandit, NormalBandit)
        np.random.seed(int(time.time()))

        self.bandit = bandit
        self.counts = [0] * self.bandit.n
        self.actions = []  # A list of machine ids, 0 to bandit.n-1.
        self.rewards = []  # A list of rewards.
        self.regret = 0.0  # Cumulative regret.
        self.regrets = [0.0]  # History of cumulative regret.

    def update_regret(self, i):
        # i (int): index of the selected machine.
        self.regret += self.bandit.best_mu - self.bandit._mus[i]
        # self.regret += (1+0.1*self.bandit.t) \
        #              * (self.bandit.best_mu - self.bandit._mus[i])
        self.regrets.append(self.regret)

    @property
    def estimated_mus(self):
        raise NotImplementedError

    def run_one_step(self):
        """Return the machine index to take action on."""
        raise NotImplementedError

    def run(self, num_steps):
        assert self.bandit is not None
        for _ in range(num_steps):
            i = self.run_one_step()
            self.counts[i] += 1
            self.actions.append(i)
            # self.update_regret(i)


class UCB1(Solver):
    def __init__(self, bandit, mu_0=1.0):
        super(UCB1, self).__init__(bandit)
        self.t = 1
        self._mus = [mu_0] * self.bandit.n

    @property
    def estimated_mus(self):
        return self._mus

    def run_one_step(self):
        # Pick the best one with consideration of upper confidence bounds.
        i = min(
            range(self.bandit.n),
            key=lambda x: self._mus[x]
            - np.sqrt(2 * np.log(self.t) / (1 + self.counts[x])),
        )
        r = self.bandit.generate_reward(i)
        self.rewards.append(r)
        self.t += 1
        self._mus[i] += 1.0 / (self.counts[i] + 1) * (r - self._mus[i])

        return i


class ThompsonSamplingNorm(Solver):
    def __init__(self, bandit: MIPBandit, mu_0: float, sigma_0: float):
        """
        Thompson sampling for normal bandits
        """
        super(ThompsonSamplingNorm, self).__init__(bandit)
        self._mu0, self._sigma0 = mu_0, sigma_0
        self._mus = [mu_0] * self.bandit.n
        self._sigmas = [sigma_0] * self.bandit.n
        self.cumulated_rewards = [0] * self.bandit.n
        self.n_call = [0] * self.bandit.n

    @property
    def estimated_mus(self) -> list[float]:
        return [self._mus[i] for i in range(self.bandit.n)]

    def get_n(self) -> int:
        return sum(self.n_call)

    def run_one_step(self) -> int:
        """
        Solve current instance
        """
        self.bandit.read_instance()
        print("[START] : ", datetime.now().isoformat())

        # Sample from the prior distribution
        samples = [
            np.random.normal(self._mus[x], self._sigmas[x])
            for x in range(self.bandit.n)
        ]
        # Get estimated best action
        i = min(range(self.bandit.n), key=lambda x: samples[x])
        self.n_call[i] += 1

        # Solve instance and observe reward
        r = self.bandit.generate_reward(i)
        self.rewards.append(r)
        self.cumulated_rewards[i] += r

        # Compute posterior parameters
        inv_sigma = 1 / self._sigma0 + self.n_call[i] / self.bandit._sigma
        self._sigmas[i] = 1 / inv_sigma
        self._mus[i] = self._sigmas[i] * (
            self._mu0 / self._sigma0 + self.cumulated_rewards[i] / self.bandit._sigma
        )

        print("[END] : ", datetime.now().isoformat())

        self.bandit.write_solutions()

        return i


class ThompsonSamplingNormGamma(Solver):
    def __init__(
        self, bandit: NormalBandit, mu_0: float, k_0: float, a_0: float, b_0: float
    ):
        """
        Thompson sampling for normal bandits
        """
        super(ThompsonSamplingNormGamma, self).__init__(bandit)
        self._mu0, self._k0 = mu_0, k_0
        self._a0, self._b0 = a_0, b_0
        self._mus = [mu_0] * self.bandit.n
        self._ks = [k_0] * self.bandit.n
        self._as = [a_0] * self.bandit.n
        self._bs = [b_0] * self.bandit.n
        self.cumulated_rewards = [0] * self.bandit.n
        self.historic = [[] for _ in range(self.bandit.n)]
        self.n_call = [0] * self.bandit.n

    @property
    def estimated_mus(self) -> list[float]:
        return [self._mus[i] for i in range(self.bandit.n)]

    ## Attention au sigma qui varie !
    ## peut être changer de critère

    def get_t(self) -> int:
        return sum(self.n_call)

    def get_sigmas(self) -> list[float]:
        """
        Our hypothesis is that the inverse of
        the variance is proportionnal to a Gamma law
        """
        sigmas = [
            1
            / (self._ks[x] * np.random.gamma(shape=self._as[x], scale=1 / self._bs[x]))
            for x in range(self.bandit.n)
        ]
        return sigmas

    def run_one_step(self) -> int:
        # Sample from the prior distribution
        samples = [
            np.random.normal(self._mus[x], self.get_sigmas()[x])
            for x in range(self.bandit.n)
        ]

        # Get best action and observe reward
        i = min(range(self.bandit.n), key=lambda x: samples[x])
        historic = self.historic[i]
        self.n_call[i] += 1
        r = self.bandit.generate_reward(i)
        self.cumulated_rewards[i] += r
        historic += [r]

        # Compute posterior parameters
        self._mus[i] = (self._k0 * self._mu0 + self.cumulated_rewards[i]) / (
            self._k0 + self.n_call[i]
        )
        self._ks[i] = self._k0 + self.n_call[i]
        self._as[i] = self._a0 + self.n_call[i] / 2
        second_term = (
            self._k0
            * (self.n_call[i] * self._mu0 - self.cumulated_rewards[i]) ** 2
            / (2 * (self._k0 + self.n_call[i]))
        )
        third_term = 0.5 * sum(
            [
                (self.n_call[i] * rew - self.cumulated_rewards[i]) ** 2
                for rew in historic
            ]
        )
        self._bs[i] = self._b0 + second_term + third_term

        return i


class UCB2(UCB1):
    def __init__(self, bandit, alpha, mu_0=1):
        """
        UCB 2 implementation
        """
        super(UCB2, self).__init__(bandit, mu_0=1)
        self.alpha = alpha
        self.rho = [0] * self.bandit.n
        self.__current_arm = 0
        self.__next_update = self.t
        return

    def __bonus(self, rho):
        tau = self.__tau(rho)
        bonus = math.sqrt(
            math.log(math.e * self.t / tau) * (1.0 + self.alpha) / (2 * tau)
        )
        return bonus

    def __tau(self, rho):
        return int(math.ceil((1 + self.alpha) ** rho))

    def __set_arm(self, i):
        """
        When choosing a new arm, make sure we play that arm for
        tau(r+1) - tau(r) episodes.
        """
        self.__current_arm = i
        self.__next_update += max(
            1, self.__tau(self.rho[i] + 1) - self.__tau(self.rho[i])
        )
        self.rho[i] += 1

    def select_arm(self):

        # play each arm once
        for x in range(self.bandit.n):
            if self.counts[x] == 0:
                i = x
                self.__set_arm(i)
                return i

        # make sure we aren't still playing the previous arm.
        if self.__next_update > self.t:
            return self.__current_arm

        # compute ucb and get best arm
        ucb_mus = [
            self._mus[x] - self.__bonus(self.rho[x]) for x in range(self.bandit.n)
        ]
        i = ind_min(ucb_mus)
        self.__set_arm(i)

        return i

    def run_one_step(self):
        # Pick the best one with consideration of upper confidence bounds.
        i = self.select_arm()
        # Generate reward
        r = self.bandit.generate_reward(i)
        self.rewards.append(r)
        # Update mu
        self._mus[i] += 1.0 / (self.counts[i] + 1) * (r - self._mus[i])
        self.t += 1
        return i
