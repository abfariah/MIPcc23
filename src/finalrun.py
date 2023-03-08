import os
import sys
from BandMIP import MIPBandit, ThompsonSamplingNorm

ACTIONS = [("count", 1), ("count", 5), ("countdual", 2), ("base", 3), ("dual", 3)]
MU_0, SIGMA = 0.9, 0.2
N_INSTANCES = 50

meta_file = sys.argv[-1]
if not os.path.isfile(meta_file):
    print("Usage: python run.py /path/to/meta/file")
    exit()

with open(meta_file) as file:
    all_lines = [line.rstrip() for line in file]

if not os.path.isfile(meta_file):
    print("Meta file is invalid")
    print(meta_file)
    exit()

# read instances from meta file
instances = all_lines[7 : len(all_lines)]
print("number of instances:", len(instances))

# read time limit from meta file
time_limit = int(all_lines[0].split(" ")[-1])

solution_folder = os.path.join(
    "solutions/", os.path.basename(os.path.splitext(meta_file)[0])
)
if not os.path.isdir(solution_folder):
    os.mkdir(solution_folder)

base_folder = os.path.dirname("../")

series_params = {
    "solution folder": solution_folder,
    "base folder": base_folder,
    "time limit": time_limit,
    "instances": instances,
    "actions": ACTIONS,
}

bandit = MIPBandit(series_params)
online_solver = ThompsonSamplingNorm(bandit, MU_0, SIGMA)
online_solver.run(N_INSTANCES)
