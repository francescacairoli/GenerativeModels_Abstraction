from MySSA_torch import *
import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


n_species = 3
n_reactions = 2
N = 200
rates = torch.tensor([3,1]).to(device)
updates = torch.tensor([[-1,1,0],[0,-1,1]]).to(device)
prop1 = lambda s: rates[0]*(s[0]*s[1])/N
prop2 = lambda s: rates[1]*s[1]
propensities = [prop1,prop2]

input_dict = {"n_species": n_species, "n_reactions": n_reactions, "rates": rates, 
				"propensities": propensities, "updates": updates}

nb_init_config = 10
n_trajs_per_config = 500
time_horizon = 8
nb_temp_points = 17

ssa = MySSA(input_dict, True, "SIR")

lbs = [0,0,0]
ubs = [200,200,200]

print("Generating initial configurations...")
init_states = ssa.generate_rnd_init_state(n_configs=nb_init_config, LB=lbs, UB=ubs)

st = time.time()
print("Generating SSA trajectories...")
SSA_trajs = ssa.SSA_simulation(n_trajs_per_config, time_horizon, nb_temp_points)
end = time.time()
print("total time = ", end)