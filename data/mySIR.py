from MySSA import *

n_species = 3
n_reactions = 2
N = 200
rates = [5,1]
updates = [[-1,1,0],[0,-1,1]]
prop1 = lambda s: rates[0]*(s[0]*s[1])/N
prop2 = lambda s: rates[1]*s[1]
propensities = [prop1,prop2]

input_dict = {"n_species": n_species, "n_reactions": n_reactions, "rates": rates, 
				"propensities": propensities, "updates": updates}

nb_init_config = 1
n_trajs_per_config = 10
time_horizon = 8
nb_temp_points = 100

ssa = MySSA(input_dict, True, "SIR")

ssa.n_configs = nb_init_config
lbs = np.zeros(n_species)
ubs = 200*np.ones(n_species)

print("Generating initial configurations...")
#init_states = ssa.generate_rnd_init_state(n_configs=nb_init_config, LB=lbs, UB=ubs)
ssa.init_configs = np.array([[150,50,20]])

print("Generating SSA trajectories...")
SSA_trajs = ssa.SSA_simulation(n_trajs_per_config, time_horizon, nb_temp_points)