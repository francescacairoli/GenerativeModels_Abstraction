import numpy as np
#import torch
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
import copy
import time as timer

class MySSA():

    def __init__(self, params, plots_flag = False, idx=""):
        self.n_species = params["n_species"]
        self.n_reactions = params["n_reactions"]
        self.rates = params["rates"]
        self.update_vectors = params["updates"]
        self.propensity_functions=params["propensities"]
        self.plots_flag = plots_flag
        self.colors = ['r', 'g', 'b', 'c', 'm', 'y']
        self.idx = idx


    def generate_rnd_init_state(self, LB, UB, n_configs=1):

        self.n_configs = n_configs
        X0 = np.zeros((self.n_species,n_configs))
        
        for i in range(self.n_species):
            X0[i] = np.random.randint(LB[i], UB[i])
                            
        self.init_configs = X0.T



    def evaluate_rates(self, state):

        rates = np.empty(self.n_reactions)
        for i in range(self.n_reactions):
            rates[i]=self.propensity_functions[i](state)
        return rates


        


    def SSA_simulation(self, n_trajs_per_config, final_time, n_time_points):

        self.ssa_final_time = final_time
        self.ssa_n_time_points = n_time_points
        
        self.n_trajs_per_config = n_trajs_per_config

        self.ssa_timestamp = np.linspace(0,final_time, n_time_points)
        SSA_trajs = np.empty((self.n_configs,n_trajs_per_config, self.n_species, n_time_points))

        for j in range(self.n_configs):
            print('point = {}/{}'.format(j+1, self.n_configs))

            st = timer.time()
            
            for z in range(n_trajs_per_config):

                time = 0
                print_index = 1
                state = copy.deepcopy(self.init_configs[j])

                traj = np.zeros((n_time_points, self.n_species))
                traj[0, :] = state

                # main SSA loop
                
                while time < final_time:
                    # compute rates and total rate
                    rates = self.evaluate_rates(state)
                    # sanity check, to avoid negative numbers close to zero
                    rates[rates < 1e-14] = 0.0
                    total_rate = np.sum(rates)
                    # check if total rate is non zero.
                    if total_rate > 1e-14:
                        # if so, sample next time and next state and update state and time
                        trans_index = np.random.choice(self.n_reactions, p = rates / total_rate)
                        delta_time = np.random.exponential(1 / total_rate)
                        time += delta_time
                        update_vector = self.update_vectors[trans_index]
                        state += update_vector
                    else:
                        # If not, stop simulation by skipping to final time
                        time = final_time
                    # store values in the output array
                    while print_index < n_time_points and self.ssa_timestamp[print_index] <= time:
                        traj[print_index, :] = state
                        print_index += 1          
                SSA_trajs[j,z] = traj.T

            en = timer.time()-st
            print("time per config = ", en)
            print("time per traj = ", en/n_trajs_per_config)
            lab =["S","I","R"]
            if self.plots_flag:
                fig = plt.figure()
                for zz in range(n_trajs_per_config):
                    c = 0
                    for k in range(self.n_species):
                        if zz == 0:
                            plt.plot(self.ssa_timestamp, SSA_trajs[j,zz,k], self.colors[c%len(self.colors)], label=lab[k])
                        else:
                            plt.plot(self.ssa_timestamp, SSA_trajs[j,zz,k], self.colors[c%len(self.colors)])
                        c += 1
                    #else: # plot transitioning bikes
                            #plt.plot(self.ssa_timestamp, SSA_trajs[j,zz,k],'--', self.colors[k%len(self.colors)])
                plt.title("stochastic")
                plt.legend(fontsize=14)
                plt.xlabel("time")
                plt.ylabel("count")
                plt.tight_layout()
                plt.savefig("myplots/{}_SSA_trajs_H={}_{}.png".format(self.idx,final_time,j))
                plt.close()            

        self.SSA_trajs = SSA_trajs