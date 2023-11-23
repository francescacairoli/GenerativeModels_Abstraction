import numpy as np
import torch
import time

#plt.rcParams.update({'font.size': 22})
import copy
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        X0 = torch.zeros((n_configs,self.n_species)).to(device)
        
        for i in range(self.n_species):
            X0[:,i] = torch.randint(LB[i], UB[i], size =(n_configs,))
                            
        self.init_configs = X0



    def evaluate_rates(self, state):

        rates = torch.empty(self.n_reactions).to(device)
        for i in range(self.n_reactions):
            rates[i]=self.propensity_functions[i](state)
        return rates


        


    def SSA_simulation(self, n_trajs_per_config, final_time, n_time_points):

        self.ssa_final_time = final_time
        self.ssa_n_time_points = n_time_points
        
        self.n_trajs_per_config = n_trajs_per_config

        self.ssa_timestamp = np.linspace(0,final_time, n_time_points)
        SSA_trajs = torch.empty((self.n_configs,n_trajs_per_config, self.n_species, n_time_points)).to(device)

        for j in range(self.n_configs):
            print('point = {}/{}'.format(j+1, self.n_configs))

            st = time.time()
            for z in range(n_trajs_per_config):

                timer = 0
                print_index = 1
                state = copy.deepcopy(self.init_configs[j])

                traj = torch.zeros((n_time_points, self.n_species)).to(device)
                traj[0, :] = state

                # main SSA loop
                
                while timer < final_time:
                    # compute rates and total rate
                    rates = self.evaluate_rates(state)
                    # sanity check, to avoid negative numbers close to zero
                    rates[rates < 1e-14] = 0.0
                    total_rate = torch.sum(rates)
                    # check if total rate is non zero.
                    if total_rate > 1e-14:
                        # if so, sample next time and next state and update state and time
                        trans_index = torch.multinomial(rates / total_rate,1)[0]
                        
                        delta_time = torch.distributions.Exponential(total_rate).sample()
                    
                        timer += delta_time
                        update_vector = self.update_vectors[trans_index]
                        state += update_vector
                    else:
                        # If not, stop simulation by skipping to final time
                        timer = final_time
                    # store values in the output array
                    while print_index < n_time_points and self.ssa_timestamp[print_index] <= timer:
                        traj[print_index, :] = state
                        print_index += 1          
                SSA_trajs[j,z] = traj.T
            en = time.time()-st
            print("time per config = ", en)
            print("time per traj = ", en/n_trajs_per_config)
            if self.plots_flag:
                fig = plt.figure()
                for zz in range(n_trajs_per_config):
                    c = 0
                    for k in range(self.n_species):
                        if zz == 0:
                            plt.plot(self.ssa_timestamp, SSA_trajs[j,zz,k].cpu().detach().numpy(), self.colors[c%len(self.colors)], label="S{}".format(c))
                        else:
                            plt.plot(self.ssa_timestamp, SSA_trajs[j,zz,k].cpu().detach().numpy(), self.colors[c%len(self.colors)])
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