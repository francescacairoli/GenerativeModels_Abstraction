import numpy as np
import gillespy2
import matplotlib.pyplot as plt
import pickle
import time

class EColi(gillespy2.Model):
    def __init__(self, tend, nsteps):
        # First call the gillespy2.Model initializer.
        super().__init__(self)

        # Define parameters for the rates of creation and dissociation.
        kplus = gillespy2.Parameter(name='kplus', expression=0.1)
        kminus = gillespy2.Parameter(name='kminus', expression=0.1)
        mu = gillespy2.Parameter(name='mu', expression=5)

        self.parameters_list = [kplus,kminus,mu]
        self.add_parameter(self.parameters_list)

        # Define variables for the molecular species representing M & D.
        N = gillespy2.Species(name='N', initial_value=0)
        C = gillespy2.Species(name='C', initial_value=0)
        S = gillespy2.Species(name='S', initial_value=0)
        
        self.species_list = [N,C,S]
        self.add_species(self.species_list)

        # The list of reactants and products for a Reaction object are
        # each a Python dictionary in which the dictionary keys are
        # Species objects and the values are stoichiometries of the
        # species in the reaction.
        r1 = gillespy2.Reaction(name="r1", rate=kplus,
                                 reactants={C:1}, products={N:1})

        r2 = gillespy2.Reaction(name="r2", rate=kplus,
                                 reactants={S:1}, products={N:1})
        
        r3 = gillespy2.Reaction(name="r3", rate=kminus,
                         reactants={N:1}, products={S:1})

        r4 = gillespy2.Reaction(name="r4", rate=mu,
                                 reactants={S:1}, products={C:1})
        
        self.add_reaction([r1, r2, r3, r4])

        # Set the timespan for the simulation.
        self.timespan(np.linspace(0, tend, nsteps))



    def sample_state(self):
        
        n = np.random.randint(0,6)
        c = np.random.randint(0,int(6-n))
        s = 6-n-c
        #s = np.random.randint(0,4)
        
        self.species_list[0].initial_value = n
        self.species_list[1].initial_value = c
        self.species_list[2].initial_value = s

    def set_state(self, s):
        # s = [n,c,s]
        self.species_list[0].initial_value = s[0]
        self.species_list[1].initial_value = s[1]
        self.species_list[2].initial_value = s[2]

    def compute_rates(self, m,l):
        w = 1.3
        g0 = 40
        g1 = 40
        kd = 3.06*10**6 # 3.06 microM (10-6)
        alp = 6*10**6 #    6 microM
        eps0 = 1.
        eps1 = -0.45
        ntar = 6
        ntsr = 13
        ktar_off = 0.02*10**(3)     #0.02mM
        ktar_on = 0.4*10**(3)       #0.4mM
        ktsr_off = 100.*10**(3)  #100mM
        ktsr_on = 10**9 #   10**6mM

        yp = alp/(1+np.exp(eps0+eps1*m)*((1+l/ktar_off)/(1+l/ktar_on))**ntar*((1+l/ktsr_off)/(1+l/ktsr_on))**ntsr)

        xx = g0/4-g1/2*(yp/(yp+kd))

        kp = w*np.exp(xx)
        km = w*np.exp(-xx)

        return np.array([kp, km])

    def sample_param(self):
        
        m = np.random.uniform(0,35)
        l = np.random.uniform(0,0.1)
        par = self.compute_rates(m,l)

        self.parameters_list[0].expression = par[0]
        self.parameters_list[1].expression = par[1]
        
        return par

    def compute_and_set_param(self, m, l):
        p = self.compute_rates(m,l)
        self.parameters_list[0].expression = p[0]
        self.parameters_list[1].expression = p[1]

    def set_param(self, p):
        # p = [kplus, kminus]
        self.parameters_list[0].expression = p[0]
        self.parameters_list[1].expression = p[1]
        


if __name__ == "__main__":
    list_npoints = [1]#[2000, 25, 200, 2000]
    list_ntrajs = [1]#[10, 1000, 50, 10]
    type_ds = ['']#['train', 'test', 'valid', 'active']

    for d in range(len(type_ds)): 
        nsteps = 33
        tend = 0.05
        model = EColi(tend=tend, nsteps=nsteps)
        npoints = list_npoints[d]
        ntrajs = list_ntrajs[d]
        timeline = np.linspace(0, tend, nsteps)
        trajectories = np.empty((npoints*ntrajs, nsteps, 3))
        params = np.empty((npoints*ntrajs, 1, 2))
        c=0
        for i in range(npoints):
            print(f'i={i+1}/{npoints}')
            model.sample_state()
            curr_param = model.sample_param()
            st = time.time()
            algorithm = 'Tau-Leaping' #'SSA'
            trajs = model.run(algorithm = algorithm,number_of_trajectories=ntrajs)
            end = time.time()-st
            print(f'Time to generate {ntrajs} trajectories = {end}')
            print(f'Time to generate single trajectory = {end/ntrajs}')

            #fig,axs = plt.subplots(3)

            for j in range(ntrajs):
                trajectories[c,:,0] = trajs[j]['N']
                trajectories[c,:,1] = trajs[j]['C']
                trajectories[c,:,2] = trajs[j]['S']
                params[c, 0] = curr_param
                c += 1

                #axs[0].plot(timeline, trajectories[c-1,:,0],c='b')
                #axs[1].plot(timeline, trajectories[c-1,:,1],c='r')
                #axs[2].plot(timeline, trajectories[c-1,:,2],c='g')
            #fig.savefig(f'EColi/plots/trajs_point_{i}')
            #plt.close()
        '''
        filename = 'EColi/EColi2_'+type_ds[d]+f'_set_H={nsteps-1}_{list_npoints[d]}x{list_ntrajs[d]}.pickle'
        data = {'init':np.concatenate((params,trajectories[:,:1]),axis=2),'trajs': trajectories[:,1:]}
        with open(filename, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)        
        '''