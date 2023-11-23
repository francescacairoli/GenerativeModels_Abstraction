import numpy as np
import gillespy2
import matplotlib.pyplot as plt
import pickle
import time
from tqdm import tqdm

class MAPK(gillespy2.Model):
    def __init__(self, nsteps, timestep):
        # First call the gillespy2.Model initializer.
        super().__init__(self)

        # Define parameters for the rates of creation and dissociation.
        V1 = gillespy2.Parameter(name='V1', expression=0.1)
        n = gillespy2.Parameter(name='n', expression=1)
        Kl = gillespy2.Parameter(name='Kl', expression=9)
        K1 = gillespy2.Parameter(name='K1', expression=10)
        V2 = gillespy2.Parameter(name='V2', expression=0.25)
        K2 = gillespy2.Parameter(name='K2', expression=8)
        k3 = gillespy2.Parameter(name='k3', expression=0.025)
        K3 = gillespy2.Parameter(name='K3', expression=15)
        k4 = gillespy2.Parameter(name='k4', expression=0.025)
        K4 = gillespy2.Parameter(name='K4', expression=15)
        V5 = gillespy2.Parameter(name='V5', expression=0.75)
        K5 = gillespy2.Parameter(name='K5', expression=15)
        V6 = gillespy2.Parameter(name='V6', expression=0.75)
        K6 = gillespy2.Parameter(name='K6', expression=15)
        k7 = gillespy2.Parameter(name='k7', expression=0.025)
        K7 = gillespy2.Parameter(name='K7', expression=15)
        k8 = gillespy2.Parameter(name='k8', expression=0.025)
        K8 = gillespy2.Parameter(name='K8', expression=15)
        V9 = gillespy2.Parameter(name='V9', expression=0.5)
        K9 = gillespy2.Parameter(name='K9', expression=15)
        V10 = gillespy2.Parameter(name='V10', expression=0.5)
        K10 = gillespy2.Parameter(name='K10', expression=15)
        self.parameters_list = [V1,n,Kl,K1,V2,K2,k3,K3,k4,K4,V5,K5,V6,K6,k7,K7,k8,K8,V9,K9,V10,K10]
        self.add_parameter(self.parameters_list)

        # Define variables for the molecular species representing M & D.
        M3K = gillespy2.Species(name='M3K', initial_value=50)
        M3Kp = gillespy2.Species(name='M3Kp', initial_value=50)
        M2K = gillespy2.Species(name='M2K', initial_value=100)
        M2Kp = gillespy2.Species(name='M2Kp', initial_value=100)
        M2Kpp = gillespy2.Species(name='M2Kpp', initial_value=100)
        MAPK = gillespy2.Species(name='MAPK', initial_value=100)
        MAPKp = gillespy2.Species(name='MAPKp', initial_value=100)
        MAPKpp = gillespy2.Species(name='MAPKpp', initial_value=100)
        
        self.species_list = [M3K,M3Kp,M2K,M2Kp,M2Kpp,MAPK,MAPKp,MAPKpp]
        self.add_species(self.species_list)

        # The list of reactants and products for a Reaction object are
        # each a Python dictionary in which the dictionary keys are
        # Species objects and the values are stoichiometries of the
        # species in the reaction.
        r1 = gillespy2.Reaction(name="r1", propensity_function='V1*M3K/( (1+(MAPKpp/Kl)**n)*(K1+M3K) )',
                                 reactants={M3K:1}, products={M3Kp:1})

        r2 = gillespy2.Reaction(name="r2", propensity_function='V2*M3Kp/(K2+M3Kp)',
                                 reactants={M3Kp:1}, products={M3K:1})
        
        r3 = gillespy2.Reaction(name="r3", propensity_function='k3*M3Kp*M2K/(K3+M2K)',
                         reactants={M2K:1}, products={M2Kp:1})

        r4 = gillespy2.Reaction(name="r4", propensity_function='k4*M3Kp*M2Kp/(K4+M2Kp)',
                                 reactants={M2Kp:1}, products={M2Kpp:1})
        
        r5 = gillespy2.Reaction(name="r5", propensity_function='V5*M2Kpp/(K5+M2Kpp)',
                                 reactants={M2Kpp:1}, products={M2Kp:1})
        
        r6 = gillespy2.Reaction(name="r6", propensity_function='V6*M2Kp/(K6+M2Kp)',
                         reactants={M2Kp:1}, products={M2K:1})

        r7 = gillespy2.Reaction(name="r7", propensity_function='k7*M2Kpp*MAPK/(K7+MAPK)',
                                 reactants={MAPK:1}, products={MAPKp:1})
        
        r8 = gillespy2.Reaction(name="r8", propensity_function='k8*M2Kpp*MAPKp/(K8+MAPKp)',
                                 reactants={MAPKp:1}, products={MAPKpp:1})
        
        r9 = gillespy2.Reaction(name="r9", propensity_function='V9*MAPKpp/(K9+MAPKpp)',
                         reactants={MAPKpp:1}, products={MAPKp:1})

        r10 = gillespy2.Reaction(name="r10", propensity_function='V10*MAPKp/(K10+MAPKp)',
                                 reactants={MAPKp:1}, products={MAPK:1})
        
        self.add_reaction([r1, r2, r3, r4, r5, r6, r7, r8, r9, r10])

        # Set the timespan for the simulation.
        self.timespan(np.arange(start=0, stop=(nsteps+1)*timestep, step=timestep))


    def set_obs_state(self):
        
        mapkpp = np.random.randint(low=10, high=290)
        self.species_list[-1].initial_value = mapkpp

        return mapkpp

    def set_nonobs_state(self, obs_state):
        nb_unobs_species = 7
        set_of_non_obs_states = np.empty(nb_unobs_species)
        m3k = np.random.randint(low=0, high=100)
        m2k = np.random.randint(low=0, high=300)
        m2kp = np.random.randint(low=0, high=int(300-m2k))
        mapk = np.random.randint(low=0, high=int(300-obs_state))
        mapkp = np.random.randint(low=0, high=int(300-mapk))
        
        set_of_non_obs_states = np.array([m3k, 100-m3k, m2k, m2kp, 300-m2k-m2kp, mapk, mapkp])
        for i in range(nb_unobs_species):
            self.species_list[i].initial_value = set_of_non_obs_states[i]
    
    def set_param(self):
        
        param = (2.5 - 0.1)*np.random.rand()+0.1
        self.parameters_list[0].expression = param #V1

        return param

list_npoints = [2000, 25, 200, 2000]
list_ntrajs = [50, 1000, 100, 50]
type_ds = ['train', 'test', 'valid', 'active']

for d in range(len(type_ds)): 
    nsteps = 32
    timestep = 60
    model = MAPK(nsteps,timestep)
    npoints = list_npoints[d]
    ntrajs = list_ntrajs[d]
    timeline = np.arange(start=0, stop=(nsteps+1)*timestep, step=timestep)
    trajectories = np.empty((npoints*ntrajs, nsteps+1, 1))
    params = np.empty((npoints*ntrajs, 1, 1))
    c=0
    for i in tqdm(range(npoints)):
        #print(f'i={i+1}/{npoints}')
        obs_state = model.set_obs_state()
        curr_param = model.set_param()
        st = time.time()
        trajs = model.run(algorithm = "SSA",number_of_trajectories=ntrajs)
        end = time.time()-st
        #print(f'Time to generate {ntrajs} trajectories = {end}')
        #print(f'Time to generate single trajectory = {end/ntrajs}')

        #fig = plt.figure()

        for j in range(ntrajs):
            model.set_nonobs_state(obs_state)
            trajectories[c,:,0] = trajs[j]['MAPK']
            params[c, 0, 0] = curr_param
            c += 1

            #plt.plot(timeline, trajectories[c-1],c='b')
        #plt.savefig(f'MAPK/plots/trajs_point_{i}')
        #plt.close()
    
    filename = 'MAPK/SSA_MAPK_'+type_ds[d]+f'_set_H={nsteps}_{list_npoints[d]}x{list_ntrajs[d]}.pickle'
    data = {'init':np.concatenate((params,trajectories[:,:1]),axis=2),'trajs': trajectories[:,1:]}
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)        
    