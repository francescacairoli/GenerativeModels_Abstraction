import numpy as np
import gillespy2
import matplotlib.pyplot as plt
import pickle
import time

class LotkaVolterra(gillespy2.Model):
    def __init__(self, H):
        # First call the gillespy2.Model initializer.
        super().__init__(self)

        # Define parameters for the rates of creation and dissociation.
        k1 = gillespy2.Parameter(name='k1', expression=10)
        k2 = gillespy2.Parameter(name='k2', expression=0.1)
        k3 = gillespy2.Parameter(name='k3', expression=10)
        
        self.add_parameter([k1, k2, k3])

        # Define variables for the molecular species representing M & D.
        A = gillespy2.Species(name='A', initial_value=10)
        B = gillespy2.Species(name='B', initial_value=10)
        self.species_list = [A, B]
        self.add_species(self.species_list)

        # The list of reactants and products for a Reaction object are
        # each a Python dictionary in which the dictionary keys are
        # Species objects and the values are stoichiometries of the
        # species in the reaction.
        r1 = gillespy2.Reaction(name="r1", propensity_function='k1*A',
                                 reactants={A:1}, products={A:2})
        r2 = gillespy2.Reaction(name="r2", propensity_function='k2*(A-1)*B',
                                 reactants={A:1,B:1}, products={B:2})
        r3 = gillespy2.Reaction(name="r3", propensity_function='k3*(B-1)',
                         reactants={B:1}, products={B:0})
        self.add_reaction([r1, r2, r3])

        # Set the timespan for the simulation.
        self.timespan(np.linspace(0, 4, H))


    def set_initial_state(self):
        a = np.random.randint(500,1000)
        b = np.random.randint(500,1000)
        self.species_list[0].initial_value = a
        self.species_list[1].initial_value = b

list_npoints = [500, 25, 100, 200]
list_ntrajs = [10, 1000, 50, 10]
type_ds = ['train', 'test', 'valid', 'active']

for d in range(len(type_ds)): 

    H = 65
    model = LotkaVolterra(H)
    npoints = list_npoints[d]
    ntrajs = list_ntrajs[d]
    
    trajectories = np.empty((npoints*ntrajs, H, 2))
    c=0
    for i in range(npoints):
        print(f'i={i+1}/{npoints}')
        model.set_initial_state()
        st = time.time()
        trajs = model.run(algorithm = "SSA",number_of_trajectories=ntrajs)
        end = time.time()-st
        #print(f'Time to generate {ntrajs} trajectories = {end}')
        #print(f'Time to generate single trajectory = {end/ntrajs}')
        #fig,ax = plt.subplots(2)
        for j in range(ntrajs):
            trajectories[c,:,0] = trajs[j]['A']
            trajectories[c,:,1] = trajs[j]['B']
            c += 1
       
            
            #ax[0].plot(trajs[j]['time'], trajs[j]['A'], 'r', label="A")
            #ax[1].plot(trajs[j]['time'], trajs[j]['B'],   'b', label="B")
        #plt.legend()   
        #plt.savefig(f'Stiff/lv_traj_{i}.png')
        
        plt.close()
    filename = 'LV64/LV64_'+type_ds[d]+f'_set_H={H-1}_{list_npoints[d]}x{list_ntrajs[d]}.pickle'
    data = {'init':trajectories[:,:1],'trajs': trajectories[:,1:]}
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)        
    
