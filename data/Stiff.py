import numpy as np
import gillespy2
import matplotlib.pyplot as plt
import pickle
import time

class Spike(gillespy2.Model):
    def __init__(self, H):
        # First call the gillespy2.Model initializer.
        super().__init__(self)

        # Define parameters for the rates of creation and dissociation.
        k1 = gillespy2.Parameter(name='k1', expression=0.001)
        k2 = gillespy2.Parameter(name='k2', expression=1)
        k3 = gillespy2.Parameter(name='k3', expression=0.1)
        k4 = gillespy2.Parameter(name='k4', expression=0.001)
        
        self.add_parameter([k1, k2, k3,k4])

        # Define variables for the molecular species representing M & D.
        A = gillespy2.Species(name='A', initial_value=2)
        B = gillespy2.Species(name='B', initial_value=2)
        self.species_list = [A, B]
        self.add_species(self.species_list)

        # The list of reactants and products for a Reaction object are
        # each a Python dictionary in which the dictionary keys are
        # Species objects and the values are stoichiometries of the
        # species in the reaction.
        r1 = gillespy2.Reaction(name="r1", propensity_function='k1',
                                 reactants={}, products={A:20})

        r2 = gillespy2.Reaction(name="r2", propensity_function='k2*A*B',
                                 reactants={A:1,B:1}, products={B:20})
        r3 = gillespy2.Reaction(name="r3", propensity_function='k3*(B-1)',
                         reactants={B:1}, products={})
        r4 = gillespy2.Reaction(name="r4", propensity_function='k4*A',
                                 reactants={A:20}, products={})
        self.add_reaction([r1, r2, r3,r4])

        # Set the timespan for the simulation.
        self.timespan(np.linspace(0, 2, H))


    def set_initial_state(self):
        a = 0#np.random.randint(1,5)
        b = np.random.randint(1,5)
        self.species_list[0].initial_value = a
        self.species_list[1].initial_value = b

list_npoints = [3]#[2000, 25, 200, 2000]
list_ntrajs = [5]#[10, 1000, 50, 10]
type_ds = ['valid']#['train', 'test', 'valid', 'active']

for d in range(len(type_ds)): 

    H = 100
    model = Spike(H)
    npoints = list_npoints[d]
    ntrajs = list_ntrajs[d]
    
    trajectories = np.empty((npoints*ntrajs, H, 3))
    c=0
    for i in range(npoints):
        print(f'i={i+1}/{npoints}')
        model.set_initial_state()
        st = time.time()
        trajs = model.run(algorithm = "SSA",number_of_trajectories=ntrajs)
        end = time.time()-st
        #print(f'Time to generate {ntrajs} trajectories = {end}')
        #print(f'Time to generate single trajectory = {end/ntrajs}')
        fig,ax = plt.subplots(2)
        for j in range(ntrajs):
            trajectories[c,:,0] = trajs[j]['A']
            trajectories[c,:,1] = trajs[j]['B']
            c += 1
       
            
            ax[0].plot(trajs[j]['time'], trajs[j]['A'], 'r')
            ax[1].plot(trajs[j]['time'], trajs[j]['B'],   'b')
            
        plt.savefig(f'Stiff/stiff_traj_{i}.png')
        plt.close()
    #filename = 'Oscillator/Oscillator_'+type_ds[d]+f'_set_H={H-1}_{list_npoints[d]}x{list_ntrajs[d]}.pickle'
    #data = {'init':trajectories[:,:1],'trajs': trajectories[:,1:]}
    #with open(filename, 'wb') as handle:
    #    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)        
    
