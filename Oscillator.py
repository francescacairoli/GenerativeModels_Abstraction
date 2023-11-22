import numpy as np
import gillespy2
import matplotlib.pyplot as plt
import pickle
import time
from numba import jit,cuda

 
class Oscillator(gillespy2.Model):
    #@jit(target_backend='cuda')
    def __init__(self, H):
        # First call the gillespy2.Model initializer.
        super().__init__(self)

        # Define parameters for the rates of creation and dissociation.
        k1 = gillespy2.Parameter(name='k1', expression=5)
        k2 = gillespy2.Parameter(name='k2', expression=0.5)
        k3 = gillespy2.Parameter(name='k3', expression=0.5)
        self.add_parameter([k1, k2, k3])

        # Define variables for the molecular species representing M & D.
        A = gillespy2.Species(name='A', initial_value=10)
        B = gillespy2.Species(name='B', initial_value=10)
        C = gillespy2.Species(name='C', initial_value=10)
        self.species_list = [A, B, C]
        self.add_species(self.species_list)

        # The list of reactants and products for a Reaction object are
        # each a Python dictionary in which the dictionary keys are
        # Species objects and the values are stoichiometries of the
        # species in the reaction.
        r1 = gillespy2.Reaction(name="r1", propensity_function='k1*A*B/(A+B+C)',
                                 reactants={A:1,B:1}, products={A:2})
        r2 = gillespy2.Reaction(name="r2", propensity_function='k2*B*C/(A+B+C)',
                                 reactants={B:1,C:1}, products={B:2})
        r3 = gillespy2.Reaction(name="r3", propensity_function='k3*A*C/(A+B+C)',
                         reactants={C:1,A:1}, products={C:2})
        self.add_reaction([r1, r2, r3])

        # Set the timespan for the simulation.
        self.timespan(np.linspace(0, 32, H))

    @jit(target_backend='cuda')
    def set_initial_state(self):
        a = np.random.randint(20,100)
        b = np.random.randint(20,100)
        c = np.random.randint(20,100)
        self.species_list[0].initial_value = a
        self.species_list[1].initial_value = b
        self.species_list[2].initial_value = c

list_npoints = [3]#[2000, 25, 200, 2000]
list_ntrajs = [500]#[10, 1000, 50, 10]
type_ds = ['valid']#['train', 'test', 'valid', 'active']

@jit(target_backend='cuda')
def exec(model,ntrajs):

    trajs = model.run(algorithm = "Tau-Leaping",number_of_trajectories=ntrajs)
        
    return trajs

for d in range(len(type_ds)): 

    H = 33
    model = Oscillator(H)
    npoints = list_npoints[d]
    ntrajs = list_ntrajs[d]
    
    trajectories = np.empty((npoints*ntrajs, H, 3))
    c=0
    for i in range(npoints):
        print(f'i={i+1}/{npoints}')
        model.set_initial_state()
        st = time.time()

        trajs = exec(model,ntrajs)
        #trajs = model.run(algorithm = "Tau-Leaping",number_of_trajectories=ntrajs)
        
        end = time.time()-st
        print(f'Time to generate {ntrajs} trajectories = {end}')
        print(f'Time to generate single trajectory = {end/ntrajs}')
        #fig,ax = plt.subplots(3)
        for j in range(ntrajs):
            trajectories[c,:,0] = trajs[j]['A']
            trajectories[c,:,1] = trajs[j]['B']
            trajectories[c,:,2] = trajs[j]['C']
            c += 1
       
            
            #ax[0].plot(trajs[j]['time'], trajs[j]['A'], 'r')
            #ax[1].plot(trajs[j]['time'], trajs[j]['B'],   'b')
            #ax[2].plot(trajs[j]['time'], trajs[j]['C'],   'g')
        #plt.savefig(f'StiffOsc/traj_{i}.png')
        #plt.close()
    #filename = 'Oscillator/Oscillator_'+type_ds[d]+f'_set_H={H-1}_{list_npoints[d]}x{list_ntrajs[d]}.pickle'
    #data = {'init':trajectories[:,:1],'trajs': trajectories[:,1:]}
    #with open(filename, 'wb') as handle:
    #    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)        
    
