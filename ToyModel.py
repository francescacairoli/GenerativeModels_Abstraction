import numpy as np
import gillespy2
import matplotlib.pyplot as plt
import pickle

class ToyModel(gillespy2.Model):
    def __init__(self, N, H):
        # First call the gillespy2.Model initializer.
        super().__init__(self)

        # Define parameters for the rates of creation and dissociation.
        k1 = gillespy2.Parameter(name='k1', expression=0.1)
        k2 = gillespy2.Parameter(name='k2', expression=0.2)
        k3 = gillespy2.Parameter(name='k3', expression=0.3)
        self.add_parameter([k1, k2, k3])
        self.N = N
        # Define variables for the molecular species representing M & D.
        A = gillespy2.Species(name='A', initial_value=N)
        B = gillespy2.Species(name='B', initial_value=0)
        C = gillespy2.Species(name='C', initial_value=0)
        self.species_list = [A, B, C]
        self.add_species(self.species_list)

        # The list of reactants and products for a Reaction object are
        # each a Python dictionary in which the dictionary keys are
        # Species objects and the values are stoichiometries of the
        # species in the reaction.
        r1 = gillespy2.Reaction(name="r1", rate=k1,
                                 reactants={A:1}, products={B:1})
        r2 = gillespy2.Reaction(name="r2", rate=k2,
                                 reactants={B:1}, products={C:1})
        r3 = gillespy2.Reaction(name="r3", rate=k3,
                         reactants={C:1}, products={A:1})
        self.add_reaction([r1, r2, r3])

        # Set the timespan for the simulation.
        self.timespan(np.linspace(0, 16, H))


    def set_initial_state(self):
        a = np.random.randint(0,self.N)
        b = np.random.randint(0,self.N-a)
        c = self.N-a-b
        self.species_list[0].initial_value = a
        self.species_list[1].initial_value = b
        self.species_list[2].initial_value = c

list_npoints = [2000, 25, 200, 2000]
list_ntrajs = [10, 1000, 50, 10]
type_ds = ['train', 'test', 'valid', 'active']

for d in range(len(type_ds)): 
    N = 200
    H = 33
    model = ToyModel(N,H)
    npoints = list_npoints[d]
    ntrajs = list_ntrajs[d]
    
    trajectories = np.empty((npoints*ntrajs, H, 3))
    c=0
    for i in range(npoints):
        print(f'i={i+1}/{npoints}')
        model.set_initial_state()
        trajs = model.run(algorithm = "SSA",number_of_trajectories=ntrajs)
        for j in range(ntrajs):
            trajectories[c,:,0] = trajs[j]['A']
            trajectories[c,:,1] = trajs[j]['B']
            trajectories[c,:,2] = trajs[j]['C']
            c += 1

    filename = 'Toy/Toy_'+type_ds[d]+f'_set_H={H-1}_{list_npoints[d]}x{list_ntrajs[d]}.pickle'
    data = {'init':trajectories[:,:1],'trajs': trajectories[:,1:]}
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)        
