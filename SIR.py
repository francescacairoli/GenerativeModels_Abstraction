import numpy as np
import gillespy2
import matplotlib.pyplot as plt
import pickle
import time

class SIR(gillespy2.Model):
    def __init__(self, N, endtime, n_steps):
        # First call the gillespy2.Model initializer.
        super().__init__(self)
        # Define parameters for the rates of creation and dissociation.
        self.N = N
        # Define variables for the molecular species representing M & D.
        S = gillespy2.Species(name='S', initial_value=N)
        I = gillespy2.Species(name='I', initial_value=0)
        R = gillespy2.Species(name='R', initial_value=0)
        self.species_list = [S, I, R]
        self.add_species(self.species_list)

        # The list of reactants and products for a Reaction object are
        # each a Python dictionary in which the dictionary keys are
        # Species objects and the values are stoichiometries of the
        # species in the reaction.
        k1 = gillespy2.Parameter(name='k1', expression=3)
        k2 = gillespy2.Parameter(name='k2', expression=1)
        self.params_list = [k1,k2]
        self.add_parameter(self.params_list)
        

        r1 = gillespy2.Reaction(name="r1", rate=k1,
                                 reactants={S:1,I:1}, products={I:2})
        r2 = gillespy2.Reaction(name="r2", rate=k2,
                                 reactants={I:1}, products={R:1})
        self.add_reaction([r1, r2])

        # Set the timespan for the simulation.
        self.timespan(np.linspace(0, endtime, n_steps))


    def set_initial_state(self):
        s = np.random.randint(0,self.N)
        i = np.random.randint(0,self.N)
        r = np.random.randint(0,self.N)
        self.species_list[0].initial_value = s
        self.species_list[1].initial_value = i
        self.species_list[2].initial_value = r

        self.params_list[0].expression = 3/(s+i+r)

N = 200
H = 17
endtime = 8
dt = endtime/(H-1)
model = SIR(N,n_steps=H, endtime=endtime)

npoints = 500
ntrajs = 50

trajectories = np.empty((npoints*ntrajs, H, 3))
c=0
for i in range(npoints):
    print(f'i={i+1}/{npoints}')
    model.set_initial_state()
    st = time.time()
    trajs = model.run(algorithm="Tau-Leaping",number_of_trajectories=ntrajs)
    end = time.time()-st
    #print(f'Time to generate {ntrajs} trajectories = {end}')
    #print(f'Time to generate single trajectory = {end/ntrajs}')

    for j in range(ntrajs):
        trajectories[c,:,0] = trajs[j]['S']
        trajectories[c,:,1] = trajs[j]['I']
        trajectories[c,:,2] = trajs[j]['R']
        c += 1

filename = f'SIR/SIR_valid_set_H={H-1}_{npoints}x{ntrajs}.pickle'
data = {'init':trajectories[:,:1],'trajs': trajectories[:,1:]}
with open(filename, 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)        

if False:
    fig = plt.figure()
    for index in range(0, 5):
        trajectory = trajs[index]
        plt.plot(trajectory['time'], trajectory['S'], 'r')
        plt.plot(trajectory['time'], trajectory['I'],   'b')
        plt.plot(trajectory['time'], trajectory['R'],   'g')
    plt.savefig('data/SIR/last_trajs_05.png')
