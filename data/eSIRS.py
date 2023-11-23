import numpy as np
import gillespy2
import matplotlib.pyplot as plt
import pickle
import time

class eSIRS(gillespy2.Model):
    def __init__(self, N, endtime, n_steps):
        # First call the gillespy2.Model initializer.
        super().__init__(self)
        # Define parameters for the rates of creation and dissociation.
        k1 = gillespy2.Parameter(name='k1', expression=2.36012158/N)
        k2 = gillespy2.Parameter(name='k2', expression=1.6711464)
        k3 = gillespy2.Parameter(name='k3', expression=0.90665231)
        k4 = gillespy2.Parameter(name='k4', expression=0.63583386)
     
        self.add_parameter([k1, k2, k3, k4])
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

        # infection
        r1 = gillespy2.Reaction(name="r1", rate=k1,
                                 reactants={S:1,I:1}, products={I:2})
        # recovery
        r2 = gillespy2.Reaction(name="r2", rate=k2,
                                 reactants={I:1}, products={R:1})
        
        # loss of immunity
        r3 = gillespy2.Reaction(name="r3", rate=k3,
                                 reactants={R:1}, products={S:1})
        
        # external infection
        r4 = gillespy2.Reaction(name="r4", rate=k4,
                                 reactants={S:1}, products={I:1})

        self.add_reaction([r1, r2, r3, r4])

        # Set the timespan for the simulation.
        self.timespan(np.linspace(0, endtime, n_steps))


    def set_initial_state(self):
        s = np.random.randint(0,self.N)
        i = np.random.randint(0,self.N-s)
        r = self.N-i-s
        self.species_list[0].initial_value = s
        self.species_list[1].initial_value = i
        self.species_list[2].initial_value = r

N = 100
H = 33
endtime = 3.2
dt = endtime/(H-1)
model = eSIRS(N,n_steps=H, endtime=endtime)

npoints = 500
ntrajs = 50

trajectories = np.empty((npoints*ntrajs, H, 3))
c=0
for i in range(npoints):
    print(f'i={i+1}/{npoints}')
    model.set_initial_state()
    st = time.time()
    trajs = model.run(algorithm = 'SSA',number_of_trajectories=ntrajs)#Tau-Leaping
    end = time.time()-st
    #print(f'Time to generate {ntrajs} trajectories = {end}')
    #print(f'Time to generate single trajectory = {end/ntrajs}')

    for j in range(ntrajs):
        trajectories[c,:,0] = trajs[j]['S']
        trajectories[c,:,1] = trajs[j]['I']
        trajectories[c,:,2] = trajs[j]['R']
        c += 1

filename = f'eSIRS/eSIRS_valid_set_H={H-1}_{npoints}x{ntrajs}.pickle'
data = {'init':trajectories[:,:1,:2],'trajs': trajectories[:,1:,:2]}
with open(filename, 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)        


if False:
    fig = plt.figure()
    for index in range(0, 5):
        trajectory = trajs[index]
        plt.plot(trajectory['time'], trajectory['S'], 'r')
        plt.plot(trajectory['time'], trajectory['I'],   'b')
        plt.plot(trajectory['time'], trajectory['R'],   'g')
    plt.savefig('data/eSIRS/last_trajs_05.png')
