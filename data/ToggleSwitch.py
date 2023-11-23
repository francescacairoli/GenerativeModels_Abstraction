import numpy as np
import gillespy2
import matplotlib.pyplot as plt
import pickle
import time

class ToggleSwitch(gillespy2.Model):
    def __init__(self, endtime, n_steps):
        # First call the gillespy2.Model initializer.
        super().__init__(self)
        # Define parameters for the rates of creation and dissociation.
        kp1 = gillespy2.Parameter(name='kp1', expression=100.)#100
        kd1 = gillespy2.Parameter(name='kd1', expression=1.8)#1.8
        kb1 = gillespy2.Parameter(name='kb1', expression=0.0007)#0.0014
        ku1 = gillespy2.Parameter(name='ku1', expression=0.18)#0.18
        kp2 = gillespy2.Parameter(name='kp2', expression=110.)#110
        kd2 = gillespy2.Parameter(name='kd2', expression=0.9)#0.9
        kb2 = gillespy2.Parameter(name='kb2', expression=0.00065)#0.0013
        ku2 = gillespy2.Parameter(name='ku2', expression=0.42)#0.42    
        self.add_parameter([kp1,kd1,kb1,ku1,kp2,kd2,kb2,ku2])
       
        # Define variables for the molecular species representing M & D.
        G1on = gillespy2.Species(name='G1on', initial_value=1.)
        G1off = gillespy2.Species(name='G1off', initial_value=0.)
        G2on = gillespy2.Species(name='G2on', initial_value=1.)
        G2off = gillespy2.Species(name='G2off', initial_value=0.)
        P1 = gillespy2.Species(name='P1', initial_value=0.)
        P2 = gillespy2.Species(name='P2', initial_value=0.)
        #V = gillespy2.Species(name='V', initial_value=40)
        self.species_list = [G1on,G1off,G2on,G2off,P1,P2]#,V
        self.add_species(self.species_list)

        # The list of reactants and products for a Reaction object are
        # each a Python dictionary in which the dictionary keys are
        # Species objects and the values are stoichiometries of the
        # species in the reaction.

        # production of proteins
        r1 = gillespy2.Reaction(name="r1", rate=kp1, reactants={G1on:1}, products={G1on:1,P1:1})
        r2 = gillespy2.Reaction(name="r2", rate=kp2, reactants={G2on:1}, products={G2on:1,P2:1})
        # binding
        #r3 = gillespy2.Reaction(name="r3", rate=kb2, reactants={P1:2,G2on:1}, products={G2off:1})
        #r4 = gillespy2.Reaction(name="r4", rate=kb1, reactants={P2:2,G1on:1}, products={G1off:1})
        r3 = gillespy2.Reaction(name="r3", propensity_function = 'kb2*G2on*P1*(P1-1)/2', reactants={P1:2,G2on:1}, products={G2off:1})
        r4 = gillespy2.Reaction(name="r4", propensity_function = 'kb1*G1on*P2*(P2-1)/2', reactants={P2:2,G1on:1}, products={G1off:1})
        
        # unbinding
        r5 = gillespy2.Reaction(name="r5", rate=ku2, reactants={G2off:1}, products={G2on:1,P1:2})
        r6 = gillespy2.Reaction(name="r6", rate=ku1, reactants={G1off:1}, products={G1on:1,P2:2})
        
        # degradation
        r7 = gillespy2.Reaction(name="r7", rate=kd1, reactants={P1:1}, products={P1:0})
        r8 = gillespy2.Reaction(name="r8", rate=kd2, reactants={P2:1}, products={P2:0})

        self.add_reaction([r1, r2, r3, r4, r5, r6, r7, r8])

        # Set the timespan for the simulation.
        self.timespan(np.linspace(0, endtime, n_steps))


    def set_initial_state(self):
        p1 = np.random.randint(5,20)
        p2 = np.random.randint(5,20)
        

list_npoints = [500]#[2000, 25, 200, 2000]
list_ntrajs = [50]#[10, 1000, 50, 10]
type_ds = ['valid']#['train', 'test', 'valid', 'active']


for d in range(len(type_ds)): 
    H = 33
    endtime = 3.2
    dt = endtime/(H-1)
    model = ToggleSwitch(n_steps=H, endtime=endtime)

    npoints = list_npoints[d]
    ntrajs = list_ntrajs[d]

    trajectories = np.empty((npoints*ntrajs, H, 2))
    c=0
    for i in range(npoints):
        print(f'i={i+1}/{npoints}')
        model.set_initial_state()
        st = time.time()
        trajs = model.run(algorithm = 'Tau-Leaping',number_of_trajectories=ntrajs)
        end = time.time()-st
        #print(f'Time to generate {ntrajs} trajectories = {end}')
        #print(f'Time to generate single trajectory = {end/ntrajs}')

        for j in range(ntrajs):
            trajectories[c,:,0] = trajs[j]['P1']
            trajectories[c,:,1] = trajs[j]['P2']
            c += 1

        if False:
            fig, axs = plt.subplots(2)
            for index in range(0, ntrajs):
                trajectory = trajs[index]
                axs[0].plot(trajectory['time'], trajectory['P1'], 'r')
                axs[1].plot(trajectory['time'], trajectory['P2'],   'b')
            fig.savefig(f'TS/plots/trajs_point{i}.png')
            plt.close()
    filename = 'TS/TS_'+type_ds[d]+f'_set_H={H-1}_{list_npoints[d]}x{list_ntrajs[d]}.pickle'
    data = {'init':trajectories[:,:1,:2],'trajs': trajectories[:,1:,:2]}
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)        

