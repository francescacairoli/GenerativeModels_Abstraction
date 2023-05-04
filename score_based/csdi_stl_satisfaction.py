import os
import sys
sys.path.append(".")

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import torch
import pickle
import numpy as np 
import pandas as pd
from stl_utils import *
from DatasetStd import *
import matplotlib.pyplot as plt

#model_name = 'eSIRS'
#idx = '20230310_115007'
idx = '20230310_115320'
model_name = 'SIR'

datafolder = model_name+'_fold0_'+idx
nsample = 100 # number of generated sample

if model_name == "eSIRS":
    species_labels = ["S", "I"]
else:
    species_labels = ["S", "I", "R"]

n_test_points = 25
n_test_trajs = 1000

path = 'save/'+datafolder+'/generated_outputs_nsample' + str(nsample) + '.pk' 
with open(path, 'rb') as f:
	gen_samples,all_target,all_evalpoint,all_observed,all_observed_time,scaler,mean_scaler = pickle.load(f)

K = gen_samples.shape[-1] #feature
L = gen_samples.shape[-2] #time length

gen_samples_res = (gen_samples[:,0]).reshape((n_test_points,n_test_trajs,L,K))

trainset_fn = "data/"+model_name+"/"+model_name+f"_train_set_H={L-1}_2000x10.pickle"
testset_fn = "data/"+model_name+"/"+model_name+f"_test_set_H={L-1}_25x1000.pickle"

ds = Dataset(trainset_fn, testset_fn, K, K, L-1)
ds.load_train_data()
ds.load_test_data()


colors = ['blue', 'orange']
leg = ['real', 'gen']


print('Computing STL satisfaction...')
ssa_sat = np.empty(n_test_points)
gen_sat = np.empty(n_test_points)
for i in range(n_test_points):
    print("\tinit_state n = ", i)
    
    ssa_trajs_i = torch.Tensor(ds.X_test_count[i].transpose((0,2,1)))
    gen_samples_rescales_i = np.round(gen_samples_res[i,:,1:].cpu().numpy()*ds.test_std+ds.test_mean)
    gen_trajs_i = torch.Tensor(gen_samples_rescales_i.transpose((0,2,1)))
    #print("SSA: ", ssa_trajs_i.shape)
    #print("GEN: ", gen_trajs_i.shape)
    if model_name == 'SIR':
        ssa_sat_i = eval_sir_property(ssa_trajs_i).float()
        gen_sat_i = eval_sir_property(gen_trajs_i).float()
    elif model_name == 'eSIRS':
        ssa_sat_i = eval_esirs_property(ssa_trajs_i).float()
        gen_sat_i = eval_esirs_property(gen_trajs_i).float()
    elif model_name == 'TS':
        ssa_sat_i = eval_ts_property(ssa_trajs_i).float()
        gen_sat_i = eval_ts_property(gen_trajs_i).float()
    elif model_name == 'Toy':
        ssa_sat_i = eval_toy_property(ssa_trajs_i).float()
        gen_sat_i = eval_toy_property(gen_trajs_i).float()
    else:
        ssa_sat_i, gen_sat_i = 0,0
    ssa_sat[i] = ssa_sat_i.mean().detach().cpu().numpy()
    gen_sat[i] = gen_sat_i.mean().detach().cpu().numpy()



fig = plt.figure()
plt.plot(np.arange(n_test_points), ssa_sat, 'o-', color=colors[0], label=leg[0])
plt.plot(np.arange(n_test_points), gen_sat, 'o-', color=colors[1], label=leg[1])
plt.legend()
plt.xlabel("test points")
plt.ylabel("exp. satisfaction")
plt.tight_layout()
figname_stl = 'score_based/save/'+datafolder+"/"+model_name+"_stl_satisfaction.png"
fig.savefig(figname_stl)


