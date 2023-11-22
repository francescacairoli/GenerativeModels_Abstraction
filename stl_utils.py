import torch
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import stl

print("CUDA availabe = " + str(torch.cuda.is_available()))
device = torch.device("cuda") 

def eval_sir_property(signal, rob_flag = True):
	n_timesteps = signal.shape[2]
	atom = stl.Atom(var_index=1, threshold=1, lte=True) # lte = True is <=
	glob = stl.Globally(atom, unbound=True)
	phi = stl.Eventually(glob, unbound=False, time_bound=n_timesteps-1)
	if rob_flag:
		satisf = phi.quantitative(signal)
	else:
		satisf = phi.boolean(signal)
	return satisf


def eval_esirs_property(signal, rob_flag = True):
	n_timesteps = signal.shape[2]
	atom = stl.Atom(var_index=1, threshold=25, lte=True) # lte = True is <=
	glob = stl.Globally(atom, unbound=True)
	phi = stl.Eventually(glob, unbound=False, time_bound=n_timesteps-1)
	if rob_flag:
		satisf = phi.quantitative(signal)
	else:
		satisf = phi.boolean(signal)
	return satisf

def eval_toy_property(signal, rob_flag = True):
	n_timesteps = signal.shape[2]
	sum_signal = torch.sum(signal,dim=1).unsqueeze(dim=1)
	atom = stl.Equal(var_index=0, rhs=200) 
	phi = stl.Globally(atom, unbound=True)
	if rob_flag:
		satisf = phi.quantitative(sum_signal)
	else:
		satisf = phi.boolean(sum_signal)
	return satisf

def eval_oscillator_property(signal,ref, rob_flag = True):
	n_timesteps = signal.shape[2]
	sum_signal = torch.sum(signal,dim=1).unsqueeze(dim=1)
	atom = stl.Equal(var_index=0, rhs=ref) 
	phi = stl.Globally(atom, unbound=True)
	if rob_flag:
		satisf = phi.quantitative(sum_signal)
	else:
		satisf = phi.boolean(sum_signal)
	return satisf
	
def eval_oscillator_soft_property(signal, ref, rob_flag = True): 
	# eventually A < B
	n_timesteps = signal.shape[2]
	AmB_signal = signal[:,:1]-signal[:,1:2]
	atom = stl.Atom(var_index=0, threshold=0, lte=True) # lte = True is <=

	phi = stl.Eventually(atom, unbound=True)
	if rob_flag:
		satisf = phi.quantitative(AmB_signal)
	else:
		satisf = phi.boolean(AmB_signal)
	return satisf
	
def eval_toy_soft_property(signal, rob_flag = True):
	n_timesteps = signal.shape[2]
	sum_signal = torch.sum(signal,dim=1).unsqueeze(dim=1)
	atom_up = stl.Atom(var_index=0, threshold=210,lte=True) 
	atom_low = stl.Atom(var_index=0, threshold=190,lte=False) 
	atom = stl.And(atom_low, atom_up)
	phi = stl.Globally(atom, unbound=True)
	if rob_flag:
		satisf = phi.quantitative(sum_signal)
	else:
		satisf = phi.boolean(sum_signal)
	return satisf

def eval_ts_property(signal, rob_flag = True):
	n_timesteps = signal.shape[2]

	atom0 = stl.Atom(var_index=0, threshold=1, lte=True) # lte = True is <=
	glob0 = stl.Globally(atom0, unbound=True)
	phi0 = stl.Eventually(glob0, unbound=False, time_bound=n_timesteps-1)
	atom1 = stl.Atom(var_index=1, threshold=1, lte=True) # lte = True is <=
	glob1 = stl.Globally(atom1, unbound=True)
	phi1 = stl.Eventually(glob1, unbound=False, time_bound=n_timesteps-1)
	phi = stl.Or(phi0, phi1)
	if rob_flag:
		satisf = phi.quantitative(signal)
	else:
		satisf = phi.boolean(signal)
	return satisf


def eval_mapk_property(signal, rob_flag):
	n_timesteps = signal.shape[2]
	atom = stl.Atom(var_index=0, threshold=100, lte=False) # lte = True is <=
	glob = stl.Globally(atom, unbound=True)
	phi = stl.Eventually(glob, unbound=False, time_bound=n_timesteps-1)
	if rob_flag:
		satisf = phi.quantitative(signal)
	else:
		satisf = phi.boolean(signal)
	return satisf


def eval_ecoli_property(signal, ref, rob_flag = True):
	n_timesteps = signal.shape[2]
	sum_signal = torch.sum(signal,dim=1).unsqueeze(dim=1)
	atom = stl.Equal(var_index=0, rhs=ref) 
	phi = stl.Globally(atom, unbound=True)
	if rob_flag:
		satisf = phi.quantitative(sum_signal)
	else:
		satisf = phi.boolean(sum_signal)
	return satisf
