import numpy as np
from scipy.stats import gamma, norm
import time
from dataset import *
from tqdm import tqdm
from sympy.solvers import solve
from sympy import solve, exp
from sympy import Symbol
import torch
import yaml
import sys
import os
sys.path.append(".")
sys.path.append(".")
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from torch.utils.data import DataLoader

from score_based.diff_models import *
from score_based.main_model import absCSDI
from data.EColi import *
path = "config/base.yaml"
with open(path, "r") as f:
	config = yaml.safe_load(f)

config["train"]["epochs"] = 100
config["train"]["batch_size"] = 256
config["train"]["lr"] = 0.0005

config["model"]["is_unconditional"] = 'False'
config["model"]["test_missing_ratio"] = -3

device = 'cuda:0'

dataset = Dataset(model_name='EColi', target_dim=3, eval_length=35, missing_ratio=-3, seed=0, idx='train', scaling_flag=True, retrain_id = '')


def gamma_sample():
	return gamma.rvs(4, loc=0, scale=18.32)

def rotation(theta):
	return np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])

def custom_landscape(x):
	#V = 100
	#return  0.025*(np.sin(V*x[0])-np.sin(V*x[1])+2)
	return x[0]+x[1]

def compute_m0(l):

	kr = 350
	Kr = 300 * 10**(-6)
	kb = 266
	Kb = 200 * 10**(-6)
	T = 1500
	B = 240
	R = 140


	
	#x = Symbol('x')
	#a0 = solve(kr*R*(1-x)/(1-x+Kr/T)-kb*B*x/(x+Kb/T), x)
	#print(a0)
	#a0 = float(a0[0])
	a0 = 4.402511935447183e-07

	eps0 = 1.
	eps1 = -0.45
	ntar = 6
	ntsr = 13
	ktar_off = 0.02*10**(3)     #0.02mM
	ktar_on = 0.4*10**(3)       #0.4mM
	ktsr_off = 100.*10**(3)  #100mM
	ktsr_on = 10**9 #   10**6mM


	y = Symbol('y')
	m0 = solve(a0-1/(1+exp(eps0+eps1*y)*((1+l/ktar_off)/(1+l/ktar_on))**ntar*((1+l/ktsr_off)/(1+l/ktsr_on))**ntsr), y)
	m0 = float(m0[0])

	return m0

def euler_maruayma(m, m0, dt):
	tau = 15
	sigma_m = 10**(0)#0.01
	dev = sigma_m*np.sqrt(2/tau)
	N = 5
	step = dt/N
	for i in range(N):
		g = norm.rvs()
		dm = (m-m0)/tau + dev*g
		m = m+step*dm

	return m



def simulate(tend, s, r, m, sim):
	t = 0
	nsteps = 32

	dt = 0.05
	theta = 0
	v = 20*10**(-6)*np.array([np.cos(theta),np.sin(theta)]) #m/s
	pctmc_model = EColi(tend=dt, nsteps=nsteps)
	if sim == 'csdi':
		model = absCSDI(config, device,target_dim=3).to(device)
		foldername = f"./save/EColi/ID_2/"
		model.load_state_dict(torch.load(foldername+ "model.pth"))
		with torch.no_grad():
			model.eval()

		
		
	n_sim_steps = int(tend/dt)

	r_traj = np.zeros((n_sim_steps+1,2))

	r_traj[0] = r
	for c in tqdm(range(n_sim_steps)):
		L = custom_landscape(r)
		m0 = compute_m0(L) #0.65
		print(f'L = {L}, m0 = {m0}')
		par = pctmc_model.compute_rates(m,L)
		if sim == 'csdi':
			s_scaled = (s-dataset.means)/dataset.stds   
            
			ctr = torch.zeros((1,35,3))
			ctr[0,0] = torch.tensor([par[0],par[0],par[0]])
			ctr[0,1] = torch.tensor([par[1],par[1],par[1]])
			ctr[0,2] = torch.tensor([s_scaled[0],s_scaled[1],s_scaled[2]])
			xxx = dataset.build_dict_from_trajs(ctr)
			
			ini = time.time()
			samples, _, _, _, _ = model.evaluate(xxx, 1)
			print('CSDI time: ', time.time()-ini)
			

			s_scaled = samples[0,0,:,-1].detach().cpu().numpy()
			s = np.round(s_scaled*dataset.stds+dataset.means)     
            
		else: #ssa
			ini = time.time()
			pctmc_model.set_state(s)
			pctmc_model.set_param(par)
			trajs = pctmc_model.run(algorithm = "SSA",number_of_trajectories=1)
			s = np.array([trajs[0]['N'][-1],trajs[0]['C'][-1],trajs[0]['S'][-1]])
			print('SSA time: ', time.time()-ini)
		
		
		if s[0] >= 2 and s[2] == 0:
			r = r+v*dt
		else:
			theta_sample = gamma_sample()
			v = np.dot(rotation(theta_sample), v)
		r_traj[c+1] = r
		m = euler_maruayma(m,m0,dt)

		t = t+dt
		print(f's={s}, r={r}, v={v}, m = {m}')


	return r_traj

# COMPARE THE 2 TRAJS (PCTMC and SUROGATE) and the computational time
NSIM = 3
abstr_flag = 'csdi'
tend = 500
fig = plt.figure()
for i in range(NSIM):

	# initialize variables
	r0 = np.array([0,0])
	s0 = np.array([6,0,0]) # 6 flagellum all normal
	m = 1

	start = time.time()

	traj = simulate(tend, s0, r0, m, abstr_flag)

	finish = time.time()- start

	print(f'Simulation time for {abstr_flag} = ', finish)

	plt.plot(traj[:,0], traj[:,1])

fig.savefig(f'./multiscale/sim={abstr_flag}_H={tend}_nSIM={NSIM}.png')
plt.close()