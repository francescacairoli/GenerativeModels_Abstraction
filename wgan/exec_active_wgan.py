import os
import sys
import math
import pickle
import argparse
import numpy as np

sys.path.append(".")
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
from torch.utils.data import DataLoader

from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})

from Dataset import *

from critic import *
from generator import *
from stl_utils import *
from model_details import *


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=1, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=256, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=480, help="dimensionality of the latent space")
parser.add_argument("--hidden_dim", type=int, default=50, help="bnn hidden dimension")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--traj_len", type=int, default=16, help="number of steps")
parser.add_argument("--n_test_trajs", type=int, default=1000, help="number of trajectories per point at test time")
parser.add_argument("--x_dim", type=int, default=3, help="number of channels of x")
parser.add_argument("--model_name", type=str, default="SIR", help="name of the model")
parser.add_argument("--species_labels", type=str, default=["S", "I"], help="list of species names")
parser.add_argument("--training_flag", type=eval, default=True, help="do training or not")
parser.add_argument("--loading_id", type=str, default="", help="id of the model to load")
parser.add_argument("--bayes", type=str, default="sklearn", help="bayesian inference technique")
parser.add_argument("--po_flag", type=eval, default=False, help="id of the model to load")
parser.add_argument("--q", type=float, default=0.9)
parser.add_argument("--rob_flag", type=eval, default=False)
parser.add_argument("--active_flag", type=eval, default=False)

opt = parser.parse_args()
print(opt)

opt = get_model_details(opt)

opt.y_dim = opt.x_dim
cuda = True if torch.cuda.is_available() else False

# import and load datasets

trainset_fn = "../data/"+opt.model_name+"/"+opt.model_name+f"_train_set_H={opt.traj_len}_2000x10.pickle"
testset_fn = "../data/"+opt.model_name+"/"+opt.model_name+f"_test_set_H={opt.traj_len}_25x1000.pickle"
validset_fn = "../data/"+opt.model_name+"/"+opt.model_name+f"_valid_set_H={opt.traj_len}_200x50.pickle" # to train the err predict
activeset_fn = "../data/"+opt.model_name+"/"+opt.model_name+f"_active_set_H={opt.traj_len}_2000x10.pickle" # pool to actively query uncertain points

ds = Dataset(trainset_fn, testset_fn, opt.x_dim, opt.y_dim, opt.traj_len)
ds.add_valid_data(validset_fn) # training set for the error generalization
ds.add_active_data(activeset_fn)
ds.load_train_data()
ds.load_test_data(opt.n_test_trajs)
ds.load_active_data()

plots_path = "save/"+opt.model_name+"/ID_"+opt.loading_id
model_path = plots_path+"/generator_{}epochs.pt".format(opt.n_epochs)

# load pretrained surrogate model
print("Model_path: ", model_path)
generator = torch.load(model_path)
generator.eval()
if cuda:
	generator.cuda()

# load the datasets of point-stl_residual pairs
if opt.rob_flag:
    residuals_fn = plots_path+f'/quantitative_satisf_distances_valid_set_active={opt.active_flag}.pickle'
else:
    residuals_fn= plots_path+f'/boolean_satisf_distances_valid_set_active={opt.active_flag}.pickle'


file = open(residuals_fn, 'rb')
active_ds = pickle.load(file) # init sat_diff
file.close()

print('residuals = ', active_ds['sat_diff'])

if opt.bayes == 'sklearn':
	from sklearn.gaussian_process import GaussianProcessRegressor
	from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF
	def optimizer(obj_func, initial_theta, bounds):
		rng = np.random.RandomState(0)
		theta_opt, func_min = initial_theta, obj_func(initial_theta, eval_gradient=False)
		for _ in range(50):
			theta = np.atleast_1d(rng.uniform(np.maximum(-2, bounds[:, 0]),
											  np.minimum(1, bounds[:, 1])))
			f = obj_func(theta, eval_gradient=False)
			if f < func_min:
				theta_opt, func_min = theta, f
		return theta_opt, func_min
	#kernel = DotProduct() + WhiteKernel()
	kernel = 0.1 * RBF(length_scale=0.1, length_scale_bounds=(1e-3, 1e3))
	gpr = GaussianProcessRegressor(kernel=kernel, optimizer=optimizer).fit(active_ds['init'], active_ds['sat_diff'])
	print(gpr.kernel_)
	print('GPR score = ',gpr.score(active_ds['init'], active_ds['sat_diff']))
	mean_prediction, std_prediction = gpr.predict(ds.Y_active_transp[:,0,:,0], return_std=True)

	print('mean_prediction = ', mean_prediction)
	print('std_prediction = ', std_prediction)

#elif bayes == 'svigp':

elif opt.bayes == 'svibnn':

	from SVI_BNNs.bnn import BNN_smMC
	bnn_smmc = BNN_smMC(model_name = model_name, input_size=args.x_dim, n_hidden=args.n_hidden, architecture_name=args.architecture)


else:
	print('Unknown Bayesian inference technique.')

THRESHOLD = opt.q
# Gaussian Process Upper Conﬁdence Bound (GP-UCB) algorithm
# upper conﬁdence bound optimisation
UB = mean_prediction+1.96 *std_prediction
print('UB = ', UB)
err_threshold = np.quantile(UB, q=THRESHOLD)
print('Err threshold = ', err_threshold)

# select the pairs state-traj to add to the training set
active_y = ds.Y_active_transp[(UB>=err_threshold)]
active_x = ds.X_active_transp[(UB>=err_threshold)]

x_act = active_x.reshape(active_x.shape[0]*active_x.shape[1], active_x.shape[2], active_x.shape[3])
y_act = active_y.reshape(active_y.shape[0]*active_y.shape[1], active_y.shape[2], active_y.shape[3])

Xactive_scaled = np.concatenate((ds.X_train_transp, x_act),axis=0) 
Yactive_scaled = np.concatenate((ds.Y_train_transp, y_act),axis=0)   

Xactive_scaled = Xactive_scaled.transpose((0,2,1))
Yactive_scaled = Yactive_scaled.transpose((0,2,1))

Xactive = ds.HMIN+(ds.HMAX-ds.HMIN)*(Xactive_scaled+1)/2
Yactive = ds.HMIN+(ds.HMAX-ds.HMIN)*(Yactive_scaled+1)/2

act_ax = np.arange(len(ds.Y_active_transp[:,0,:,0]))
print(UB[UB>=err_threshold])
fig = plt.figure(figsize=(24,16))
plt.scatter(np.nonzero(UB>=err_threshold),UB[UB>=err_threshold], color='r')
plt.plot(act_ax, mean_prediction+ 1.96 * std_prediction, label="UCB", color = "r")
#plt.fill_between(
#    act_ax,
#    mean_prediction - 1.96 * std_prediction,
#    mean_prediction + 1.96 * std_prediction,
#    alpha=0.5,
#    label=r"95% CI",
#)
plt.legend()
plt.xlabel("initial states")
plt.ylabel("satisf. error")
_ = plt.title("error estimate")
plt.tight_layout()
figname = plots_path+f"/active_gp_stl_error_estimates_q={opt.q}_rob={opt.rob_flag}.png"
fig.savefig(figname)
plt.close()

act_filename = f'../data/{opt.model_name}/{opt.model_name}_wgan{opt.loading_id}_{int(THRESHOLD*100)}perc_retrain_set_H={opt.traj_len}_{len(Xactive)//10}x10.pickle'
act_data = {'init':Yactive,'trajs': Xactive}
with open(act_filename, 'wb') as handle:
	pickle.dump(act_data, handle, protocol=pickle.HIGHEST_PROTOCOL)        

if False:
	# plot performances over the training set
	mean_prediction_train, std_prediction_train = gpr.predict(active_ds['init'], return_std=True)

	Xt = np.arange(len(ds.Y_test_transp[:,0,:,0]))
	Xv = np.arange(len(active_ds['init']))

	fig = plt.figure(figsize=(24,16))
	plt.scatter(Xv, active_ds['sat_diff'], label='train')
	plt.plot(Xv, mean_prediction_train, label="gp mean")
	plt.fill_between(
		Xv,
		mean_prediction_train - 1.96 * std_prediction_train,
		mean_prediction_train + 1.96 * std_prediction_train,
		alpha=0.5,
		label=r"95% CI",
	)
	plt.legend()
	plt.xlabel("initial states")
	plt.ylabel("satisf. error")
	_ = plt.title("error estimate")
	plt.tight_layout()
	fig.savefig(plots_path+"/gp_stl_error_estimates.png")
	plt.close()


