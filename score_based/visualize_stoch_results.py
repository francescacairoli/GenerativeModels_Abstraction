import matplotlib.pyplot as plt
import numpy as np 
import torch
import pickle
import pandas as pd

def get_quantile(samples,q,dim=1):
	return torch.quantile(samples,q,dim=dim).cpu().numpy()

idx = '20230418_094002'

dataset = 'Toy'
datafolder = dataset+'_fold0_'+idx
nsample = 1 # number of generated sample

path = './save/'+datafolder+'/generated_outputs_nsample' + str(nsample) + '.pk' 
with open(path, 'rb') as f:
	samples,all_target,all_evalpoint,all_observed,all_observed_time,scaler,mean_scaler = pickle.load(f)

all_target_np = all_target.cpu().numpy()
all_evalpoint_np = all_evalpoint.cpu().numpy()
all_observed_np = all_observed.cpu().numpy()
all_given_np = all_observed_np - all_evalpoint_np

K = samples.shape[-1] #feature
L = samples.shape[-2] #time length

n_test_points = 25
n_trajs_per_point = 1000

samples_np = samples.cpu().numpy()
samples_res = (samples[:,0]).reshape((n_test_points,n_trajs_per_point,L,K))

all_target_res = (all_target_np).reshape((n_test_points,n_trajs_per_point,L,K))
all_given_res = (all_given_np).reshape((n_test_points,n_trajs_per_point,L,K))
all_evalpoint_res = (all_evalpoint_np).reshape((n_test_points,n_trajs_per_point,L,K))

qlist =[0.05,0.25,0.5,0.75,0.95]
quantiles_imp= []
for q in qlist:
	quant = get_quantile(samples_res, q, dim=1)
	quantiles_imp.append(quant*(1-all_given_res[:,0]) + all_target_res[:,0] * all_given_res[:,0])

samples_res = samples_res.cpu().numpy()

Mred = 10

for dataind in range(n_test_points):	 #change to visualize a different time-series sample

	plt.rcParams["font.size"] = 16
	fig, axes = plt.subplots(K,figsize=(24.0, 12.0))

	for k in range(K):
		if K == 1:
			axes.plot(range(0,L), quantiles_imp[2][dataind,:,k], color = 'g',linestyle='solid',label='CSDI')
			axes.fill_between(range(0,L), quantiles_imp[0][dataind,:,k],quantiles_imp[4][dataind,:,k],
							color='g', alpha=0.3)
			for j in range(Mred):
				df = pd.DataFrame({"x":np.arange(0,L), "val":all_target_res[dataind,j,:,k], "y":all_evalpoint_res[dataind, j,:,k]})
				df = df[df.y != 0]
				df2 = pd.DataFrame({"x":np.arange(0,L), "val":all_target_res[dataind,j,:,k], "y":all_given_res[dataind, j,:,k]})
				df2 = df2[df2.y != 0]
				axes.plot(df.x,df.val, color = 'b',marker = 'o', linestyle='None')
				axes.plot(df2.x,df2.val, color = 'r',marker = 'x', linestyle='None')
			plt.setp(axes, ylabel='value')
			plt.setp(axes, xlabel='time')
		else:
			axes[k].plot(range(0,L), quantiles_imp[2][dataind,:,k], color = 'g',linestyle='solid',label='CSDI')
			axes[k].fill_between(range(0,L), quantiles_imp[0][dataind,:,k],quantiles_imp[4][dataind,:,k],
							color='g', alpha=0.3)
			for j in range(Mred):
				df = pd.DataFrame({"x":np.arange(0,L), "val":all_target_res[dataind, j,:,k], "y":all_evalpoint_res[dataind, j,:,k]})
				df = df[df.y != 0]
				df2 = pd.DataFrame({"x":np.arange(0,L), "val":all_target_res[dataind, j,:,k], "y":all_given_res[dataind, j,:,k]})
				df2 = df2[df2.y != 0]
				axes[k].plot(df.x,df.val, color = 'b',marker = 'o', linestyle='None')
				axes[k].plot(df2.x,df2.val, color = 'r',marker = 'x', linestyle='None')
			plt.setp(axes[k], ylabel='value')
			if k == 1:
				plt.setp(axes[k], xlabel='time')
	plt.legend()
	fig.savefig('./save/'+datafolder+f'/stoch_dataset_point={dataind}.png')
	plt.close()
	
	samples_scaled_res = samples_res*(1-all_given_res[:]) + all_target_res[:] * all_given_res[:]
	fig2, axes2 = plt.subplots(K,figsize=(24.0, 12.0))

	for kk in range(K):
		if K == 1:
			for jj in range(Mred):
				if jj == 0:
					axes2.plot(range(0,L), samples_scaled_res[dataind,jj,:,kk], color = 'b',linestyle='solid',label='CSDI')
					axes2.plot(range(0,L), all_target_res[dataind,jj,:,kk], color = 'orange',linestyle='solid',label='SSA')
				else:
					axes2.plot(range(0,L), samples_scaled_res[dataind,jj,:,kk], color = 'b',linestyle='solid')
					axes2.plot(range(0,L), all_target_res[dataind,jj,:,kk], color = 'orange',linestyle='solid')
					
			plt.setp(axes2, ylabel='value')
			if kk == 1:
				plt.setp(axes2[kk], xlabel='time')

		else:
			for jj in range(Mred):
				if jj == 0:
					axes2[kk].plot(range(0,L), samples_scaled_res[dataind,jj,:,kk], color = 'b',linestyle='solid',label='CSDI')
					axes2[kk].plot(range(0,L), all_target_res[dataind,jj,:,kk], color = 'orange',linestyle='solid',label='SSA')
				else:
					axes2[kk].plot(range(0,L), samples_scaled_res[dataind,jj,:,kk], color = 'b',linestyle='solid')
					axes2[kk].plot(range(0,L), all_target_res[dataind,jj,:,kk], color = 'orange',linestyle='solid')
					
			plt.setp(axes2[kk], ylabel='value')
			if kk == 1:
				plt.setp(axes2[kk], xlabel='time')
	plt.legend()
	fig2.savefig('./save/'+datafolder+f'/stoch_trajs_point={dataind}.png')
	plt.close()
	




