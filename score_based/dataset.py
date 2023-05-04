import pickle

import os
import re
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset



class Dataset(Dataset):
	def __init__(self, model_name, target_dim, eval_length, missing_ratio=0.0, seed=0, idx='', scaling_flag=False, retrain_id = ''):
		self.eval_length = eval_length
		self.target_dim = target_dim
		np.random.seed(seed)  # seed for ground truth choice
		self.model_name = model_name
		self.observed_values = []
		self.observed_masks = []
		self.gt_masks = []
		use_index_list = None
		if scaling_flag:
			newpath = (f"../data/{model_name}/{model_name}_scaled_"+idx+f"_missing{missing_ratio}_gtmask.pickle")
		else:
			newpath = (f"../data/{model_name}/{model_name}_"+idx+f"_missing{missing_ratio}_gtmask.pickle")
		if os.path.isfile(newpath) == False:  # if datasetfile is none, create

			if model_name =="MAPK":
				if idx == 'test':
					observed_values, mask = self.build_mask(f'../data/{model_name}/{model_name}_test_trajs_H={eval_length}_25x1000.pickle', missing_ratio, use_index_list)
				elif idx == 'valid':
					observed_values, mask = self.build_mask(f'../data/{model_name}/{model_name}_valid_trajs_H={eval_length}_200x100.pickle', missing_ratio, use_index_list)
				elif idx == 'active':
					observed_values, mask = self.build_mask(f'../data/{model_name}/{model_name}_active_trajs_H={eval_length}_2000x50.pickle', missing_ratio, use_index_list)
				else:
					if retrain_id == '':
						observed_values, mask = self.build_mask(f'../data/{model_name}/{model_name}_train_trajs_H={eval_length}_2000x50.pickle', missing_ratio, use_index_list)
					else:
						observed_values, mask = self.build_mask(f'../data/{model_name}/{model_name}_csdi{retrain_id}_{50}perc_retrain_set_H=_3000x50.pickle', missing_ratio, use_index_list)

			else:
				if idx == 'test':
					observed_values, mask = self.build_mask(f'../data/{model_name}/{model_name}_test_trajs_H={eval_length}_25x1000.pickle', missing_ratio, use_index_list)
				elif idx == 'valid':
					observed_values, mask = self.build_mask(f'../data/{model_name}/{model_name}_valid_trajs_H={eval_length}_200x50.pickle', missing_ratio, use_index_list)
				elif idx == 'active':
					observed_values, mask = self.build_mask(f'../data/{model_name}/{model_name}_active_trajs_H={eval_length}_2000x10.pickle', missing_ratio, use_index_list)
				else:
					if retrain_id == '':
						observed_values, mask = self.build_mask(f'../data/{model_name}/{model_name}_train_trajs_H={eval_length}_2000x10.pickle', missing_ratio, use_index_list)
					else:
						observed_values, mask = self.build_mask(f'../data/{model_name}/{model_name}_csdi{retrain_id}_{50}perc_retrain_set_H=_3000x10.pickle', missing_ratio, use_index_list)

			self.observed_values = observed_values
			self.observed_masks = np.ones(observed_values.shape)#mask
			self.gt_masks =  mask
			# calc mean and std and normalize values
			# (it is the same normalization as Cao et al. (2018) (https://github.com/caow13/BRITS))
			
			if scaling_flag:
				tmp_values = self.observed_values.reshape(-1, self.target_dim)
				tmp_masks = self.observed_masks.reshape(-1, self.target_dim)
				self.means = np.zeros(self.target_dim)
				self.stds = np.zeros(self.target_dim)
				for k in range(self.target_dim):
					c_data = tmp_values[:, k][tmp_masks[:, k] == 1]
					self.means[k] = c_data.mean()
					self.stds[k] = c_data.std()
				self.observed_values = (
					(self.observed_values - self.means) / self.stds * self.observed_masks
				)

			with open(newpath, "wb") as f:
				pickle.dump(
					[self.observed_values, self.observed_masks, self.gt_masks, self.means, self.stds], f
				)
		else:  # load datasetfile
			with open(newpath, "rb") as f:
				self.observed_values, self.observed_masks, self.gt_masks, self.means, self.stds = pickle.load(
					f
				)

	def __getitem__(self, org_index):
		index = org_index
		s = {
			"observed_data": self.observed_values[index],
			"observed_mask": self.observed_masks[index],
			"gt_mask": self.gt_masks[index],
			"timepoints": np.arange(self.eval_length),
		}
		return s

	def __len__(self):
		return len(self.observed_values)

	def build_mask(self, path, missing_ratio, index_list=None):
		with open(path, "rb") as f:
			datadict = pickle.load(f)
		full_trajs = datadict['trajs']
		if index_list is not None:
			full_trajs = full_trajs[index_list]
		mask = np.zeros(full_trajs.shape)
		n_steps = full_trajs.shape[1]
		if missing_ratio < 0: #only initial state is observed
			mask[:,:int(-missing_ratio)] = 1
		else:
			mask[:,:int(n_steps*(1-missing_ratio))] = 1

		print("------------------", full_trajs.shape, mask.shape)
		return full_trajs, mask


def get_dataloader(model_name, eval_length, target_dim, seed=1, nfold=None, batch_size=16, missing_ratio=0.1, scaling_flag = False, retrain_id = ''):

	# only to obtain total length of dataset
	train_dataset = Dataset(model_name=model_name, target_dim=target_dim, eval_length=eval_length, missing_ratio=missing_ratio, seed=seed, idx='train', scaling_flag=scaling_flag, retrain_id = retrain_id)
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=1)
	
	valid_dataset = Dataset(model_name=model_name, target_dim=target_dim, eval_length=eval_length, missing_ratio=missing_ratio, seed=seed, idx='valid', scaling_flag=scaling_flag)
	valid_loader = DataLoader(valid_dataset, batch_size=50, shuffle=0)

	test_dataset = Dataset(model_name=model_name, target_dim=target_dim, eval_length=eval_length, missing_ratio=missing_ratio, seed=seed, idx='test', scaling_flag=scaling_flag)
	test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=0)

	active_dataset = Dataset(model_name=model_name, target_dim=target_dim, eval_length=eval_length, missing_ratio=missing_ratio, seed=seed, idx='active', scaling_flag=scaling_flag)
	active_loader = DataLoader(active_dataset, batch_size=batch_size, shuffle=1)
	
	return train_loader, valid_loader, test_loader, active_loader
