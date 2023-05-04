import numpy as np
import math
import sys

import pickle

class Dataset(object):

    def __init__(self, trainset_fn, testset_fn, x_dim, y_dim, traj_len):
        self.trainset_fn = trainset_fn
        self.testset_fn = testset_fn
        self.x_dim = x_dim 
        self.y_dim = y_dim 
        self.traj_len = traj_len

    def add_valid_data(self, validset_fn):
        self.validset_fn = validset_fn

    def load_train_data(self):

        file = open(self.trainset_fn, 'rb')
        data = pickle.load(file)
        file.close()

        self.X_train_count = data["trajs"][:,:self.traj_len,:]
        self.Y_train_count = data["init"]
        
        self.train_mean = np.empty(self.x_dim)
        self.train_std = np.empty(self.x_dim)
        for i in range(self.x_dim):
            self.train_mean[i] = self.X_train_count[:,:,i].mean()
            self.train_std[i] = self.X_train_count[:,:,i].std()

        print('----', (self.X_train_count-self.train_mean).shape)
        # data scales between [-1,1]
        self.X_train = (self.X_train_count-self.train_mean)/self.train_std
        self.Y_train = (self.Y_train_count-self.train_mean)/self.train_std
        self.n_points_dataset = self.X_train.shape[0]

        self.X_train_transp = np.swapaxes(self.X_train,1,2)
        self.Y_train_transp = np.swapaxes(self.Y_train,1,2)       
        
        
    def load_test_data(self, nb_trajs_per_point=1000):

        self.n_test_traj_per_point = nb_trajs_per_point

        file = open(self.testset_fn, 'rb')
        data = pickle.load(file)
        file.close()

        X = data["trajs"][:,:self.traj_len,:]
        Y = data["init"]
        
        self.test_mean = np.empty(self.x_dim)
        self.test_std = np.empty(self.x_dim)
        for i in range(self.x_dim):
            self.test_mean[i] = X[:,:,i].mean()
            self.test_std[i] = X[:,:,i].std()


        # data scales between [-1,1]
        Xscaled = (X-self.test_mean)/self.test_std
        Yscaled = (Y-self.test_mean)/self.test_std
        print(Y.shape, Yscaled.shape)
        self.n_points_test = X.shape[0]//nb_trajs_per_point
        
        self.X_test_count = X.reshape((self.n_points_test, self.n_test_traj_per_point, self.traj_len, self.x_dim))
        self.Y_test_count = Y.reshape((self.n_points_test, self.n_test_traj_per_point, 1, self.y_dim))
        
        self.X_test = Xscaled.reshape((self.n_points_test, self.n_test_traj_per_point, self.traj_len, self.x_dim))
        self.Y_test = Yscaled.reshape((self.n_points_test, self.n_test_traj_per_point, 1, self.y_dim))
        
        self.X_test_transp = np.swapaxes(self.X_test,2,3)
        self.Y_test_transp = np.swapaxes(self.Y_test,2,3)
        

    def load_valid_data(self, nb_trajs_per_point=50):

        self.n_valid_traj_per_point = nb_trajs_per_point

        file = open(self.validset_fn, 'rb')
        data = pickle.load(file)
        file.close()

        X = data["trajs"][:,:self.traj_len,:]
        Y = data["init"]
        
        self.valid_mean = np.empty(self.x_dim)
        self.valid_std = np.empty(self.x_dim)
        for i in range(self.x_dim):
            self.valid_mean[i] = X[:,:,i].mean()
            self.valid_std[i] = X[:,:,i].std()

        Xscaled = (X-self.valid_mean)/self.valid_std
        Yscaled = (Y-self.valid_mean)/self.valid_std
        
        self.n_points_valid = X.shape[0]//nb_trajs_per_point
        
        self.X_valid_count = X.reshape((self.n_points_valid, self.n_valid_traj_per_point, self.traj_len, self.x_dim))
        self.Y_valid_count = Y.reshape((self.n_points_valid, self.n_valid_traj_per_point, 1, self.y_dim))
        
        self.X_valid = Xscaled.reshape((self.n_points_valid, self.n_valid_traj_per_point, self.traj_len, self.x_dim))
        self.Y_valid = Yscaled.reshape((self.n_points_valid, self.n_valid_traj_per_point, 1, self.y_dim))
        
        self.X_valid_transp = np.swapaxes(self.X_valid,2,3)
        self.Y_valid_transp = np.swapaxes(self.Y_valid,2,3)


    def generate_mini_batches(self, n_samples):
        
        ix = np.random.randint(0, self.X_train_transp.shape[0], n_samples)
        Xb = self.X_train_transp[ix]
        Yb = self.Y_train_transp[ix]

        return Xb, Yb
