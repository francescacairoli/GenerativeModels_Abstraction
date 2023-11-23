import pickle
import numpy as np



path = 'SIR/SIR_test_set.pickle'

with open(path, "rb") as f:
    datadict = pickle.load(
        f
    )

print(datadict.keys())
traj_len = 16
X = datadict['X'][:,:,:traj_len,:2]
X = np.reshape(X,(X.shape[0]*X.shape[1],X.shape[2], X.shape[3]))

Y = np.expand_dims(datadict['Y_s0'],axis=1)[:,:,:2]
Yr = np.repeat(Y, list(2000*np.ones(Y.shape[0])), axis=0)


print(X.shape, Yr.shape)
new_dict = {'trajs': X, 'init':Yr}
with open(path, "wb") as f:
    pickle.dump(new_dict, f)



path = 'SIR/SIR_training_set.pickle'

with open(path, "rb") as f:
    datadict = pickle.load(
        f
    )
print('train: ', datadict['trajs'][:,:traj_len,:2].shape)

new_dict = {'trajs': datadict['trajs'][:,:traj_len,:2], 'init': datadict['init'][:,:,:2]}

with open(path, "wb") as f:
    pickle.dump(new_dict, f)

