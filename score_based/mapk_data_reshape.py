import pickle
import numpy as np

model_name = 'MAPK'
traj_len = 32

# ---- TRAIN

train_path = '../data/'+model_name+'/'+model_name+f'_train_set_H={traj_len}_2000x50.pickle'


with open(train_path, "rb") as f:
    train_dict = pickle.load(
        f
    )
print(train_dict['init'].transpose((0,2,1)).shape)
print(train_dict['trajs'].shape)
train_cat_trajs = np.concatenate((train_dict['init'].transpose((0,2,1)), train_dict['trajs']), axis=1)

new_train_dict = {'trajs': train_cat_trajs, 'n_init_states': 2000, 'n_trajs_per_state':50}

new_train_path = '../data/'+model_name+'/'+model_name+f'_train_trajs_H={traj_len+2}_2000x50.pickle'
with open(new_train_path, "wb") as f:
    pickle.dump(new_train_dict, f)

#------------ VALID

valid_path = '../data/'+model_name+'/'+model_name+f'_valid_set_H={traj_len}_200x100.pickle'

with open(valid_path, "rb") as f:
    valid_dict = pickle.load(
        f
    )

valid_cat_trajs = np.concatenate((valid_dict['init'].transpose((0,2,1)), valid_dict['trajs']), axis=1)

new_valid_dict = {'trajs': valid_cat_trajs, 'n_init_states': 200, 'n_trajs_per_state':100}

new_valid_path = '../data/'+model_name+'/'+model_name+f'_valid_trajs_H={traj_len+2}_200x100.pickle'
with open(new_valid_path, "wb") as f:
    pickle.dump(new_valid_dict, f)


#----------------------- TEST

test_path = '../data/'+model_name+'/'+model_name+f'_test_set_H={traj_len}_25x1000.pickle'

with open(test_path, "rb") as f:
    test_dict = pickle.load(
        f
    )

test_cat_trajs = np.concatenate((test_dict['init'].transpose((0,2,1)), test_dict['trajs']), axis=1)

new_test_dict = {'trajs': test_cat_trajs, 'n_init_states': 25, 'n_trajs_per_state': 1000}

new_test_path = '../data/'+model_name+'/'+model_name+f'_test_trajs_H={traj_len+2}_25x1000.pickle'
with open(new_test_path, "wb") as f:
    pickle.dump(new_test_dict, f)


# ---- ACTIVE

active_path = '../data/'+model_name+'/'+model_name+f'_active_set_H={traj_len}_2000x50.pickle'

with open(active_path, "rb") as f:
    active_dict = pickle.load(
        f
    )

active_cat_trajs = np.concatenate((active_dict['init'].transpose((0,2,1)), active_dict['trajs']), axis=1)

new_active_dict = {'trajs': active_cat_trajs, 'n_init_states': 2000, 'n_trajs_per_state':10}

new_active_path = '../data/'+model_name+'/'+model_name+f'_active_trajs_H={traj_len+2}_2000x50.pickle'
with open(new_active_path, "wb") as f:
    pickle.dump(new_active_dict, f)