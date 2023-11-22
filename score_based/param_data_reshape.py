import pickle
import numpy as np
import sys
import os
sys.path.append(".")
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import argparse
from model_details import *

parser = argparse.ArgumentParser(description="data_reshape")
parser.add_argument("--model_name", type=str, default='')
parser.add_argument("--Q", type=int, default=50)
parser.add_argument("--active_flag", type=eval, default=False)
parser.add_argument("--model_id", type=int, default=1)

opt = parser.parse_args()

opt = get_model_details(opt)

print(opt)
if opt.model_name == 'MAPK':
    n_obs = [50,100,1000,50]
else:
    n_obs = [10,50,1000,10]

if opt.active_flag:
    # ---- TRAIN

    train_path = '../data/'+opt.model_name+'/'+opt.model_name+f'_csdi{opt.model_id}_{opt.Q}perc_retrain_set_H={opt.traj_len+opt.p_dim}_{int(2000+(100-opt.Q)*20)}x{n_obs[0]}.pickle'
    print(train_path)
    with open(train_path, "rb") as f:
        train_dict = pickle.load(
            f
        )
    print(train_dict['init'].shape,train_dict['trajs'].shape)

       
    Xt = np.concatenate((p, s), axis=2).transpose((0,2,1))
    train_cat_trajs = np.concatenate((Xt, train_dict['trajs']), axis=1)

    new_train_dict = {'trajs': train_cat_trajs, 'n_init_states': 2000, 'n_trajs_per_state':50}
    new_train_path = '../data/'+opt.model_name+'/'+opt.model_name+f'_csdi{opt.model_id}_{opt.Q}perc_retrain_trajs_H={opt.traj_len+opt.p_dim+1}_{int(2000+(100-opt.Q)*20)}x{n_obs[0]}.pickle'


    with open(new_train_path, "wb") as f:
        pickle.dump(new_train_dict, f)
else:
    # ---- TRAIN

    train_path = '../data/'+opt.model_name+'/'+opt.model_name+f'_train_set_H={opt.traj_len}_2000x{n_obs[0]}.pickle'


    with open(train_path, "rb") as f:
        train_dict = pickle.load(
            f
        )
    p = train_dict['init'][:,:,:opt.p_dim]
    s = train_dict['init'][:,:,opt.p_dim:]
    pp = p.transpose((0,2,1)).repeat(opt.x_dim,axis=2)
    Xt = np.concatenate((pp, s), axis=1)
    train_cat_trajs = np.concatenate((Xt, train_dict['trajs']), axis=1)

    new_train_dict = {'trajs': train_cat_trajs, 'n_init_states': 2000, 'n_trajs_per_state':50}

    new_train_path = '../data/'+opt.model_name+'/'+opt.model_name+f'_train_trajs_H={opt.traj_len+opt.p_dim+1}_2000x{n_obs[0]}.pickle'
    with open(new_train_path, "wb") as f:
        pickle.dump(new_train_dict, f)

    #------------ VALID
    valid_path = '../data/'+opt.model_name+'/'+opt.model_name+f'_valid_set_H={opt.traj_len}_200x{n_obs[1]}.pickle'

    with open(valid_path, "rb") as f:
        valid_dict = pickle.load(
            f
        )

    p = valid_dict['init'][:,:,:opt.p_dim]
    s = valid_dict['init'][:,:,opt.p_dim:]
    pp = p.transpose((0,2,1)).repeat(opt.x_dim,axis=2)
    Xv = np.concatenate((pp, s), axis=1)

    valid_cat_trajs = np.concatenate((Xv, valid_dict['trajs']), axis=1)

    new_valid_dict = {'trajs': valid_cat_trajs, 'n_init_states': 200, 'n_trajs_per_state':100}

    new_valid_path = '../data/'+opt.model_name+'/'+opt.model_name+f'_valid_trajs_H={opt.traj_len+opt.p_dim+1}_200x{n_obs[1]}.pickle'
    with open(new_valid_path, "wb") as f:
        pickle.dump(new_valid_dict, f)


    #----------------------- TEST

    test_path = '../data/'+opt.model_name+'/'+opt.model_name+f'_test_set_H={opt.traj_len}_25x1000.pickle'

    with open(test_path, "rb") as f:
        test_dict = pickle.load(
            f
        )

    p = test_dict['init'][:,:,:opt.p_dim]
    s = test_dict['init'][:,:,opt.p_dim:]
    pp = p.transpose((0,2,1)).repeat(opt.x_dim,axis=2)
    Xte = np.concatenate((pp, s), axis=1)

    test_cat_trajs = np.concatenate((Xte, test_dict['trajs']), axis=1)

    new_test_dict = {'trajs': test_cat_trajs, 'n_init_states': 25, 'n_trajs_per_state': 1000}

    new_test_path = '../data/'+opt.model_name+'/'+opt.model_name+f'_test_trajs_H={opt.traj_len+opt.p_dim+1}_25x1000.pickle'
    with open(new_test_path, "wb") as f:
        pickle.dump(new_test_dict, f)


    # ---- ACTIVE

    active_path = '../data/'+opt.model_name+'/'+opt.model_name+f'_active_set_H={opt.traj_len}_2000x{n_obs[-1]}.pickle'


    with open(active_path, "rb") as f:
        active_dict = pickle.load(
            f
        )

    p = active_dict['init'][:,:,:opt.p_dim]
    s = active_dict['init'][:,:,opt.p_dim:]
    pp = p.transpose((0,2,1)).repeat(opt.x_dim,axis=2)
    Xa = np.concatenate((pp, s), axis=1)

    active_cat_trajs = np.concatenate((Xa, active_dict['trajs']), axis=1)

    new_active_dict = {'trajs': active_cat_trajs, 'n_init_states': 2000, 'n_trajs_per_state':10}

    new_active_path = '../data/'+opt.model_name+'/'+opt.model_name+f'_active_trajs_H={opt.traj_len+opt.p_dim+1}_2000x{n_obs[-1]}.pickle'
    with open(new_active_path, "wb") as f:
        pickle.dump(new_active_dict, f)