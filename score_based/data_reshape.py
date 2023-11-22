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

if opt.active_flag:
    # ---- TRAIN

    if opt.model_name == 'TS' and opt.Q == 50:
        train_path = '../data/'+opt.model_name+'/'+opt.model_name+f'_csdi{opt.model_id}_{opt.Q}perc_retrain_set_H={opt.traj_len}_{3984}x10.pickle'
    else:
        train_path = '../data/'+opt.model_name+'/'+opt.model_name+f'_csdi{opt.model_id}_{opt.Q}perc_retrain_set_H={opt.traj_len}_{int(2000+(100-opt.Q)*20)}x10.pickle'

    with open(train_path, "rb") as f:
        train_dict = pickle.load(
            f
        )

    print(train_dict['init'].shape,train_dict['trajs'].shape)
    train_cat_trajs = np.concatenate((train_dict['init'], train_dict['trajs']), axis=1)

    new_train_dict = {'trajs': train_cat_trajs, 'n_init_states': 2000, 'n_trajs_per_state':10}

    if opt.model_name == 'TS' and opt.Q == 50:
        new_train_path = '../data/'+opt.model_name+'/'+opt.model_name+f'_csdi{opt.model_id}_{opt.Q}perc_retrain_trajs_H={opt.traj_len+1}_{3984}x10.pickle'
    else:
        new_train_path = '../data/'+opt.model_name+'/'+opt.model_name+f'_csdi{opt.model_id}_{opt.Q}perc_retrain_trajs_H={opt.traj_len+1}_{int(2000+(100-opt.Q)*20)}x10.pickle'
    with open(new_train_path, "wb") as f:
        pickle.dump(new_train_dict, f)
else:
    # ---- TRAIN

    indexes = np.arange(0,opt.traj_len,4)
    #indexes = np.arange(opt.traj_len+1)
    train_path = '../data/'+opt.model_name+'/'+opt.model_name+f'_train_set_H={opt.traj_len}_500x10.pickle'

    with open(train_path, "rb") as f:
        train_dict = pickle.load(
            f
        )
    print(train_dict['init'].shape,train_dict['trajs'].shape)

    train_cat_trajs = np.concatenate((train_dict['init'], train_dict['trajs']), axis=1)[:,indexes,:opt.x_dim]

    new_train_dict = {'trajs': train_cat_trajs, 'n_init_states': 500, 'n_trajs_per_state':10}
    print(new_train_dict['trajs'].shape)
    
    new_train_path = '../data/'+opt.model_name+'/'+opt.model_name+f'_train_trajs_H={len(indexes)}_500x10.pickle'
    with open(new_train_path, "wb") as f:
        pickle.dump(new_train_dict, f)

    #------------ VALID

    valid_path = '../data/'+opt.model_name+'/'+opt.model_name+f'_valid_set_H={opt.traj_len}_100x50.pickle'

    with open(valid_path, "rb") as f:
        valid_dict = pickle.load(
            f
        )

    valid_cat_trajs = np.concatenate((valid_dict['init'], valid_dict['trajs']), axis=1)[:,indexes,:opt.x_dim]

    new_valid_dict = {'trajs': valid_cat_trajs, 'n_init_states': 100, 'n_trajs_per_state':50}

    new_valid_path = '../data/'+opt.model_name+'/'+opt.model_name+f'_valid_trajs_H={len(indexes)}_100x50.pickle'
    with open(new_valid_path, "wb") as f:
        pickle.dump(new_valid_dict, f)


    #----------------------- TEST

    test_path = '../data/'+opt.model_name+'/'+opt.model_name+f'_test_set_H={opt.traj_len}_25x1000.pickle'

    with open(test_path, "rb") as f:
        test_dict = pickle.load(
            f
        )

    test_cat_trajs = np.concatenate((test_dict['init'], test_dict['trajs']), axis=1)[:,indexes,:opt.x_dim]

    new_test_dict = {'trajs': test_cat_trajs, 'n_init_states': 25, 'n_trajs_per_state': 1000}

    new_test_path = '../data/'+opt.model_name+'/'+opt.model_name+f'_test_trajs_H={len(indexes)}_25x1000.pickle'
    with open(new_test_path, "wb") as f:
        pickle.dump(new_test_dict, f)

    # ---- ACTIVE

    active_path = '../data/'+opt.model_name+'/'+opt.model_name+f'_active_set_H={opt.traj_len}_200x10.pickle'

    with open(active_path, "rb") as f:
        active_dict = pickle.load(
            f
        )

    active_cat_trajs = np.concatenate((active_dict['init'], active_dict['trajs']), axis=1)[:,indexes,:opt.x_dim]

    new_active_dict = {'trajs': active_cat_trajs, 'n_init_states': 200, 'n_trajs_per_state':10}

    new_active_path = '../data/'+opt.model_name+'/'+opt.model_name+f'_active_trajs_H={len(indexes)}_200x10.pickle'
    with open(new_active_path, "wb") as f:
        pickle.dump(new_active_dict, f)