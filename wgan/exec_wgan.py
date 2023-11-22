import os
import sys
import math
import time
import pickle
import argparse
import numpy as np
from tqdm import tqdm
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
import torch_two_sample 

from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt

from Dataset import *

from critic import *
from generator import *
from stl_utils import *
from model_details import *

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=1, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=256, help="size of the batches")
parser.add_argument("--gen_lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--crit_lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=480, help="dimensionality of the latent space")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--traj_len", type=int, default=16, help="number of steps")
parser.add_argument("--n_test_trajs", type=int, default=1000, help="number of trajectories per point at test time")
parser.add_argument("--x_dim", type=int, default=3, help="number of channels of x")
parser.add_argument("--y_dim", type=int, default=3, help="number of channels of y")
parser.add_argument("--model_name", type=str, default="SIR", help="name of the model")
parser.add_argument("--species_labels", type=str, default=["S", "I"], help="list of species names")
parser.add_argument("--training_flag", type=eval, default=True, help="do training or not")
parser.add_argument("--loading_id", type=str, default="", help="id of the model to load")
parser.add_argument("--active_flag", type=eval, default=False)
parser.add_argument("--Q", type=int, default=90, help="threshold quantile for the active query strategy")
parser.add_argument("--rob_flag", type=eval, default=False)
parser.add_argument("--finetune_flag", type=eval, default=True)

opt = parser.parse_args()
print(opt)

opt = get_model_details(opt)

opt.y_dim = opt.x_dim
cuda = True if torch.cuda.is_available() else False

model_name = opt.model_name
if opt.active_flag:
    if opt.model_name == 'TS':
        trainset_fn = "../data/"+model_name+"/"+model_name+f"_wgan{opt.loading_id}_{opt.Q}perc_retrain_set_H={opt.traj_len}_{4000}x10.pickle"
    else:
        if not opt.finetune_flag:
            trainset_fn = "../data/"+model_name+"/"+model_name+f"_wgan{opt.loading_id}_{opt.Q}perc_retrain_set_H={opt.traj_len}_{int(2000+(100-opt.Q)*20)}x10.pickle"
        else:
            trainset_fn = "../data/"+model_name+"/"+model_name+f"_wgan{opt.loading_id}_{opt.Q}perc_retrain_set_H={opt.traj_len}_{int((100-opt.Q)*20)}x10.pickle"


    testset_fn = "../data/"+model_name+"/"+model_name+f"_test_set_H={opt.traj_len}_25x1000.pickle"
    validset_fn = "../data/"+model_name+"/"+model_name+f"_valid_set_H={opt.traj_len}_500x50.pickle"
else:

    trainset_fn = "../data/"+model_name+"/"+model_name+f"_train_set_H={opt.traj_len}_2000x10.pickle"
    testset_fn = "../data/"+model_name+"/"+model_name+f"_test_set_H={opt.traj_len}_25x1000.pickle"
    validset_fn = "../data/"+model_name+"/"+model_name+f"_valid_set_H={opt.traj_len}_200x50.pickle"


ds = Dataset(trainset_fn, testset_fn, opt.x_dim, opt.y_dim, opt.traj_len)
ds.add_valid_data(validset_fn)
ds.load_train_data()


if opt.training_flag and not opt.active_flag:
    ID = str(np.random.randint(0,500))
    #ID = str(10)
    print("ID = ", ID)
    plots_path = "save/"+model_name+"/ID_"+ID
    parent_path = plots_path
elif opt.active_flag:
    #ID = 'Retrain_'+opt.loading_id+f'_{opt.Q}perc'
    parent_path = "save/"+model_name+f'/ID_{opt.loading_id}'
    if opt.finetune_flag:
        plots_path = parent_path+f'/FineTune_{opt.n_epochs}ep_lr={opt.gen_lr}_{opt.Q}perc'
    else:
        plots_path = parent_path+f'/Retrain_{opt.n_epochs}ep_lr={opt.gen_lr}_{opt.Q}perc'

else:
    ID = opt.loading_id
    plots_path = "save/"+model_name+"/ID_"+ID
    parent_path = plots_path


os.makedirs(plots_path, exist_ok=True)
f = open(plots_path+"/log.txt", "w")
f.write(str(opt))
f.close()

GEN_PATH = plots_path+"/generator.pt"
CRIT_PATH = plots_path+"/critic.pt"
# Loss weight for gradient penalty
lambda_gp = 10


Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def compute_gradient_penalty(C, real_samples, fake_samples, lab):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    c_interpolates = C(interpolates, lab)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=c_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]
    gradients = gradients.reshape(gradients.shape[0], opt.traj_len*opt.x_dim)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def generate_random_conditions():
    return (np.random.rand(opt.batch_size, opt.y_dim, 1)-0.5)*2

# ----------
#  Training
# ----------
# Initialize generator and critic


if opt.training_flag:

    st = time.time()

    if opt.active_flag and opt.finetune_flag:
        critic = torch.load(parent_path+"/critic.pt")
        generator = torch.load(parent_path+"/generator.pt")
        critic.train()
        generator.train()
    else:
        generator = Generator(opt.x_dim, opt.traj_len, opt.latent_dim)
        critic = Critic(opt.x_dim, opt.traj_len)

    if cuda:
        generator.cuda()
        critic.cuda()
    
    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.gen_lr, betas=(opt.b1, opt.b2))
    optimizer_C = torch.optim.Adam(critic.parameters(), lr=opt.crit_lr, betas=(opt.b1, opt.b2))

    
    batches_done = 0
    G_losses = []
    C_losses = []
    real_comp = []
    gen_comp = []
    gp_comp = []

    full_G_loss = []
    full_C_loss = []
    for epoch in range(opt.n_epochs):
        bat_per_epo = int(ds.n_points_dataset / opt.batch_size)
        n_steps = bat_per_epo * opt.n_epochs
        
        tmp_G_loss = []
        tmp_C_loss = []

        
        for i in range(bat_per_epo):
            trajs_np, conds_np = ds.generate_mini_batches(opt.batch_size)
            # Configure input
            real_trajs = Variable(Tensor(trajs_np))
            conds = Variable(Tensor(conds_np))
            # ---------------------
            #  Train Critic
            # ---------------------

            optimizer_C.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (opt.batch_size, opt.latent_dim))))

            # Generate a batch of images
            fake_trajs = generator(z, conds)
            
            # Real images
            real_validity = critic(real_trajs, conds)
            # Fake images
            fake_validity = critic(fake_trajs, conds)
            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(critic, real_trajs.data, fake_trajs.data, conds.data)
            # Adversarial loss
            c_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
            real_comp.append(torch.mean(real_validity).item())
            gen_comp.append(torch.mean(fake_validity).item())
            gp_comp.append(lambda_gp * gradient_penalty.item())
            tmp_C_loss.append(c_loss.item())
            full_C_loss.append(c_loss.item())

            c_loss.backward(retain_graph=True)
            optimizer_C.step()

            # Train the generator every n_critic steps
            if i % opt.n_critic == 0:

                # -----------------
                #  Train Generator
                # -----------------
                optimizer_G.zero_grad()
                gen_conds = Variable(Tensor(generate_random_conditions()))

                # Generate a batch of images
                gen_trajs = generator(z, gen_conds)

                # Loss measures generator's ability to fool the discriminator
                # Train on fake images
                fake_validity = critic(gen_trajs, gen_conds)
                g_loss = -torch.mean(fake_validity)
                tmp_G_loss.append(g_loss.item())
                full_G_loss.append(g_loss.item())
                g_loss.backward(retain_graph=True)
                optimizer_G.step()

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [C loss: %f] [G loss: %f]"
                    % (epoch+1, opt.n_epochs, i, bat_per_epo, c_loss.item(), g_loss.item())
                )

                batches_done += opt.n_critic
        if (epoch+1) % 500 == 0:
            torch.save(generator, plots_path+"/generator.pt")    
        C_losses.append(np.mean(tmp_C_loss))
        G_losses.append(np.mean(tmp_G_loss))
    
    training_time = time.time()-st
    print("WGAN Training time: ", training_time)
    
    f = open(plots_path+"/log.txt", "w")
    f.write("WGAN Training time: ")
    f.write(str(training_time))
    f.close()
    fig, axs = plt.subplots(2,1,figsize = (12,6))
    axs[0].plot(np.arange(opt.n_epochs), G_losses)
    axs[1].plot(np.arange(opt.n_epochs), C_losses)
    axs[0].set_title("generator loss")
    axs[1].set_title("critic loss")
    plt.tight_layout()
    fig.savefig(plots_path+"/losses.png")
    plt.close()

    fig1, axs1 = plt.subplots(2,1,figsize = (12,6))
    axs1[0].plot(np.arange(len(full_G_loss)), full_G_loss)
    axs1[1].plot(np.arange(len(full_C_loss)), full_C_loss)
    axs1[0].set_title("generator loss")
    axs1[1].set_title("critic loss")
    plt.tight_layout()
    fig1.savefig(plots_path+"/full_losses.png")
    plt.close()

    fig2, axs2 = plt.subplots(3,1, figsize = (12,9))
    axs2[0].plot(np.arange(n_steps), real_comp)
    axs2[1].plot(np.arange(n_steps), gen_comp)
    axs2[2].plot(np.arange(n_steps), gp_comp)
    axs2[0].set_title("real term")
    axs2[1].set_title("generated term")
    axs2[2].set_title("gradient penalty term")
    plt.tight_layout()
    fig2.savefig(plots_path+"/components.png")
    plt.close()

    # save the ultimate trained generator    
    torch.save(generator, GEN_PATH)
    torch.save(critic, CRIT_PATH)
else:
    # load the ultimate trained generator
    print("MODEL_PATH: ", GEN_PATH)
    generator = torch.load(GEN_PATH)
    generator.eval()
    if cuda:
        generator.cuda()

ds.load_test_data(opt.n_test_trajs)
ds.load_valid_data()


TEST_TRAJ_FLAG = opt.training_flag
VALID_TRAJ_FLAG = opt.training_flag
TEST_PLOT_FLAG = False

HIST_FLAG = False
WASS_FLAG = False
STAT_TEST = True
TEST_STL_FLAG = False
VALID_STL_FLAG = False


if TEST_TRAJ_FLAG:
    print("Computing test trajectories...")
    n_gen_trajs = ds.n_test_traj_per_point
    gen_trajectories = np.empty(shape=(ds.n_points_test, n_gen_trajs, opt.x_dim, opt.traj_len))
    for iii in range(ds.n_points_test):
        #print("Test point nb ", iii+1, " / ", ds.n_points_test)
        for jjj in range(n_gen_trajs):
            st = time.time()
            z_noise = np.random.normal(0, 1, (1, opt.latent_dim))
            #print(ds.Y_test_transp[iii,jjj])
            temp_out = generator(Variable(Tensor(z_noise)), Variable(Tensor([ds.Y_test_transp[iii,jjj]])))
            #print('WGAN time to generate one traj: ', time.time()-st)
            gen_trajectories[iii,jjj] = temp_out.detach().cpu().numpy()[0]

    trajs_dict = {"gen_trajectories": gen_trajectories}
    file = open(plots_path+'/generated_test_trajectories.pickle', 'wb')
    # dump information to that file
    pickle.dump(trajs_dict, file)
    # close the file
    file.close()
else:
    file = open(plots_path+'/generated_test_trajectories.pickle', 'rb')
    trajs_dict = pickle.load(file)
    file.close()
    gen_trajectories = trajs_dict["gen_trajectories"]


if VALID_TRAJ_FLAG:
    print("Computing valid trajectories...")
    n_gen_trajs = ds.n_valid_traj_per_point
    gen_valid_trajectories = np.empty(shape=(ds.n_points_valid, n_gen_trajs, opt.x_dim, opt.traj_len))
    for iii in range(ds.n_points_valid):
        #print("Valid int nb ", iii+1, " / ", ds.n_points_valid)
        for jjj in range(n_gen_trajs):
            z_noise = np.random.normal(0, 1, (1, opt.latent_dim))
            #print(ds.Y_test_transp[iii,jjj])
            temp_out = generator(Variable(Tensor(z_noise)), Variable(Tensor([ds.Y_valid_transp[iii,jjj]])))
            gen_valid_trajectories[iii,jjj] = temp_out.detach().cpu().numpy()[0]

    valid_trajs_dict = {"gen_trajectories": gen_valid_trajectories}
    file = open(plots_path+'/generated_valid_trajectories.pickle', 'wb')
    # dump information to that file
    pickle.dump(valid_trajs_dict, file)
    # close the file
    file.close()
else:
    file = open(plots_path+'/generated_valid_trajectories.pickle', 'rb')
    valid_trajs_dict = pickle.load(file)
    file.close()
    gen_valid_trajectories = valid_trajs_dict["gen_trajectories"]

colors = ['blue', 'orange']
leg = ['real', 'gen']

#PLOT TRAJECTORIES
if TEST_PLOT_FLAG:
    plt.rcParams.update({'font.size': 25})

    n_trajs_to_plot = 10
    print("Plotting test trajectories...")      
    tspan = range(opt.traj_len)
    for kkk in range(ds.n_points_test):
        #print("Test point nb ", kkk+1, " / ", ds.n_points_test)
        fig, axs = plt.subplots(opt.x_dim,figsize=(16.0, opt.x_dim*4))
        G = np.array([np.round(ds.HMIN+(gen_trajectories[kkk, it].T+1)*(ds.HMAX-ds.HMIN)/2).T for it in range(ds.n_test_traj_per_point)])
        R = np.array([np.round(ds.HMIN+(ds.X_test_transp[kkk, it].T+1)*(ds.HMAX-ds.HMIN)/2).T for it in range(ds.n_test_traj_per_point)])
        #G = (gen_trajectories[kkk].transpose((0,2,1))*ds.test_std+ds.test_mean).transpose((0,2,1))
        #R = ds.X_test_count[kkk].transpose((0,2,1))

        for d in range(opt.x_dim):
            
            if d == 0:
                for traj_idx in range(n_trajs_to_plot):
                    if traj_idx == 0:
                        axs[d].plot(tspan, R[traj_idx, d], color=colors[0],label=leg[0])
                        axs[d].plot(tspan, G[traj_idx,d], color=colors[1],label=leg[1])
                    else:
                        axs[d].plot(tspan, R[traj_idx, d], color=colors[0])
                        axs[d].plot(tspan, G[traj_idx,d], color=colors[1])
                        
                axs[d].set_ylabel(opt.species_labels[d])
                axs[d].legend()
            else:
                for traj_idx in range(n_trajs_to_plot):
                    axs[d].plot(tspan, R[traj_idx, d], color=colors[0])
                    axs[d].plot(tspan, G[traj_idx,d], color=colors[1])
                axs[d].set_ylabel(opt.species_labels[d])
            if d == (opt.x_dim-1):
                plt.setp(axs[d], xlabel='time')
        fig.suptitle('cwgan-gp',fontsize=40)
        plt.tight_layout()
        fig.savefig(plots_path+"/WGAN_"+opt.model_name+"_stoch_rescaled_trajectories_point_"+str(kkk)+".png")
        plt.close()

#PLOT HISTOGRAMS
if HIST_FLAG:
    plt.rcParams.update({'font.size': 25})

    bins = 50
    time_instant = -1
    print("Plotting histograms...")
    for kkk in range(ds.n_points_test):
        fig, ax = plt.subplots(opt.x_dim,1, figsize = (12,opt.x_dim*4))
        for d in range(opt.x_dim):
            G = np.array([np.round(ds.HMIN+(gen_trajectories[kkk, it].T+1)*(ds.HMAX-ds.HMIN)/2).T for it in range(ds.n_test_traj_per_point)])
            R = np.array([np.round(ds.HMIN+(ds.X_test_transp[kkk, it].T+1)*(ds.HMAX-ds.HMIN)/2).T for it in range(ds.n_test_traj_per_point)])
            #G = (gen_trajectories[kkk].transpose((0,2,1))*ds.test_std+ds.test_mean).transpose((0,2,1))
            #R = ds.X_test_count[kkk].transpose((0,2,1))

            XXX = np.vstack((R[:, d,time_instant], G[:, d, time_instant])).T
            
            ax[d].hist(XXX, bins = bins, stacked=False, density=False, color=colors, label=leg)
            ax[d].legend()
            ax[d].set_ylabel(opt.species_labels[d])

        figname = plots_path+"/WGAN_"+opt.model_name+"_rescaled_hist_comparison_{}th_timestep_{}.png".format(time_instant, kkk)
        fig.suptitle('cwgan-gp',fontsize=40)
        fig.savefig(figname)
        plt.tight_layout()
        plt.close()






if WASS_FLAG:
    plt.rcParams.update({'font.size': 22})

    dist = np.zeros(shape=(ds.n_points_test, opt.x_dim, opt.traj_len))
    print("Computing and Plotting Wasserstein distances...") 
    for kkk in tqdm(range(ds.n_points_test)):
        #print("\tinit_state n = ", kkk)
        for m in range(opt.x_dim):
            for t in range(opt.traj_len):    
                A = ds.X_test_transp[kkk,:,m,t]
                B = gen_trajectories[kkk,:,m,t]
                
                dist[kkk, m, t] = wasserstein_distance(A, B)
                

    avg_dist = np.mean(dist, axis=0)
    markers = ['--','-.',':']
    fig = plt.figure()
    for spec in range(opt.x_dim):
        plt.plot(np.arange(opt.traj_len), avg_dist[spec], markers[spec], label=opt.species_labels[spec])
    plt.legend()
    plt.xlabel("time")
    plt.ylabel("wass dist")
    plt.title('cwgan-gp')
    plt.tight_layout()

    figname = plots_path+"/WGAN_"+opt.model_name+"_avg_wass_distance.png"
    fig.savefig(figname)
    plt.close()

    distances_dict = {"gen_hist":B, "ssa_hist":A, "wass_dist":dist}
    file = open(plots_path+f'/WGAN_{opt.model_name}_avg_wass_distances.pickle', 'wb')
    # dump information to that file
    pickle.dump(distances_dict, file)
    # close the file
    file.close()

if not WASS_FLAG:
    plt.rcParams.update({'font.size': 22})

    dist = np.zeros(shape=(ds.n_points_test, opt.x_dim, opt.traj_len))
    print("Computing and Plotting Rescaled Wasserstein distances...") 
    for kkk in range(ds.n_points_test):
        #print("\tinit_state n = ", kkk)
        for t in range(opt.traj_len):    
            
            Gt = np.round(ds.HMIN+(gen_trajectories[kkk, :, :, t]+1)*(ds.HMAX-ds.HMIN)/2)
            Rt = np.round(ds.HMIN+(ds.X_test_transp[kkk, :, :, t]+1)*(ds.HMAX-ds.HMIN)/2)

            for m in range(opt.x_dim):
                A = Rt[m]
                B = Gt[m]
                
                dist[kkk, m, t] = wasserstein_distance(A, B)
                

    avg_dist = np.mean(dist, axis=0)
    markers = ['--','-.',':']
    fig = plt.figure()
    for spec in range(opt.x_dim):
        plt.plot(np.arange(opt.traj_len), avg_dist[spec], markers[spec], label=opt.species_labels[spec])
    plt.legend()
    plt.title('cwgan-gp')
    plt.xlabel("time")
    plt.ylabel("wass dist")
    plt.tight_layout()

    figname = plots_path+"/WGAN_"+opt.model_name+"_rescaled_avg_wass_distance.png"
    fig.savefig(figname)
    distances_dict = {"gen_hist":B, "ssa_hist":A, "wass_dist":dist}
    file = open(plots_path+f'/WGAN_{opt.model_name}_rescaled_avg_wass_distances.pickle', 'wb')
    # dump information to that file
    pickle.dump(distances_dict, file)
    # close the file
    file.close()


# COMPUTE THE EXPECTED SATISFACTION OF AN STL PROPERTY
if TEST_STL_FLAG:
    plt.rcParams.update({'font.size': 22})

    print('Computing STL satisfaction...')
    ssa_sat = np.empty(ds.n_points_test)
    gen_sat = np.empty(ds.n_points_test)
    for i in tqdm(range(ds.n_points_test)):
        #print("\tinit_state n = ", i)
        rescaled_ssa_trajs_i = np.transpose(np.round(ds.HMIN+(np.transpose(ds.X_test_transp[i],(0,2,1))+1)*(ds.HMAX-ds.HMIN)/2),(0,2,1)) 
        rescaled_gen_trajs_i = np.transpose(np.round(ds.HMIN+(np.transpose(gen_trajectories[i],(0,2,1))+1)*(ds.HMAX-ds.HMIN)/2),(0,2,1))
        #rescaled_gen_trajs_i = (gen_trajectories[i].transpose((0,2,1))*ds.test_std+ds.test_mean).transpose((0,2,1))
        #rescaled_ssa_trajs_i = ds.X_test_count[i].transpose((0,2,1))

        ssa_trajs_i = torch.Tensor(rescaled_ssa_trajs_i)
        gen_trajs_i = torch.Tensor(rescaled_gen_trajs_i)

        if opt.model_name == 'SIR':
            ssa_sat_i = eval_sir_property(ssa_trajs_i,opt.rob_flag).float()
            gen_sat_i = eval_sir_property(gen_trajs_i,opt.rob_flag).float()
        elif opt.model_name == 'eSIRS':
            ssa_sat_i = eval_esirs_property(ssa_trajs_i,opt.rob_flag).float()
            gen_sat_i = eval_esirs_property(gen_trajs_i,opt.rob_flag).float()
        elif opt.model_name == 'TS':
            ssa_sat_i = eval_ts_property(ssa_trajs_i,opt.rob_flag).float()
            gen_sat_i = eval_ts_property(gen_trajs_i,opt.rob_flag).float()
        elif opt.model_name == 'Toy':
            ssa_sat_i = eval_toy_property(ssa_trajs_i,opt.rob_flag).float()
            gen_sat_i = eval_toy_property(gen_trajs_i,opt.rob_flag).float()
            #ssa_sat_i = eval_toy_soft_property(ssa_trajs_i).float()
            #gen_sat_i = eval_toy_soft_property(gen_trajs_i).float()
        elif model_name == 'Oscillator':
            sum_value = torch.sum(ssa_trajs_i,dim=1)

            ssa_sat_i = eval_oscillator_property(ssa_trajs_i,sum_value,opt.rob_flag).float()
            gen_sat_i = eval_oscillator_property(gen_trajs_i,sum_value,opt.rob_flag).float()

        else:
            ssa_sat_i, gen_sat_i = 0, 0
            print('ERROR: No properties for the '+ opt.model_name +' model!')
        ssa_sat[i] = ssa_sat_i.mean().detach().cpu().numpy()
        gen_sat[i] = gen_sat_i.mean().detach().cpu().numpy()

    sat_test_diff = np.absolute(ssa_sat-gen_sat)#/(ssa_sat+1e-6)
    print("Mean stl error: ", sat_test_diff.mean())
    print("Min stl error: ", sat_test_diff.min())
    print("Max stl error: ", sat_test_diff.max())

    fig = plt.figure()
    plt.plot(np.arange(ds.n_points_test), ssa_sat, 'o-', color=colors[0], label=leg[0])
    plt.plot(np.arange(ds.n_points_test), gen_sat, 'o-', color=colors[1], label=leg[1])
    plt.legend()
    plt.title('cwgan-gp')
    plt.xlabel("test points")
    if opt.rob_flag:
        plt.ylabel("exp. robustness")
    else:
        plt.ylabel("exp. satisfaction")
    plt.tight_layout()
    if opt.rob_flag:
        figname_stl = plots_path+"/wgan_test_stl_quantitative_satisfaction.png"
    else:
        figname_stl = plots_path+"/wgan_test_stl_boolean_satisfaction.png"
    fig.savefig(figname_stl)
    plt.close()

    test_dist_dict = {'init': ds.Y_test_transp[:,0,:,0], 'sat_diff':sat_test_diff}
    file = open(plots_path+f'/quantitative_satisf_distances_test_set_active={opt.active_flag}.pickle', 'wb')
    
    pickle.dump(test_dist_dict, file)
    file.close()

    if opt.active_flag:
        stl_fn = parent_path+f'/quantitative_satisf_distances_test_set_active=False.pickle'
        with open(stl_fn, 'rb') as f:
            stl_dist = pickle.load(f)

        act_stl_fn = plots_path+f'/quantitative_satisf_distances_test_set_active=True.pickle'
        with open(act_stl_fn, 'rb') as f:
            act_stl_dist = pickle.load(f)

        plt.rcParams.update({'font.size': 22})
        colors = ['darkgreen', 'lightgreen']
        leg = ['initial', 'active']

        N = len(stl_dist["init"])
        fig = plt.figure()
        plt.plot(np.arange(N), stl_dist["sat_diff"], 'o-', color=colors[0], label=leg[0])
        plt.plot(np.arange(N), act_stl_dist["sat_diff"], 'x-', color=colors[1], label=leg[1])
        plt.legend()
        plt.title("cwgan-gp")
        plt.xlabel("test points")
        plt.ylabel("sat. diff.")

        plt.tight_layout()
        figname_stl = plots_path+"/wgan_test_difference_stl_quantitative_satisfaction.png"

        fig.savefig(figname_stl)
        plt.close()


if VALID_STL_FLAG:
    plt.rcParams.update({'font.size': 22})

    print('Computing STL satisfaction over validation set...')
    ssa_valid_sat = np.empty(ds.n_points_valid)
    gen_valid_sat = np.empty(ds.n_points_valid)
    for i in tqdm(range(ds.n_points_valid)):
        #print("\tinit_state n = ", i)
        rescaled_ssa_valid_trajs_i = np.transpose(np.round(ds.HMIN+(np.transpose(ds.X_valid_transp[i],(0,2,1))+1)*(ds.HMAX-ds.HMIN)/2),(0,2,1)) 
        rescaled_gen_valid_trajs_i = np.transpose(np.round(ds.HMIN+(np.transpose(gen_valid_trajectories[i],(0,2,1))+1)*(ds.HMAX-ds.HMIN)/2),(0,2,1))
        
        #rescaled_gen_valid_trajs_i = (gen_valid_trajectories[i].transpose((0,2,1))*ds.valid_std+ds.valid_mean).transpose((0,2,1))
        #rescaled_ssa_valid_trajs_i = ds.X_valid_count[i].transpose((0,2,1))

        ssa_trajs_i = torch.Tensor(rescaled_ssa_valid_trajs_i)
        gen_trajs_i = torch.Tensor(rescaled_gen_valid_trajs_i)

        if opt.model_name == 'SIR':
            ssa_sat_i = eval_sir_property(ssa_trajs_i).float()
            gen_sat_i = eval_sir_property(gen_trajs_i).float()
        else:
            ssa_sat_i = eval_esirs_property(ssa_trajs_i).float()
            gen_sat_i = eval_esirs_property(gen_trajs_i).float()
        
        ssa_valid_sat[i] = ssa_sat_i.mean().detach().cpu().numpy()
        gen_valid_sat[i] = gen_sat_i.mean().detach().cpu().numpy()

    sat_valid_diff = np.absolute(ssa_valid_sat-gen_valid_sat)#/(ssa_valid_sat+1e-6)

    valid_dist_dict = {'init': ds.Y_valid_transp[:,0,:,0], 'sat_diff':sat_valid_diff}
    print('DS SIZES: ', valid_dist_dict['init'].shape, valid_dist_dict['sat_diff'].shape)
    if opt.rob_flag:
        file = open(plots_path+f'/quantitative_satisf_distances_valid_set_active={opt.active_flag}.pickle', 'wb')
    else:
        file = open(plots_path+f'/boolean_satisf_distances_valid_set_active={opt.active_flag}.pickle', 'wb')
    # dump information to that file
    pickle.dump(valid_dist_dict, file)
    # close the file
    file.close()


#COMPUTE p-VALUES

if STAT_TEST:
    plt.rcParams.update({'font.size': 22})

    samples_grid = [5,10, 25, 50, 100, 200,300]
    G = len(samples_grid)
    filename = plots_path+f'/WGAN_{opt.model_name}_pvalues.pickle'
        
    if False:
        print("Statistical test...") 
            
        pvals_mean = np.empty((opt.x_dim,G))
        pvals_std = np.empty((opt.x_dim,G))
        for jj in range(G):

            pvals = torch.zeros((ds.n_points_test, opt.x_dim, opt.traj_len))
            for kkk in tqdm(range(ds.n_points_test)):
                #print("\tinit_state n = ", kkk)
                for m in range(opt.x_dim):
                    for t in range(opt.traj_len):    
                        A = torch.Tensor(ds.X_test_transp[kkk,:samples_grid[jj],m,t]).unsqueeze(1)
                        B = torch.Tensor(gen_trajectories[kkk,:samples_grid[jj],m,t]).unsqueeze(1)
                        st = torch_two_sample.statistics_diff.EnergyStatistic(samples_grid[jj], samples_grid[jj])
                        stat, dist = st(A,B, ret_matrix = True)
                        pvals[kkk, m, t] = st.pval(dist)
                            
            pvals_mean[:,jj] = np.mean(np.mean(pvals.cpu().numpy(), axis=0),axis=1)
            pvals_std[:,jj] = np.std(np.std(pvals.cpu().numpy(), axis=0),axis=1)
        
        file = open(filename, 'wb')
    
        pvalues_dict = {"samples_grid":samples_grid, "pvals_mean":pvals_mean, "pvals_std":pvals_std}
        # dump information to that file
        pickle.dump(pvalues_dict, file)
        # close the file
        file.close()

    else:
        with open(filename, 'rb') as f:
            pvalues_dict = pickle.load(f)
        pvals_mean, pvals_std = pvalues_dict["pvals_mean"], pvalues_dict["pvals_std"]

    colors = ['b','r','g']
    fig = plt.figure()
    for spec in range(opt.x_dim):
        plt.plot(np.array(samples_grid), pvals_mean[spec], color =colors[spec],label=opt.species_labels[spec])
        plt.fill_between(np.array(samples_grid), pvals_mean[spec]-1.96*pvals_std[spec],
                                    pvals_mean[spec]+1.96*pvals_std[spec], color =colors[spec],alpha =0.1)
    plt.plot(np.array(samples_grid), np.ones(G)*0.05,'k--') 
    plt.legend()
    #plt.xticks(samples_grid)
    plt.grid()

    plt.title(f'cwgan-gp: {opt.model_name}')
    plt.xlabel("nb of samples")
    plt.ylabel("p-values")
    plt.tight_layout()
    figname = plots_path+f"/WGAN_{opt.model_name}_statistical_test.png"
    fig.savefig(figname)
    plt.close()
