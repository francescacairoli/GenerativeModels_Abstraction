import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
from scipy.stats import wasserstein_distance
import pandas as pd
from stl_utils import *
from model_details import *

def train(
    model,
    config,
    train_loader,
    valid_loader=None,
    valid_epoch_interval=5,
    foldername="",
):
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
    if foldername != "":
        output_path = foldername + "model.pth"

    losses = []
    val_losses = []

    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )

    best_valid_loss = 1e10
    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        model.train() # setting the model to training mode
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:

            for batch_no, train_batch in enumerate(it, start=1):

                optimizer.zero_grad()
                
                loss = model(train_batch)
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
            lr_scheduler.step()

        losses.append(avg_loss / batch_no)
        valid_loader = None
        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval() # setting the model to evaluation mode
            avg_loss_valid = 0
            with torch.no_grad(): 
                with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
                    for batch_no, valid_batch in enumerate(it, start=1):
                        loss = model(valid_batch, is_train=0)
                        avg_loss_valid += loss.item()
                        it.set_postfix(
                            ordered_dict={
                                "valid_avg_epoch_loss": avg_loss_valid / batch_no,
                                "epoch": epoch_no,
                            },
                            refresh=False,
                        )
            if best_valid_loss > avg_loss_valid:
                best_valid_loss = avg_loss_valid
                print(
                    "\n best loss is updated to ",
                    avg_loss_valid / batch_no,
                    "at",
                    epoch_no,
                )
            val_losses.append(avg_loss_valid / batch_no)
    
    # plotting the loss function
    fig = plt.plot()
    plt.plot(np.arange(len(losses)), losses, color='b', label='train')
    if valid_loader is not None:
        plt.plot(np.arange(start=0,stop=len(losses),step=valid_epoch_interval), val_losses, color='g', label='valid')
    plt.tight_layout()
    plt.title('losses')
    plt.savefig(foldername+'losses.png')
    plt.close()
    
    if foldername != "":
        torch.save(model.state_dict(), output_path)


def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )


def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))


def calc_quantile_CRPS(target, forecast, eval_points, mean_scaler, scaler):
    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j : j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)


def evaluate(model, test_loader, nsample=100, scaler=1, mean_scaler=0, foldername="", ds_id = 'test'):
    print('Evaluation over the '+ds_id+' dataset..')
    with torch.no_grad():
        model.eval()
        mse_total = 0
        mae_total = 0
        evalpoints_total = 0

        all_target = []
        all_observed_point = []
        all_observed_time = []
        all_evalpoint = []
        all_generated_samples = []
        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            
            for batch_no, test_batch in enumerate(it, start=1):
                
                output = model.evaluate(test_batch, nsample)
                samples, c_target, eval_points, observed_points, observed_time = output
                samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                eval_points = eval_points.permute(0, 2, 1)
                observed_points = observed_points.permute(0, 2, 1)

                samples_median = samples.median(dim=1)
                all_target.append(c_target)
                all_evalpoint.append(eval_points)
                all_observed_point.append(observed_points)
                all_observed_time.append(observed_time)
                all_generated_samples.append(samples)

                mse_current = (
                    ((samples_median.values - c_target) * eval_points) ** 2
                ) * (scaler ** 2)
                mae_current = (
                    torch.abs((samples_median.values - c_target) * eval_points) 
                ) * scaler

                mse_total += mse_current.sum().item()
                mae_total += mae_current.sum().item()
                evalpoints_total += eval_points.sum().item()

                it.set_postfix(
                    ordered_dict={
                        "rmse_total": np.sqrt(mse_total / evalpoints_total),
                        "mae_total": mae_total / evalpoints_total,
                        "batch_no": batch_no,
                    },
                    refresh=True,
                )

            with open(
                foldername + "generated_"+ds_id+"_reshaped_outputs_nsample" + str(nsample) + ".pk", "wb"
            ) as f:
                pickle.dump(
                    [
                        all_generated_samples,
                        all_target,
                    ],
                    f,
                )
            with open(
                foldername + "generated_"+ds_id+"_outputs_nsample" + str(nsample) + ".pk", "wb"
            ) as f:
                all_target = torch.cat(all_target, dim=0)
                all_evalpoint = torch.cat(all_evalpoint, dim=0)
                all_observed_point = torch.cat(all_observed_point, dim=0)
                all_observed_time = torch.cat(all_observed_time, dim=0)
                all_generated_samples = torch.cat(all_generated_samples, dim=0)

                pickle.dump(
                    [
                        all_generated_samples,
                        all_target,
                        all_evalpoint,
                        all_observed_point,
                        all_observed_time,
                        scaler,
                        mean_scaler,
                    ],
                    f,
                )

            CRPS = calc_quantile_CRPS(
                all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler
            )

            with open(
                foldername + ds_id+"_result_nsample" + str(nsample) + ".pk", "wb"
            ) as f:
                pickle.dump(
                    [
                        np.sqrt(mse_total / evalpoints_total),
                        mae_total / evalpoints_total,
                        CRPS,
                    ],
                    f,
                )
                print("RMSE:", np.sqrt(mse_total / evalpoints_total))
                print("MAE:", mae_total / evalpoints_total)
                print("CRPS:", CRPS)


def compute_wass_distance(opt, model, dataloader, nsample=1, scaler=1, mean_scaler=0, foldername=""):
    print("Computing and Plotting Wasserstein distances...") 
    
    path = foldername+'generated_test_reshaped_outputs_nsample' + str(nsample) + '.pk' 
    with open(path, 'rb') as f:
        samples, target = pickle.load(f)


    N = len(samples) #nb points
    M = samples[0].shape[0] #nb trajs per point
    K = samples[0].shape[-1] #feature
    L = samples[0].shape[-2] #time length
    
    wass_dist = np.empty((N, K, L))
    
    for i in range(N):
        Ti = target[i].cpu().numpy()
        Si = samples[i].cpu().numpy()
        for m in range(K):
            for t in range(L):    
                A = Ti[:,t,m]
                B = Si[:,0,t,m]
                wd = wasserstein_distance(A, B)
                wass_dist[i,m,t] = wd
                
    avg_dist = np.mean(wass_dist, axis=0)
    markers = ['--','-.',':']
    fig = plt.figure()
    for spec in range(K):
        plt.plot(np.arange(int(-opt.testmissingratio),L), avg_dist[spec][int(-opt.testmissingratio):], markers[spec],label=opt.species_labels[spec])
    plt.legend()
    plt.xlabel("time")
    plt.ylabel("wass dist")
    plt.tight_layout()
    figname = foldername+"avg_wass_distance.png"
    fig.savefig(figname)
    plt.close()
    
    return wass_dist

def plot_histograms(opt, foldername, dataloader, nsample):

    print('Plotting histograms...')
    path = foldername+'generated_test_reshaped_outputs_nsample' + str(nsample) + '.pk' 
    with open(path, 'rb') as f:
        samples, target = pickle.load(f)

    colors = ['blue', 'orange']
    leg = ['real', 'gen']

    N = len(samples) #nb points
    M = samples[0].shape[0] #nb trajs per point
    K = samples[0].shape[-1] #feature
    L = samples[0].shape[-2] #time length

    
    bins = 50
    time_instant = -1 #last time step

    if K == 1:
        
        for kkk in range(N):
            fig = plt.figure()
            G = np.round(samples[kkk][:,0].cpu().numpy()*dataloader.dataset.stds+dataloader.dataset.means)     
            R = np.round(target[kkk].cpu().numpy()*dataloader.dataset.stds+dataloader.dataset.means)        
            
            for d in range(K):

                XXX = np.vstack((R[:, time_instant, d], G[:, time_instant, d])).T
                
                plt.hist(XXX, bins = bins, stacked=False, density=False, color=colors, label = leg)
                plt.legend()
                plt.ylabel(opt.species_labels[d])

            figname = foldername+"CSDI_{}_rescaled_hist_comparison_{}th_timestep_{}.png".format(opt.model_name,time_instant, kkk)
            
            fig.savefig(figname)
            plt.close()
        
    else:
        
        for kkk in range(N):
            fig, ax = plt.subplots(K,1, figsize = (12,K*3))
            G = np.round(samples[kkk][:,0].cpu().numpy()*dataloader.dataset.stds+dataloader.dataset.means)     
            R = np.round(target[kkk].cpu().numpy()*dataloader.dataset.stds+dataloader.dataset.means)        
            for d in range(K):

                XXX = np.vstack((R[:, time_instant, d], G[:, time_instant, d])).T
                
                ax[d].hist(XXX, bins = bins, stacked=False, density=False, color=colors)
                #ax[d].legend()
                ax[d].set_ylabel(opt.species_labels[d])

            figname = foldername+"CSDI_{}_rescaled_hist_comparison_{}th_timestep_{}.png".format(opt.model_name,time_instant, kkk)
            
            fig.savefig(figname)
            plt.close()
        

def get_quantile(samples,q,dim=1):
    return torch.quantile(samples,q,dim=dim).cpu().numpy()

def plot_rescaled_trajectories(opt, foldername, dataloader, nsample, Mred = 10):
    print('Plotting rescaled trajectories...')
    path = foldername+'generated_test_reshaped_outputs_nsample' + str(nsample) + '.pk' 
    with open(path, 'rb') as f:
        samples, target = pickle.load(f)

    colors = ['blue', 'orange']
    leg = ['real', 'gen']

    N = len(samples) #nb points
    M = samples[0].shape[0] #nb trajs per point
    K = samples[0].shape[-1] #feature
    L = samples[0].shape[-2] #time length
    tspan = range(int(-opt.testmissingratio),L)
    for dataind in range(N):

        G = np.round(samples[dataind][:,0].cpu().numpy()*dataloader.dataset.stds+dataloader.dataset.means)     
        R = np.round(target[dataind].cpu().numpy()*dataloader.dataset.stds+dataloader.dataset.means)        
        fig, axes = plt.subplots(K,figsize=(24.0, 12.0))
        
        G[:,:int(-opt.testmissingratio)] = R[:,:int(-opt.testmissingratio)].copy()
        
        for kk in range(K):
            if K == 1:
                for jj in range(Mred):
                    if jj == 0:
                        axes.plot(tspan, G[jj,int(-opt.testmissingratio):,kk], color = colors[1],linestyle='solid',label=leg[1])
                        axes.plot(tspan, R[jj,int(-opt.testmissingratio):,kk], color = colors[0],linestyle='solid',label=leg[0])
                        
                    else:
                        axes.plot(tspan, G[jj,int(-opt.testmissingratio):,kk], color = colors[1],linestyle='solid')
                        axes.plot(tspan, R[jj,int(-opt.testmissingratio):,kk], color = colors[0],linestyle='solid')
                        
                axes.set_ylabel('value')
                axes.set_xlabel('time')

            else:

                for jj in range(Mred):
                    if jj == 0:
                        axes[kk].plot(tspan, G[jj,int(-opt.testmissingratio):,kk], color = colors[1],linestyle='solid',label=leg[1])
                        axes[kk].plot(tspan, R[jj,int(-opt.testmissingratio):,kk], color = colors[0],linestyle='solid',label=leg[0])
                        
                    else:
                        axes[kk].plot(tspan, G[jj,int(-opt.testmissingratio):,kk], color = colors[1],linestyle='solid')
                        axes[kk].plot(tspan, R[jj,int(-opt.testmissingratio):,kk], color = colors[0],linestyle='solid')
                        
                plt.setp(axes[kk], ylabel=opt.species_labels[kk])
                if kk == (K-1):
                    plt.setp(axes[kk], xlabel='time')
        plt.legend()
        plt.tight_layout()
        fig.savefig(foldername+f'CSDI_{opt.model_name}_stoch_rescaled_trajectories_point_{dataind}.png')
        plt.close()

def plot_trajectories(foldername, nsample):
    print('Plotting trajectories...')
    path = foldername+'generated_test_outputs_nsample' + str(nsample) + '.pk' 
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

    for dataind in range(n_test_points):     #change to visualize a different time-series sample

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
        fig.savefig(foldername+f'stoch_dataset_point={dataind}.png')
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
        fig2.savefig(foldername+f'stoch_trajs_point={dataind}.png')
        plt.close()


def avg_stl_satisfaction(opt, foldername, dataloader, model_name, ds_id = 'test', nsample=1, rob_flag = False):

    print('Computing STL satisfaction over the '+ds_id+ ' set...')

    colors = ['blue', 'orange']
    leg = ['real', 'gen']

    path = foldername+'generated_'+ds_id+'_reshaped_outputs_nsample' + str(nsample) + '.pk' 
    with open(path, 'rb') as f:
        samples, target = pickle.load(f)

    N = len(samples) #nb points
    M = samples[0].shape[0] #nb trajs per point
    K = samples[0].shape[-1] #feature
    L = samples[0].shape[-2] #time length

    ssa_sat = np.empty(N)
    gen_sat = np.empty(N)
    for i in range(N):
        #print("\tinit_state n = ", i)
        
        rescaled_samples = np.round(samples[i].cpu().numpy()*dataloader.dataset.stds+dataloader.dataset.means)     
        rescaled_target = np.round(target[i].cpu().numpy()*dataloader.dataset.stds+dataloader.dataset.means)        
        rescaled_samples[:,0,:int(-opt.testmissingratio)] = rescaled_target[:,:int(-opt.testmissingratio)].copy()
        
        ssa_trajs_i = torch.tensor(rescaled_target.transpose((0,2,1)))[:,:,int(-opt.testmissingratio-1):]
        gen_trajs_i = torch.tensor(rescaled_samples[:,0].transpose((0,2,1)))[:,:,int(-opt.testmissingratio-1):]

        if model_name == 'SIR':
            ssa_sat_i = eval_sir_property(ssa_trajs_i,rob_flag).float()
            gen_sat_i = eval_sir_property(gen_trajs_i,rob_flag).float()
        elif model_name == 'eSIRS':
            ssa_sat_i = eval_esirs_property(ssa_trajs_i,rob_flag).float()
            gen_sat_i = eval_esirs_property(gen_trajs_i,rob_flag).float()
        elif model_name == 'TS':
            ssa_sat_i = eval_ts_property(ssa_trajs_i,rob_flag).float()
            gen_sat_i = eval_ts_property(gen_trajs_i,rob_flag).float()
        elif model_name == 'Toy':
            ssa_sat_i = eval_toy_property(ssa_trajs_i,rob_flag).float()
            gen_sat_i = eval_toy_property(gen_trajs_i,rob_flag).float()
        elif model_name == 'Oscillator':
            ssa_sat_i = eval_oscillator_property(ssa_trajs_i,rob_flag).float()
            gen_sat_i = eval_oscillator_property(gen_trajs_i,rob_flag).float()
        elif model_name == 'MAPK':
            ssa_sat_i = eval_mapk_property(ssa_trajs_i,rob_flag).float()
            gen_sat_i = eval_mapk_property(gen_trajs_i,rob_flag).float()
        elif model_name == 'EColi':
            ssa_sat_i = eval_ecoli_property(ssa_trajs_i,rob_flag).float()
            gen_sat_i = eval_ecoli_property(gen_trajs_i,rob_flag).float()
        
        else:
            ssa_sat_i, gen_sat_i = [0],[0]
        ssa_sat[i] = ssa_sat_i.mean().detach().cpu().numpy()
        gen_sat[i] = gen_sat_i.mean().detach().cpu().numpy()



    fig = plt.figure()
    plt.plot(np.arange(N), ssa_sat, 'o-', color=colors[0], label=leg[0])
    plt.plot(np.arange(N), gen_sat, 'o-', color=colors[1], label=leg[1])
    plt.legend()
    plt.xlabel("test points")
    plt.ylabel("exp. satisfaction")
    plt.tight_layout()
    if rob_flag:
        figname_stl = foldername+ds_id+"_stl_quantitative_satisfaction.png"
    else:
        figname_stl = foldername+ds_id+"_stl_boolean_satisfaction.png"

    fig.savefig(figname_stl)
    plt.close()

    init_states = np.array([target[i].cpu().numpy()[0,0] for i in range(N)])
    sat_diff = np.absolute(ssa_sat-gen_sat)#/(ssa_sat+1e-16)
    
    dist_dict = {'init': init_states, 'sat_diff':sat_diff}
    if rob_flag:
        file = open(foldername+f'quantitative_satisf_distances_'+ds_id+'_set_active={opt.active_flag}.pickle', 'wb')
    else:
        file = open(foldername+f'boolean_satisf_distances_'+ds_id+'_set_active={opt.active_flag}.pickle', 'wb')
    pickle.dump(dist_dict, file)
    file.close()

    return dist_dict