import argparse
import torch
import datetime
import json
import yaml
import sys
import os


sys.path.append(".")
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from main_model import absCSDI
from dataset import *
from utils import *


parser = argparse.ArgumentParser(description="CSDI")
parser.add_argument("--config", type=str, default="base.yaml")
parser.add_argument('--device', default='cuda:0', help='Device for Attack')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--testmissingratio", type=float, default=-1.0)
parser.add_argument(
    "--nfold", type=int, default=0, help="for 5fold test (valid value:[0-4])"
)
parser.add_argument("--target_dim", type=int, default=2)
parser.add_argument("--eval_length", type=int, default=33)
parser.add_argument("--model_name", type=str, default="eSIRS")
parser.add_argument("--unconditional", default='False')#, action="store_true"
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=1)
parser.add_argument("--ntrajs", type=int, default=10)
parser.add_argument("--nepochs", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--lr", type=float, default=0.0005)
parser.add_argument("--scaling_flag", type=eval, default=False)
parser.add_argument("--q", type=float, default=0.9)
parser.add_argument("--load", type=eval, default=False)
parser.add_argument("--rob_flag", type=eval, default=False)
parser.add_argument("--active_flag", type=eval, default=False)

args = parser.parse_args()
print(args)

path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["train"]["epochs"] = args.nepochs
config["train"]["batch_size"] = args.batch_size
config["train"]["lr"] = args.lr

config["model"]["is_unconditional"] = args.unconditional
config["model"]["test_missing_ratio"] = args.testmissingratio

print(json.dumps(config, indent=4))
if args.modelfolder == "":
    args.modelfolder = str(np.random.randint(0,500))
    foldername = f"./save/{args.model_name}/ID_{args.modelfolder}/"
    print('model folder:', foldername)
    os.makedirs(foldername, exist_ok=True)
    with open(foldername + "config.json", "w") as f:
        json.dump(config, f, indent=4)
else:
    foldername = f"./save/{args.model_name}/ID_{args.modelfolder}/"
    if args.active_flag:
        parent_foldername = foldername
        foldername = parent_foldername+'FineTune_1/'
        os.makedirs(foldername, exist_ok=True)
if args.active_flag:
    retrain_id = args.modelfolder
else:
    retrain_id = ''

train_loader, valid_loader, test_loader, active_loader = get_dataloader(
    model_name=args.model_name,
    eval_length=args.eval_length,
    target_dim=args.target_dim,
    seed=args.seed,
    nfold=args.nfold,
    batch_size=config["train"]["batch_size"],
    missing_ratio=config["model"]["test_missing_ratio"],
    scaling_flag=args.scaling_flag,
    retrain_id = retrain_id
)

args = get_model_details(args)

model = absCSDI(config, args.device,target_dim=args.target_dim).to(args.device)

if not args.load:
    if args.active_flag:
        model.load_state_dict(torch.load(parent_foldername+ "model.pth"))
        train(
            model,
            config["train"],
            train_loader,
            valid_loader=valid_loader,
            foldername=foldername,
        )
    else:
        train(
            model,
            config["train"],
            train_loader,
            valid_loader=valid_loader,
            foldername=foldername,
        )

else:
    model.load_state_dict(torch.load(foldername+ "model.pth"))

# Evaluate over the test set
if not args.load:
    evaluate(model, test_loader, nsample=args.nsample, scaler=1, foldername=foldername, ds_id = 'test')

#plot_trajectories(foldername=foldername, nsample=args.nsample)
plot_rescaled_trajectories(opt=args, foldername=foldername, dataloader=test_loader, nsample=args.nsample)

plot_histograms(opt=args, foldername=foldername, dataloader=test_loader, nsample=args.nsample)

# Compute Wasserstein distance over test set
wd = compute_wass_distance(opt=args,model=model, dataloader=test_loader, nsample=args.nsample, scaler=1, foldername=foldername)

# Evaluate over the validation set
if not args.load:
    evaluate(model, valid_loader, nsample=args.nsample, scaler=1, foldername=foldername, ds_id = 'valid')

# Evaluate average STL satisfaction
_ = avg_stl_satisfaction(opt=args, foldername = foldername, model_name = args.model_name, dataloader=test_loader, ds_id = 'test', nsample=1, rob_flag = args.rob_flag)
valid_dist_dict = avg_stl_satisfaction(opt=args, foldername = foldername, model_name = args.model_name, dataloader=valid_loader, ds_id = 'valid', nsample=1, rob_flag = args.rob_flag)

active_target = active_loader.dataset.observed_values.reshape((2000,args.ntrajs,args.eval_length,args.target_dim))
pool_init_states = active_target[:,0,0]
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF

kernel = DotProduct() + WhiteKernel()
gpr = GaussianProcessRegressor(kernel=kernel).fit(valid_dist_dict['init'], valid_dist_dict['sat_diff'])
mean_prediction, std_prediction = gpr.predict(pool_init_states, return_std=True)

print('mean_prediction = ', mean_prediction.shape)
print('std_prediction = ', std_prediction.shape)

THRESHOLD = args.q
# Gaussian Process Upper Conﬁdence Bound (GP-UCB) algorithm
# upper conﬁdence bound argsimisation
UB = mean_prediction+1.96 *std_prediction
print('UB = ', UB)
err_threshold = np.quantile(UB, q=THRESHOLD)
print('Err threshold = ', err_threshold)

# select the pairs state-traj to add to the training set
queried_ds = active_target[(UB>=err_threshold)]


train_target = train_loader.dataset.observed_values.reshape((2000,args.ntrajs,args.eval_length,args.target_dim))
retrain_target = np.concatenate((train_target, queried_ds),axis=0)
retrain_target = retrain_target.reshape((retrain_target.shape[0]*retrain_target.shape[1],retrain_target.shape[2],retrain_target.shape[3]))

act_ax = np.arange(len(mean_prediction))

fig = plt.figure(figsize=(24,16))
plt.scatter(np.nonzero(UB>=err_threshold),UB[UB>=err_threshold], color='r')
plt.plot(act_ax, mean_prediction+ 1.96 * std_prediction, label="UCB", color = "r")
plt.fill_between(
    act_ax,
    mean_prediction - 1.96 * std_prediction,
    mean_prediction + 1.96 * std_prediction,
    alpha=0.5,
    label=r"95% CI",
)
plt.legend()
plt.xlabel("initial states")
plt.ylabel("satisf. error")
_ = plt.title("error estimate")
plt.tight_layout()
fig.savefig(foldername+f"active_gp_stl_error_estimates_q={args.q}.png")
plt.close()

act_filename = f'../data/{args.model_name}/{args.model_name}_csdi{args.modelfolder}_{int(THRESHOLD*100)}perc_retrain_trajs_H={args.eval_length}_{len(retrain_target)//args.ntrajs}x{args.ntrajs}.pickle'
print(act_filename)
#act_data = {'init':retrain_target[:,:int(-args.testmissingratio)],'trajs': retrain_target[:,int(-args.testmissingratio):]}
act_data = {'trajs': retrain_target, 'n_init_states': len(retrain_target)//args.ntrajs, 'n_trajs_per_state':args.ntrajs}

with open(act_filename, 'wb') as handle:
    pickle.dump(act_data, handle, protocol=pickle.HIGHEST_PROTOCOL)  


