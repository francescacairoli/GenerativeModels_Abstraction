import numpy as np
import pickle
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 30})

model_name = 'eSIRS'

csdi_fold = f'score_based/save/{model_name}/ID_100/'

csdi_init = csdi_fold+'quantitative_satisf_distances_test_set_active=False.pickle'
csdi_active = csdi_fold+'FineTune_100_lr=1e-07_Q=50/quantitative_satisf_distances_test_set_active=True.pickle'

with open(csdi_init, 'rb') as f:
        csdi_init_dist = pickle.load(f)
with open(csdi_active, 'rb') as f:
        csdi_active_dist = pickle.load(f)

csdi_init_mean = np.mean(csdi_init_dist["sat_diff"])
csdi_init_std = np.std(csdi_init_dist["sat_diff"])

csdi_active_mean = np.mean(csdi_active_dist["sat_diff"])
csdi_active_std = np.std(csdi_active_dist["sat_diff"])

cwgan_fold = f'wgan/save/{model_name}/ID_10/'

cwgan_init = cwgan_fold+'quantitative_satisf_distances_test_set_active=False.pickle'
cwgan_active = cwgan_fold+'FineTune_100ep_lr=1e-08_50perc/quantitative_satisf_distances_test_set_active=True.pickle'

with open(cwgan_init, 'rb') as f:
        cwgan_init_dist = pickle.load(f)
with open(cwgan_active, 'rb') as f:
        cwgan_active_dist = pickle.load(f)

cwgan_init_mean = np.mean(cwgan_init_dist["sat_diff"])
cwgan_init_std = np.std(cwgan_init_dist["sat_diff"])

cwgan_active_mean = np.mean(cwgan_active_dist["sat_diff"])
cwgan_active_std = np.std(cwgan_active_dist["sat_diff"])



box_data = [csdi_init_dist["sat_diff"],csdi_active_dist["sat_diff"]]
_colors = ['darkgreen', 'limegreen'] 
 
 

fig = plt.figure(figsize =(8, 10))
_bp = plt.boxplot(box_data, patch_artist=True,labels=['init', 'active'])

for patch, color in zip(_bp['boxes'], _colors): patch.set_facecolor(color) 
plt.title(f'csdi: {model_name}')
   

figname = f'plots/{model_name}/csdi_boxplot_difference_stl_quantitative_satisfaction.png'

fig.savefig(figname)
plt.close()

box_data = [cwgan_init_dist["sat_diff"],cwgan_active_dist["sat_diff"]]

fig = plt.figure(figsize =(8, 10))
_bp = plt.boxplot(box_data, patch_artist=True,labels=['init', 'active'])

for patch, color in zip(_bp['boxes'], _colors): patch.set_facecolor(color) 
plt.title(f'cwgan-gp: {model_name}')
   

figname = f'plots/{model_name}/wgan_boxplot_difference_stl_quantitative_satisfaction.png'

fig.savefig(figname)
plt.close()