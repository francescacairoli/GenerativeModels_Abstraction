import numpy as np
import pickle
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 50})

names = ['eSIRS','SIR', 'Oscillator','TS', 'MAPK','EColi']
csdi_ids = [100,100,100, 100, 289,100]
cwgan_ids = [10,200,10, 10, 498,229]

_colors = ['darkgreen'] 

csdi_box_data = []
cwgan_box_data = []

fig,axs = plt.subplots(2,6,figsize =(48, 20))


for ind,n in enumerate(names):
        csdi_fold = f'score_based/save/{n}/ID_{csdi_ids[ind]}/'

        csdi = csdi_fold+'quantitative_satisf_distances_test_set_active=False.pickle'
        
        with open(csdi, 'rb') as f:
                csdi_dist = pickle.load(f)
        
        csdi_box_data = [csdi_dist["sat_diff"]]
        
        cwgan_fold = f'wgan/save/{n}/ID_{cwgan_ids[ind]}/'

        cwgan = cwgan_fold+'quantitative_satisf_distances_test_set_active=False.pickle'
        
        with open(cwgan, 'rb') as f:
                cwgan_dist = pickle.load(f)
        
        cwgan_box_data = [cwgan_dist["sat_diff"]]
        

        _bp_sbd = axs[1,ind].boxplot(csdi_box_data, patch_artist=True,labels=[names[ind]])
        for patch, color in zip(_bp_sbd['boxes'], _colors): patch.set_facecolor(color) 
        axs[1,ind].set_title('csdi')

        _bp_ga = axs[0,ind].boxplot(cwgan_box_data, patch_artist=True,labels=[names[ind]])
        for patch, color in zip(_bp_ga['boxes'], _colors): patch.set_facecolor(color) 
        axs[0,ind].set_title('cwgan-gp')



#plt.title('csdi')

#plt.ylabel('stl error')   

figname = f'plots/boxplot_stl.png'
plt.tight_layout()
fig.savefig(figname)
plt.close()

'''
fig = plt.figure(figsize =(24, 10))
_bp = plt.boxplot(cwgan_box_data, patch_artist=True,labels=names)

for patch, color in zip(_bp['boxes'], _colors): patch.set_facecolor(color) 
plt.title('cwgan-gp')
plt.ylabel('stl error')

figname = f'plots/wgan_boxplot_stl.png'

fig.savefig(figname)
plt.close()
'''