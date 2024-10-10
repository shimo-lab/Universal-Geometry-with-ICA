#%%
# ----------------------------------------------------------------------------
import os
import scipy
import numpy as np
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# ----------------------------------------------------------------------------

#%%
# ----------------------------------------------------------------------------
get_id = pickle.load(open('../data/text8_sgns/text8_word-to-id.pkl', 'rb'))
get_word = pickle.load(open('../data/text8_sgns/text8_id-to-word.pkl', 'rb'))
pca2_vecs = pickle.load(open(f"../data/ica_data/pca2_20230618_200633.pkl", 'rb'))
R_ica = pickle.load(open(f"../data/ica_data/R_ica_20230618_200633.pkl", 'rb'))
ica2_vecs = np.dot(pca2_vecs, R_ica)

def process_ica(vecs):
    vecs = vecs[:, np.flip(np.argsort(np.abs(scipy.stats.skew(vecs, axis=0))))]
    vecs = vecs * np.sign(scipy.stats.skew(vecs, axis=0))
    return vecs

ica = process_ica(vecs=ica2_vecs)
pca = pca2_vecs
# ----------------------------------------------------------------------------


#%%
# ----------------------------------------------------------------------------
gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 1])
fig = plt.figure(figsize=(16, 4))

main_font = 17
sub_font = 14

axlist = [[0, 1], [2, 3], [49, 50], [99, 100]]
color_ica = 'blue'
color_pca = 'red'
marker_size = 10
alpha = 0.5

for i, ax_indices in enumerate(axlist):
    ax = fig.add_subplot(gs[i])
    
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    
    # Plot ICA
    ax.scatter(ica[:, ax_indices[0]], ica[:, ax_indices[1]], 
               s=marker_size, marker='.', color=color_ica, alpha=alpha)
    
    # Plot PCA
    ax.scatter(pca[:, ax_indices[0]], pca[:, ax_indices[1]], 
               s=marker_size, marker='.', color=color_pca, alpha=alpha)
    
    ax.scatter([], [], s=marker_size, marker='.', color=color_ica, label='ICA', alpha=1)
    ax.scatter([], [], s=marker_size, marker='.', color=color_pca, label='PCA', alpha=1)
    
    ax.tick_params(labelsize=sub_font, direction='in')
    ax.set_xlabel(f'axis {ax_indices[0]}', fontsize=main_font, labelpad=0.5)
    ax.set_ylabel(f'axis {ax_indices[1]}', fontsize=main_font, labelpad=0.5)
    
    if i == 0:
        ax.legend(fontsize=main_font, markerscale=6)

plt.tight_layout(pad=0.5)

if os.path.exists("../output/images") is False:
    os.makedirs("../output/images")
plt.savefig('../output/images/figure5.png', dpi=600)
plt.close()
# ----------------------------------------------------------------------------