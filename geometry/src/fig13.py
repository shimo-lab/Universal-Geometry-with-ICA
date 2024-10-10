#%%
# ----------------------------------------------------------------------------
import scipy
import numpy as np
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from adjustText import adjust_text
import matplotlib.gridspec as gridspec
import seaborn as sns; sns.set(style="white")
# ----------------------------------------------------------------------------


#%%
# ----------------------------------------------------------------------------
get_id = pickle.load(open('../data/text8_sgns/text8_word-to-id.pkl', 'rb'))
get_word = pickle.load(open('../data/text8_sgns/text8_id-to-word.pkl', 'rb'))
pca2_vecs = pickle.load(open(f"../data/ica_data/pca2_20230618_200633.pkl", 'rb'))
R_ica = pickle.load(open(f"../data/ica_data/R_ica_20230618_200633.pkl", 'rb'))
ica2_vecs = np.dot(pca2_vecs, R_ica)
wordids = pickle.load(open(f"../data/ica_data/wids_20230618_200633.pkl", 'rb'))
wordcount = np.fromfile('../data/text8_sgns/text8_wordcount', dtype=np.int32)
p = np.array(wordcount)[wordids]
op = np.argsort(p)

def process_ica(vecs):
    """
    1. axis is sorted in descending order by the abs(skewness)
    2. skewness <- abs(skewness)
    """
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
axlist = [
    [0, 1],
    [2, 3],
    [49, 50],
    [99, 100]
]

for i in range(4):
    ax = fig.add_subplot(gs[i])
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.scatter(ica[:, axlist[i][0]][op], ica[:, axlist[i][1]][op], s=2, cmap='rainbow', c=np.log10(p[op]), alpha=1)
    ax.tick_params(labelsize=14, direction='in')
    ax.set_xlabel(f"axis {axlist[i][0]}", fontsize=17, labelpad=0.5)
    ax.set_ylabel(f"axis {axlist[i][1]}", fontsize=17, labelpad=0.5)
plt.tight_layout(pad=0.5)
plt.savefig('../output/images/figure13_ica.png', dpi=600)
# ----------------------------------------------------------------------------


#%%
# ----------------------------------------------------------------------------
gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 1])
fig = plt.figure(figsize=(16, 4))
axlist = [
    [0, 1],
    [2, 3],
    [49, 50],
    [99, 100]
]
## plot
for i in range(4):
    ax = fig.add_subplot(gs[i])
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.scatter(pca[:, axlist[i][0]][op], pca[:, axlist[i][1]][op], s=2, cmap='rainbow', c=np.log10(p[op]), alpha=1)
    ax.tick_params(labelsize=14, direction='in')
    ax.set_xlabel(f"axis {axlist[i][0]}", fontsize=17, labelpad=0.5)
    ax.set_ylabel(f"axis {axlist[i][1]}", fontsize=17, labelpad=0.5)
plt.tight_layout(pad=0.5)
plt.savefig('../output/images/figure13_pca.png', dpi=600)
# ----------------------------------------------------------------------------