#%%
# ----------------------------------------------------------------------------
import os
import scipy
import numpy as np
import pickle
import matplotlib.pyplot as plt
from adjustText import adjust_text
import matplotlib.gridspec as gridspec
import seaborn as sns; sns.set(style="white")
# ----------------------------------------------------------------------------


#%%
# ----------------------------------------------------------------------------
class Operation:
    def __init__(self, vecs, wids, get_word, get_id):
        self.get_word = get_word
        self.get_id = get_id

        def process_vecs(vecs):
            """
            1. axis is sorted in descending order by the abs(skewness)
            2. skewness <- abs(skewness)
            """
            vecs = vecs[:, np.flip(np.argsort(np.abs(scipy.stats.skew(vecs, axis=0))))]
            vecs = vecs * np.sign(scipy.stats.skew(vecs, axis=0))
            return vecs
        
        vecs = process_vecs(vecs=vecs)
        self.vecs = vecs / np.linalg.norm(vecs, axis=1).reshape(-1, 1)
        self.wids = list(wids)
        self.wids_nodup, idx = np.unique(wids, return_index=True)
        self.wordlist = np.array([self.get_word[wid] for wid in self.wids_nodup])
        self.vecs_nodup = self.vecs[idx]

    def vector(self, word):
        vec = np.array(self.vecs[self.wids.index(self.get_id[word])]).copy()
        return vec
    
    def get_topwords_cossim(self, vec_query, n_top=10):
        list_cossim = np.dot(self.vecs_nodup, vec_query) / (np.linalg.norm(self.vecs_nodup, axis=1) * np.linalg.norm(vec_query))
        list_args_cossim = np.argsort(list_cossim)
        wids_args_cossim = self.wids_nodup[list_args_cossim]
        topwords = [self.get_word[wid] for wid in np.flip(wids_args_cossim)[:n_top]]
        return topwords
    
    def get_topwords(self, axlist, n_top):
        query = np.zeros(self.vecs.shape[1], dtype=np.float32)
        query[axlist] = 1
        topwords_cossim = self.get_topwords_cossim(vec_query=query, n_top=n_top)
        print(f"{axlist} cossim: {topwords_cossim}")

    def get_sortedwordlist(self, ax, norm1):
        vectors = self.vecs_nodup.copy()
        if norm1:
            vectors = vectors / np.linalg.norm(vectors, axis=1).reshape(-1, 1)
        list_args_ax = np.argsort(vectors[:, ax])
        wids_args_ax = self.wids_nodup[list_args_ax]
        wordlist = [self.get_word[wid] for wid in np.flip(wids_args_ax)]
        return wordlist
    
    def get_matrix(self, axlist, norm1=True):
        """
        input: axlist
        output: (n_words, len(axlist))
        """
        ans = self.vecs_nodup.copy()
        if norm1:
            ans = ans / np.linalg.norm(ans).reshape(-1, 1)
        ans = ans[:, axlist]
        return ans

def norm1(matrix):
    return matrix / np.linalg.norm(matrix, axis=1).reshape(-1, 1)

def sort_vectors_by_nth(vecs, n):
    args_vec = np.argsort(vecs, axis=1)
    nth_largest_idx = args_vec[:, -n]
    nth_largest_val = vecs[np.arange(vecs.shape[0]), nth_largest_idx]
    sorted_idx = np.flip(np.argsort((nth_largest_val)))
    sorted_vecs = vecs[sorted_idx]
    return sorted_vecs, sorted_idx
# ----------------------------------------------------------------------------


#%%
# ----------------------------------------------------------------------------
get_id = pickle.load(open('../data/text8_sgns/text8_word-to-id.pkl', 'rb'))
get_word = pickle.load(open('../data/text8_sgns/text8_id-to-word.pkl', 'rb'))
pca2_vecs = pickle.load(open(f"../data/ica_data/pca2_20230618_200633.pkl", 'rb'))
R_ica = pickle.load(open(f"../data/ica_data/R_ica_20230618_200633.pkl", 'rb'))
wids = pickle.load(open(f"../data/ica_data/wids_20230618_200633.pkl", 'rb'))
ica_vecs = np.dot(pca2_vecs, R_ica)
op = Operation(vecs=ica_vecs, wids=wids, get_id=get_id, get_word=get_word)
# ----------------------------------------------------------------------------


#%%
# ----------------------------------------------------------------------------
# Preparing data for visualization
# ----------------------------------------------------------------------------
country = [34, 56]
category = [16, 26, 35]
axlist = category + country
fig1_matrix = op.get_matrix(axlist=axlist, norm1=True)

wordlist_1 = []
for ax in axlist:
    wordlist_1 += op.get_sortedwordlist(ax=ax, norm1=True)[:2]
wordlist_2 = []
for i, axc1 in enumerate(country):
    for j, axc2 in enumerate(category):
        query = np.zeros(op.vecs.shape[1], dtype=np.float32)
        query[[axc1, axc2]] = 1
        wordlist_2_kouho = op.get_topwords_cossim(vec_query=query, n_top=20)
        idx_kouho = [np.where(op.wordlist==word)[0][0] for word in wordlist_2_kouho]
        _, sorted_idx_kouho = sort_vectors_by_nth(norm1(fig1_matrix[idx_kouho]), n=2)
        wordlist_2 += list(np.array(wordlist_2_kouho)[sorted_idx_kouho][:5])
wordlist = wordlist_1 + wordlist_2
wordlist = list(set(wordlist))
vecs = np.array([op.vector(word=word) for word in wordlist])
vecs = vecs / np.linalg.norm(vecs, axis=1).reshape(-1, 1)
vecs = vecs[:, axlist]
word_topax_topvalue = [(wordlist[i], np.argmax(vecs[i]), np.max(vecs[i])) for i in range(len(wordlist))]
sorted_wtt = sorted(word_topax_topvalue, key=lambda x:(x[1], -x[2]))
wordlist_new = [sorted_wtt[i][0] for i in range(len(sorted_wtt))]
vecs = np.array([op.vector(word=word) for word in wordlist_new])
vecs = vecs / np.linalg.norm(vecs, axis=1).reshape(-1, 1)
vecs = vecs[:, axlist]

wids = [get_id[w] for w in wordlist_new]
fig1_op = Operation(vecs=vecs, wids=wids, get_id=get_id, get_word=get_word)
fig1ax = {"film":35, "japanese":56, "italian":34, "cars":26, "dishes":16}
ax300 = {"japanese":56, "italian":34, "dishes":16, "cars":26, "film":35}
# ----------------------------------------------------------------------------


#%%
# ----------------------------------------------------------------------------
d_wordlist = {
    "dishes" : [w for w in wordlist_new[0:12]],
    "cars" : [w for w in wordlist_new[12:22]],
    "film" : [w for w in wordlist_new[22:30]],
    "italian" : [w for w in wordlist_new[30:34]],
    "japanese" : [w for w in wordlist_new[34:]]
}
d_backcolor = {
    "dishes" : "salmon",
    "cars" : "#FFCC99",
    "film" : "#FFFE91",
    "italian" : "#98FB98",
    "japanese" : "#B0E0E6"
}
d_color = {
    word: d_backcolor[ax] for ax, wordlist in d_wordlist.items() for word in wordlist
}
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
gs = gridspec.GridSpec(2, 2, width_ratios=[0.9, 1], height_ratios=[1, 1])
fig = plt.figure(figsize=(15, 15))

xlabelsize=27
xticklabelsize=27
ylabelsize=27

# ............................................................................
ax1 = fig.add_subplot(gs[:, 0])
g = sns.heatmap(data=vecs, yticklabels=False, cmap="magma_r", vmin=-0.1, vmax=1.0, cbar_kws={"shrink": 1, "aspect": 40, "pad":0.02})
g.tick_params(left=False, bottom=True)
for i, label in enumerate(["16\n[dishes]      ", "26\n[cars]  ", "35\n   [film]     ", "34\n   [italian]  ", "56\n            [japanese]"]): # 24pt
    idx, label_ax = label.split("\n")
    ax1.text(i+0.5, ax1.get_ylim()[0]+0.4, f"{idx}",
             fontsize=27,  # main_textのフォントサイズ
             ha='center',
             va='top')
    ax1.text(i+0.5, ax1.get_ylim()[0]+0.6, f"\n{label_ax}",
             fontsize=24,  # sub_textのフォントサイズ
             ha='center',
             va='top')
ax1.set_xticks([])
ax1.set_yticklabels(g.get_yticklabels(), rotation=0, fontsize=ylabelsize)


## ============= color box 1 =================================================
for i, word in enumerate(wordlist_new):
    if word in d_wordlist["dishes"]:
        ax1.annotate(text=" "*19, xy=(-0.1, i+0.5), annotation_clip=False, fontsize=27, ha='right', va='center', 
                     bbox=dict(boxstyle="square,pad=0.01", fc=d_backcolor["dishes"], ec="none"))
    elif word in d_wordlist["cars"]:
        ax1.annotate(text=" "*19, xy=(-0.1, i+0.5), annotation_clip=False, fontsize=27, ha='right', va='center',
                     bbox=dict(boxstyle="square,pad=0.01", fc=d_backcolor["cars"], ec="none"))
    elif word in d_wordlist["film"]:
        ax1.annotate(text=" "*19, xy=(-0.1, i+0.5), annotation_clip=False, fontsize=27, ha='right', va='center',
                     bbox=dict(boxstyle="square,pad=0.01", fc=d_backcolor["film"], ec="none"))
    elif word in d_wordlist["italian"]:
        ax1.annotate(text=" "*19, xy=(-0.1, i+0.5), annotation_clip=False, fontsize=27, ha='right', va='center',
                     bbox=dict(boxstyle="square,pad=0.01", fc=d_backcolor["italian"],  ec="none"))
    elif word in d_wordlist["japanese"]:
        ax1.annotate(text=" "*19, xy=(-0.1, i+0.5), annotation_clip=False, fontsize=27, ha='right', va='center',
                     bbox=dict(boxstyle="square,pad=0.01", fc=d_backcolor["japanese"], ec="none"))
for i, word in enumerate(wordlist_new):
    if word in d_wordlist["dishes"]:
        ax1.annotate(text=word, xy=(-0.1, i+0.5), annotation_clip=False, fontsize=27, ha='right', va='center', 
                     bbox=dict(boxstyle="square,pad=0.01", fc="None", ec="none"))
    elif word in d_wordlist["cars"]:
        ax1.annotate(text=word, xy=(-0.1, i+0.5), annotation_clip=False, fontsize=27, ha='right', va='center',
                     bbox=dict(boxstyle="square,pad=0.01", fc="None", ec="none"))
    elif word in d_wordlist["film"]:
        ax1.annotate(text=word, xy=(-0.1, i+0.5), annotation_clip=False, fontsize=27, ha='right', va='center',
                     bbox=dict(boxstyle="square,pad=0.01", fc="None", ec="none"))
    elif word in d_wordlist["italian"]:
        ax1.annotate(text=word, xy=(-0.1, i+0.5), annotation_clip=False, fontsize=27, ha='right', va='center',
                     bbox=dict(boxstyle="square,pad=0.01", fc="None",  ec="none"))
    elif word in d_wordlist["japanese"]:
        ax1.annotate(text=word, xy=(-0.1, i+0.5), annotation_clip=False, fontsize=27, ha='right', va='center',
                     bbox=dict(boxstyle="square,pad=0.01", fc="None", ec="none"))
## ===========================================================================


cbar = g.collections[0].colorbar
cbar.ax.tick_params(labelsize=24)
# ............................................................................

mojisize=27
xylabel=27

# ............................................................................
wx = "italian"
wy = "cars"
wz = "dishes"
minx, maxx = -0.25, 0.7
miny, maxy = -0.25, 0.7
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(op.vecs_nodup[:, ax300[wx]], op.vecs_nodup[:, ax300[wy]], marker='.', s=120, color="lightskyblue")
ax2.plot([minx, maxx], [0, 0], color="darkgray", lw=1.5)
ax2.plot([0, 0], [miny, maxy], color="darkgray", lw=1.5)
for wid in wids:
    ax2.scatter(op.vector(word=get_word[wid])[ax300[wx]], op.vector(word=get_word[wid])[ax300[wy]], marker='.', s=180, color="black")
ax2.set_xlim(minx, maxx)
ax2.set_ylim(miny, maxy)
ax2.set_xlabel(f"{fig1ax[wx]}: [{wx}]", fontsize=xylabel, labelpad=5)
ax2.set_ylabel(f"{fig1ax[wy]}: [{wy}]", fontsize=xylabel, labelpad=5)
ax2.tick_params(axis='both', which='major', labelbottom=False, labelleft=False)
list_x = ["italian", "di", "venice", "modena", "fellini", "pizza", "scorsese"]
list_y = ["cars", "car", "ferrari", "mazda", "toyota", "honda", "suzuki", "lamborghini", "nissan", "ducati"]
words_x = [ax2.text(op.vector(word=word)[ax300[wx]], op.vector(word=word)[ax300[wy]], word, fontsize=mojisize, bbox=dict(boxstyle="square,pad=0.01", fc=d_color[word], ec="none")) for word in list(set(list_x))]
words_y = [ax2.text(op.vector(word=word)[ax300[wx]], op.vector(word=word)[ax300[wy]], word, fontsize=mojisize, bbox=dict(boxstyle="square,pad=0.01", fc=d_color[word], ec="none")) for word in list(set(list_y))]
words = words_x + words_y
adjust_text(words)
# ............................................................................

# ............................................................................
wx = "japanese"
wy = "film"
wz = "dishes"
minx, maxx = -0.25, 0.8
miny, maxy = -0.25, 0.9
ax3 = fig.add_subplot(gs[1, 1])
ax3.scatter(op.vecs_nodup[:, ax300[wx]], op.vecs_nodup[:, ax300[wy]], marker='.', s=120, color="lightskyblue")
ax3.plot([minx, maxx], [0, 0], color="darkgray", lw=1.5)
ax3.plot([0, 0], [miny, maxy], color="darkgray", lw=1.5)
for wid in wids:
    ax3.scatter(op.vector(word=get_word[wid])[ax300[wx]], op.vector(word=get_word[wid])[ax300[wy]], marker='.', s=180, color="black")
ax3.set_xlim(minx, maxx)
ax3.set_ylim(miny, maxy)
ax3.set_xlabel(f"{fig1ax[wx]}: [{wx}]", fontsize=xylabel, labelpad=5)
ax3.set_ylabel(f"{fig1ax[wy]}: [{wy}]", fontsize=xylabel, labelpad=5)
ax3.tick_params(axis='both', which='both', labelbottom=False, labelleft=False)
list_x = ["japanese", "japan", "miyazaki", "anime", "akira", "suzuki", "sushi"]
list_y = ["film", "films", "kurosawa", "scorsese", "fellini", "comedies"]
words_x = [ax3.text(op.vector(word=word)[ax300[wx]], op.vector(word=word)[ax300[wy]], word, fontsize=mojisize, bbox=dict(boxstyle="square,pad=0.01", fc=d_color[word], ec="none")) for word in list(set(list_x))]
words_y = [ax3.text(op.vector(word=word)[ax300[wx]], op.vector(word=word)[ax300[wy]], word, fontsize=mojisize, bbox=dict(boxstyle="square,pad=0.01", fc=d_color[word], ec="none")) for word in list(set(list_y))]
words = words_x + words_y
adjust_text(words)
ax3.scatter([], [], marker='.', s=100, color="black", label="Heatmap words")
ax3.scatter([], [], marker='.', s=100, color="lightskyblue", label="Other words")
ax3.legend(fontsize=24, markerscale=2, bbox_to_anchor=(1.021, 1.021), borderpad=0.2, handletextpad=0.08)
# ............................................................................
plt.subplots_adjust(left=0.158, right=0.996, bottom=0.068, top=0.990, wspace=0.12, hspace=0.018)
# ............................................................................
pos = ax3.get_position()
pad = 5.5/150
new_pos = [pos.x0, pos.y0 - pad, pos.width, pos.height]
ax3.set_position(new_pos)
# ............................................................................
if os.path.exists("../output/images") is False:
    os.makedirs("../output/images")
plt.savefig(f"../output/images/figure1.png", dpi=150)
# ----------------------------------------------------------------------------