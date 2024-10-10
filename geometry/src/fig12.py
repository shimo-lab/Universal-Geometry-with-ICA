#%%
# ----------------------------------------------------------------------------
import scipy
import numpy as np
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from adjustText import adjust_text
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
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
        # self.vecs = vecs
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
ica_vecs = np.dot(pca2_vecs, R_ica)
wids = pickle.load(open(f"../data/ica_data/wids_20230618_200633.pkl", 'rb'))
op = Operation(vecs=ica_vecs, wids=wids, get_id=get_id, get_word=get_word)
# ----------------------------------------------------------------------------


#%%
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
d_font = {
    "mojisize" : 32,
    "xylabel" : 32
}
d_range = {
    "dishes" : [-0.25, 0.7],
    "cars" : [-0.25, 0.7],
    "film" : [-0.25, 0.9],
    "italian" : [-0.25, 0.7],
    "japanese" : [-0.25, 0.8]
}
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
def scatter(config):
    global op, wids, get_word, ax300, d_font, d_color, d_range

    minx = d_range[config["axis_x"]][0]
    maxx = d_range[config["axis_x"]][1]
    miny = d_range[config["axis_y"]][0]
    maxy = d_range[config["axis_y"]][1]
    axis_x = config["axis_x"]
    axis_y = config["axis_y"]
    wordlist_x = config["words_x"]
    wordlist_y = config["words_y"]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.scatter(op.vecs_nodup[:, ax300[axis_x]], op.vecs_nodup[:, ax300[axis_y]], 
               marker='.', s=120, color="lightskyblue")
    ax.plot([minx, maxx], [0, 0], color="darkgray", lw=1.5)
    ax.plot([0, 0], [miny, maxy], color="darkgray", lw=1.5)
    for wid in wids:
        ax.scatter(op.vector(word=get_word[wid])[ax300[axis_x]], op.vector(word=get_word[wid])[ax300[axis_y]],
                    marker='.', s=180, color="black")
    ax.set_xlabel(f"{ax300[axis_x]}: [{axis_x}]", fontsize=d_font["xylabel"], labelpad=5)
    ax.set_ylabel(f"{ax300[axis_y]}: [{axis_y}]", fontsize=d_font["xylabel"], labelpad=5)
    ax.tick_params(axis='both', which='major', labelbottom=False, labelleft=False)
    
    texts_x = [ax.text(op.vector(word=word)[ax300[axis_x]], op.vector(word=word)[ax300[axis_y]], word, fontsize=d_font["mojisize"], bbox=dict(boxstyle="square,pad=0.01", fc=d_color[word], ec="none")) for word in wordlist_x]
    texts_y = [ax.text(op.vector(word=word)[ax300[axis_x]], op.vector(word=word)[ax300[axis_y]], word, fontsize=d_font["mojisize"], bbox=dict(boxstyle="square,pad=0.01", fc=d_color[word], ec="none")) for word in wordlist_y]
    texts = texts_x + texts_y
    adjust_text(texts)
    plt.tight_layout()
    if config["savefig"]:
        savename = f"figure12_{config['axis_x']}_{config['axis_y']}.png"
        plt.savefig(f"../output/images/{savename}", dpi=300)

def combined_scatter(configs):
    fig = plt.figure(figsize=(50, 20))
    gs = fig.add_gridspec(2, 5, hspace=0.1, wspace=0.1)

    for i, config in enumerate(configs):
        ax = fig.add_subplot(gs[i // 5, i % 5])
        
        axis_x, axis_y = config['axis_x'], config['axis_y']
        minx, maxx = d_range[axis_x]
        miny, maxy = d_range[axis_y]
        
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.scatter(op.vecs_nodup[:, ax300[axis_x]], op.vecs_nodup[:, ax300[axis_y]], 
                   marker='.', s=20, color="lightskyblue")
        ax.plot([minx, maxx], [0, 0], color="darkgray", lw=1)
        ax.plot([0, 0], [miny, maxy], color="darkgray", lw=1)
        
        for wid in wids:
            ax.scatter(op.vector(word=get_word[wid])[ax300[axis_x]], 
                       op.vector(word=get_word[wid])[ax300[axis_y]],
                       marker='.', s=30, color="black")
        
        ax.set_xlabel(f"{ax300[axis_x]}: [{axis_x}]", fontsize=22, labelpad=5)
        ax.set_ylabel(f"{ax300[axis_y]}: [{axis_y}]", fontsize=22, labelpad=5)
        ax.tick_params(axis='both', which='major', labelbottom=False, labelleft=False)
        
        texts = []
        for word_list in [config['words_x'], config['words_y']]:
            for word in word_list:
                x = op.vector(word=word)[ax300[axis_x]]
                y = op.vector(word=word)[ax300[axis_y]]
                texts.append(ax.text(x, y, word, fontsize=20, 
                             bbox=dict(boxstyle="square,pad=0.01", fc=d_color[word], ec="none")))
        
        adjust_text(texts, ax=ax)
        ax.set_title(f"{axis_x.capitalize()} vs {axis_y.capitalize()}", fontsize=24)

    plt.savefig("../output/images/figure12_combined_scatters.png", dpi=300, bbox_inches='tight')
    plt.close()
# ----------------------------------------------------------------------------


#%%
# ----------------------------------------------------------------------------
config1 = {
    "axis_x" : "japanese",
    "axis_y" : "dishes",
    "words_x" : ["japanese", "japan", "miyazaki", "anime", "akira", "kurosawa"],
    "words_y" : ["dishes", "meat", "rice", "sushi", "pasta", "pizza", "tomato"],
    "savefig" : True
}

config2 = {
    "axis_x" : "japanese",
    "axis_y" : "film",
    "words_x" : ["japanese", "japan", "miyazaki", "anime", "akira", "suzuki", "sushi"],
    "words_y" : ["film", "films", "kurosawa", "scorsese", "fellini", "comedies"],
    "savefig" : True
}


config3 = {
    "axis_x" : "japanese",
    "axis_y" : "cars",
    "words_x" : ["japanese", "japan", "miyazaki", "anime", "akira"],
    "words_y" : ["cars", "car", "ferrari", "mazda", "toyota", "honda", "suzuki", "lamborghini", "nissan", "ducati"],
    "savefig" : True
}

config4 = {
    "axis_x" : "italian",
    "axis_y" : "dishes",
    "words_x" : ["italian", "di", "venice", "modena", "ferrari", "fellini", "lamborghini"],
    "words_y" : ["dishes", "meat", "rice", "sushi", "pasta", "pizza", "tomato", "potatoes"],
    "savefig" : True
}

config5 = {
    "axis_x" : "italian",
    "axis_y" : "cars",
    "words_x" : ["italian", "di", "venice", "modena", "fellini", "pizza", "scorsese"],
    "words_y" : ["cars", "car", "ferrari", "mazda", "toyota", "honda", "suzuki", "lamborghini", "nissan", "ducati"],
    "savefig" : True
}

config6 = {
    "axis_x" : "italian",
    "axis_y" : "film",
    "words_x" : ["italian", "di", "venice", "modena", "ferrari", "lamborghini"],
    "words_y" : ["film", "films", "kurosawa", "scorsese", "fellini", "comedies", "akira", "miyazaki"],
    "savefig" : True
}

config7 = {
    "axis_x" : "dishes",
    "axis_y" : "film",
    "words_x" : ["dishes", "meat", "rice", "sushi", "pasta", "pizza", "noodles"],
    "words_y" : ["film", "films", "kurosawa", "scorsese", "fellini", "akira", "miyazaki"],
    "savefig" : True
}

config8 = {
    "axis_x" : "dishes",
    "axis_y" : "cars",
    "words_x" : ["dishes", "meat", "rice", "sushi", "pasta", "pizza", "tomato"],
    "words_y" : ["cars", "car", "ferrari", "mazda", "toyota", "honda", "suzuki", "lamborghini"],
    "savefig" : True
}

config9 = {
    "axis_x" : "film",
    "axis_y" : "cars",
    "words_x" : ["film", "films", "kurosawa", "scorsese", "camera", "animation", "fellini"],
    "words_y" : ["cars", "car", "ferrari", "mazda", "toyota", "honda", "suzuki", "lamborghini", "nissan"],
    "savefig" : True
}

config10 = {
    "axis_x" : "japanese",
    "axis_y" : "italian",
    "words_x" : ["japanese", "japan", "kurosawa", "sushi", "nissan", "honda", "anime"],
    "words_y" : ["italian", "di", "venice", "modena", "ferrari", "pizza", "scorsese", "fellini"],
    "savefig" : True
}

configs = [config1, config2, config3, config4, config5, config6, config7, config8, config9, config10]
combined_scatter(configs)
# ----------------------------------------------------------------------------
# for config in configs:
    # scatter(config)
# ----------------------------------------------------------------------------