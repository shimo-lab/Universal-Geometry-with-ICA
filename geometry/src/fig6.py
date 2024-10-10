#%%
# ----------------------------------------------------------------------------
import scipy
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# ----------------------------------------------------------------------------


#%%
# ----------------------------------------------------------------------------
pca2_vecs = pickle.load(open(f"../data/ica_data/pca2_20230618_200633.pkl", 'rb'))
R_ica = pickle.load(open(f"../data/ica_data/R_ica_20230618_200633.pkl", 'rb'))
ica2_vecs = np.dot(pca2_vecs, R_ica)
wordcount = np.fromfile('../data/text8_sgns/text8_wordcount', dtype=np.int32)
mat_X = np.fromfile('../data/text8_sgns/text8_sgns-Win_ep100').reshape(len(wordcount), -1)
raw_wids = pickle.load(open(f"../data/ica_data/wids_20230618_200633.pkl", 'rb'))
# ----------------------------------------------------------------------------


#%%
# ----------------------------------------------------------------------------
def process(vecs):
    """
    1. axis is sorted in descending order by the abs(skewness)
    2. skewness <- abs(skewness)
    """
    vecs = vecs[:, np.flip(np.argsort(np.abs(scipy.stats.skew(vecs, axis=0))))]
    vecs = vecs * np.sign(scipy.stats.skew(vecs, axis=0))
    return vecs

def scale(matrix):
    return (matrix - np.mean(matrix, axis=0)) / np.std(matrix, axis=0)

def skewness(matrix):
    matrix = scale(matrix)
    return np.abs(np.mean(matrix**3, axis=0))  
def kurtosis(matrix):
    matrix = scale(matrix)
    return np.mean(matrix**4, axis=0) - 3
def logcosh(matrix):
    matrix = scale(matrix)
    return np.mean(np.log(np.cosh(matrix)), axis=0) - 0.374567207491438
def gauss(matrix):
    matrix = scale(matrix)
    return np.mean(-np.exp(-(matrix**2) / 2), axis=0) + 0.707106781186548
# ----------------------------------------------------------------------------


#%%
# ----------------------------------------------------------------------------
ica = process(ica2_vecs)
pca = pca2_vecs
raw = process(scale(mat_X[raw_wids]))
random = np.random.normal(loc=0, scale=1, size=ica.shape)
ica_color = "blue"
pca_color = "red"
raw_color = "green"
# ----------------------------------------------------------------------------


#%%
# ----------------------------------------------------------------------------
gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 1])
fig = plt.figure(figsize=(16, 4))

title_font = 20
main_font = 17
sub_font = 14

ax0 = fig.add_subplot(gs[0])
ax1 = fig.add_subplot(gs[1])
ax2 = fig.add_subplot(gs[2])
ax3 = fig.add_subplot(gs[3])

ax0.set_title("skewness", fontsize=title_font)
ax0.set_xlabel("axis", fontsize=main_font)
ax0.plot(skewness(ica), label="ICA", color=ica_color)
ax0.plot(skewness(pca), label="PCA", color=pca_color)
ax0.plot(skewness(raw), label="raw", color=raw_color)
ax0.plot(skewness(random), label="random", color="#7f7f7f")
legend = ax0.legend(fontsize=main_font)
for line in legend.get_lines():
    line.set_linewidth(3.0)  # ここで線の太さを調整
ax0.tick_params(labelsize=sub_font, direction='in')

ax1.set_title("kurtosis", fontsize=title_font)
ax1.set_xlabel("axis", fontsize=main_font)
ax1.plot(kurtosis(ica), label="ICA", color=ica_color)
ax1.plot(kurtosis(pca), label="PCA", color=pca_color)
ax1.plot(kurtosis(raw), label="raw", color=raw_color)
ax1.plot(kurtosis(random), label="random", color="#7f7f7f")
ax1.tick_params(labelsize=sub_font, direction='in')

ax2.set_title("logcosh", fontsize=title_font)
ax2.set_xlabel("axis", fontsize=main_font)
ax2.plot(logcosh(ica)**2, label="ICA", color=ica_color)
ax2.plot(logcosh(pca)**2, label="PCA", color=pca_color)
ax2.plot(logcosh(raw)**2, label="raw", color=raw_color)
ax2.plot(logcosh(random)**2, label="random", color="#7f7f7f")
ax2.tick_params(labelsize=sub_font, direction='in')

ax3.set_title("Gaussian", fontsize=title_font)
ax3.set_xlabel("axis", fontsize=main_font)
ax3.plot(gauss(ica)**2, label="ICA", color=ica_color)
ax3.plot(gauss(pca)**2, label="PCA", color=pca_color)
ax3.plot(gauss(raw)**2, label="raw", color=raw_color)
ax3.plot(gauss(random)**2, label="random", color="#7f7f7f")
ax3.tick_params(labelsize=sub_font, direction='in')

plt.tight_layout(pad=0.5)
plt.savefig("../output/images/figure6.pdf")
# ----------------------------------------------------------------------------