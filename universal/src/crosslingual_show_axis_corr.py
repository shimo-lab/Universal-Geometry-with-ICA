import argparse
import pickle as pkl
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr
from tqdm import tqdm

from utils import get_lang_name, get_logger, pos_direct


def parse_args():
    parser = argparse.ArgumentParser(description='show axis correlation')

    # root directory
    parser.add_argument('--root_dir', type=str, default='/working')

    # language
    parser.add_argument('--src', type=str, default='en')
    parser.add_argument('--tgt', type=str, default='es')
    parser.add_argument('--langs', type=str, default='en-es-ru-ar-hi-zh-ja')

    # embedding
    parser.add_argument('--emb_type', type=str, default='cc')
    parser.add_argument('--pca', action='store_true')

    # figure
    parser.add_argument('--dpi', type=int, default=150)

    return parser.parse_args()


def save_corr_matrix_fig(corr_matrix, img_path, src, tgt, dim,
                         pca=False, dpi=None, permute=True):
    corr_matrix = corr_matrix[:dim, :dim]

    labelsize = 50
    titlesize = 50
    fig, ax = plt.subplots(figsize=(15, 12))
    fig.subplots_adjust(left=0.15, right=0.925, bottom=0.16, top=0.95)

    if dim == 300:
        xticlabels = 50
        yticlabels = 50
    elif dim == 100:
        xticlabels = 20
        yticlabels = 20

    ax = sns.heatmap(corr_matrix, xticklabels=xticlabels,
                     yticklabels=yticlabels, cmap='RdBu_r',
                     vmin=-1.0, vmax=1.0, square=True,
                     cbar_kws={'shrink': 1.0,
                               'ticks': np.arange(-1, 1.25, 0.25)})
    ax.xaxis.labelpad = 20

    if pca:
        if permute:
            tgt_title = 'Permuted '
            src_title = 'Permuted '
        else:
            tgt_title = ''
            src_title = ''
        tgt_title += f'PCA Axis of {get_lang_name(tgt)}'
        src_title += f'PCA Axis of {get_lang_name(src)}'
    else:
        if permute:
            tgt_title = 'Permuted '
            src_title = 'Permuted '
        else:
            tgt_title = ''
            src_title = ''
        tgt_title += f'ICA Axis of {get_lang_name(tgt)}'
        src_title += f'ICA Axis of {get_lang_name(src)}'

    ax.set_xlabel(tgt_title, fontsize=titlesize)

    yticks = np.flip(np.arange(0, dim, yticlabels))
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks, rotation=0)

    ax.set_ylabel(src_title, fontsize=titlesize)
    ax.tick_params(labelsize=labelsize, length=15, bottom=True,
                   labelbottom=True, top=False, labeltop=False)
    ax.xaxis.set_label_position('bottom')

    ax.set_ylim(0, dim)
    ax.set_xlim(0, dim)

    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(2)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=labelsize, length=15)

    if Path(img_path).suffix == '.png':
        assert dpi is not None
        fig.savefig(img_path, dpi=dpi)
    else:
        fig.savefig(img_path)

    plt.close(fig)


def save_diag_plot(both_swap_corr_matrix, img_path, src, tgt,
                   pca=False, dpi=None):
    dim = len(both_swap_corr_matrix)
    labelsize = 30
    titlesize = 40
    _ = plt.figure(figsize=(15, 15))
    assert sorted(np.diag(both_swap_corr_matrix), reverse=True) == \
        list(np.diag(both_swap_corr_matrix))
    plt.scatter(range(dim), np.diag(both_swap_corr_matrix), marker='.')
    if pca:
        pp = 'PCA'
    else:
        pp = 'ICA'
    plt.xlabel('Descending order of sorted axis pairs', fontsize=titlesize)
    plt.ylabel(f"{pp} {get_lang_name(src)}-{get_lang_name(tgt)} Pearson's r",
               fontsize=titlesize)
    plt.tick_params(labelsize=labelsize, length=15)

    if Path(img_path).suffix == '.png':
        assert dpi is not None
        plt.savefig(img_path, dpi=dpi)
    else:
        plt.savefig(img_path)


def main():
    args = parse_args()
    root_dir = Path(args.root_dir)

    src = args.src
    tgt = args.tgt
    langs = args.langs

    emb_type = args.emb_type
    assert emb_type in ['cc', 'muse']
    pca = args.pca
    pp = 'pca' if pca else 'ica'

    dpi = args.dpi

    output_dir = root_dir / f'output/crosslingual/figures/axis_corr/{langs}'
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger()

    logger.info(f'=============== src: {src}, tgt: {tgt} ===============')

    src2tgt_dict_path = root_dir /\
        f'data/crosslingual/MUSE/dictionaries/{src}-{tgt}.0-5000.txt'
    logger.info(f'src2tgt_dict_path: {src2tgt_dict_path}')
    with open(src2tgt_dict_path, 'r') as f:
        pairs = [line.strip().split() for line in f.readlines()]

    if emb_type == 'cc':
        src_dump_path = root_dir /\
            f'output/crosslingual/{langs}/cc.{src}.300_dic_and_emb.pkl'
        tgt_dump_path = root_dir /\
            f'output/crosslingual/{langs}/cc.{tgt}.300_dic_and_emb.pkl'
    else:
        src_dump_path = root_dir /\
            f'output/crosslingual/{langs}/wiki.multi.{src}_dic_and_emb.pkl'
        tgt_dump_path = root_dir / f'output/crosslingual/{langs}/'\
            f'wiki.multi.{tgt}_dic_and_emb.pkl'

    logger.info(f'src_dump_path: {src_dump_path}')
    with open(src_dump_path, 'rb') as f:
        src_word2id, src_id2word, _, src_pca_embed, src_ica_embed = pkl.load(f)

    logger.info(f'tgt_dump_path: {tgt_dump_path}')
    with open(tgt_dump_path, 'rb') as f:
        tgt_word2id, tgt_id2word, _, tgt_pca_embed, tgt_ica_embed = \
            pkl.load(f)

    if pca:
        src_embed = src_pca_embed
        tgt_embed = tgt_pca_embed
    else:
        src_embed = src_ica_embed
        tgt_embed = tgt_ica_embed

    src_embed = pos_direct(src_embed)
    tgt_embed = pos_direct(tgt_embed)
    _, dim = src_embed.shape

    src_vs = []
    tgt_vs = []
    for s, t in pairs:
        if s in src_word2id and t in tgt_word2id:
            si = src_word2id[s]
            ti = tgt_word2id[t]
            sv = src_embed[si]
            tv = tgt_embed[ti]

            src_vs.append(sv)
            tgt_vs.append(tv)

    src_vs = np.array(src_vs)
    tgt_vs = np.array(tgt_vs)

    logger.info(
        f'use pairs: {len(src_vs)}/{len(pairs)} - '
        f'{100*len(src_vs)/len(pairs):.2f}%')

    # correlation matrix
    cands = []
    corr_matrix = [[] for _ in range(dim)]
    logger.info('start calculating corr_matrix...')
    for i in tqdm(range(dim)):
        ax1 = src_vs[:, i]
        for j in range(dim):
            ax2 = tgt_vs[:, j]
            r = pearsonr(ax1, ax2)[0]
            corr_matrix[i].append(r)
            cands.append((r, i, j))
    logger.info('finish calculating corr_matrix!')
    corr_matrix = np.array(corr_matrix)
    img_path = output_dir /\
        f'{pp}-axis_corr-{src}-{tgt}-{emb_type}-100d-{dpi}dpi.png'
    save_corr_matrix_fig(corr_matrix, img_path, src,
                         tgt, dim=100, pca=pca, dpi=dpi, permute=False)

    # greedy matching
    cands.sort(reverse=True)
    used_i = set()
    used_j = set()
    rij = []
    for r, i, j in cands:
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        rij.append((r, i, j))
    swap_corr_matrix = np.zeros((dim, dim))
    for _, i, j in rij:
        swap_corr_matrix[j, :] = corr_matrix[i, :]
    img_path = output_dir /\
        f'{pp}-axis_corr-greedy-{src}-{tgt}-{emb_type}-100d-{dpi}dpi.png'
    save_corr_matrix_fig(swap_corr_matrix, img_path, src,
                         tgt, dim=100, pca=pca, dpi=dpi)

    # sort by diagonal for greedy matching
    both_swap_corr_matrix = swap_corr_matrix.copy()
    both_swap_corr_matrix = \
        both_swap_corr_matrix[np.argsort(-np.diag(swap_corr_matrix)), :]
    for i in tqdm(range(dim)):
        argmax = i + np.argmax(both_swap_corr_matrix[i][i:])
        tmp = both_swap_corr_matrix[:, i].copy()
        both_swap_corr_matrix[:, i] = both_swap_corr_matrix[:, argmax]
        both_swap_corr_matrix[:, argmax] = tmp

    img_path = output_dir / f'{pp}-axis_corr-greedy_and_sort-'\
        f'{src}-{tgt}-{emb_type}-100d-{dpi}dpi.png'
    save_corr_matrix_fig(both_swap_corr_matrix, img_path, src, tgt,
                         dim=100, pca=pca, dpi=dpi)
    img_path = output_dir / f'{pp}-axis_corr_diag_plot-greedy_and_sort-'\
        f'{src}-{tgt}-{emb_type}-100d-{dpi}dpi.png'
    save_diag_plot(both_swap_corr_matrix, img_path, src, tgt,
                   pca=pca, dpi=dpi)


if __name__ == '__main__':
    main()
