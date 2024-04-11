import argparse
import pickle as pkl
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from utils import get_logger, pos_direct
from WeightedCorr import WeightedCorr


def parse_args():
    parser = argparse.ArgumentParser(description='show axis correlation')

    # root directory
    parser.add_argument('--root_dir', type=str, default='/working')

    # language
    parser.add_argument('--langs', type=str, default='en-es-ru-ar-hi-zh-ja')

    # embedding
    parser.add_argument('--emb_type', type=str, default='cc')
    parser.add_argument('--pca', action='store_true')

    # bert vocabularies
    parser.add_argument('--num_token', type=int, default=100000)

    # figure
    parser.add_argument('--dpi', type=int, default=150)

    return parser.parse_args()


def save_corr_matrix_fig(corr_matrix, output_path, src, tgt, dim=300,
                         pp='ica', dpi=None):

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
                     yticklabels=yticlabels,
                     cmap='RdBu_r', vmin=-1.0, vmax=1.0, square=True,
                     cbar_kws={
                         "shrink": 1.0, "ticks": np.arange(-1, 1.25, 0.25)})
    ax.xaxis.labelpad = 20
    upper_pp = pp.upper()
    bert_title = f'Permuted {upper_pp} Axis of {tgt}'
    en_title = f'Permuted {upper_pp} Axis of {src}'
    ax.set_xlabel(bert_title, fontsize=titlesize)

    yticks = np.flip(np.arange(0, dim, yticlabels))
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks, rotation=0)

    ax.set_ylabel(en_title, fontsize=titlesize)
    ax.tick_params(labelsize=labelsize, length=15, bottom=True,
                   labelbottom=True, top=False, labeltop=False)
    ax.xaxis.set_label_position('bottom')

    # Set the limits for x and y axes
    ax.set_ylim(0, dim)
    ax.set_xlim(0, dim)

    # Drawing the frame
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(2)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=labelsize, length=15)

    if Path(output_path).suffix == '.png':
        assert dpi is not None
        fig.savefig(output_path, dpi=dpi)
    else:
        fig.savefig(output_path)

    plt.close(fig)


def main():

    args = parse_args()
    root_dir = Path(args.root_dir)

    langs = args.langs

    emb_type = args.emb_type
    pca = args.pca
    pp = 'pca' if pca else 'ica'

    num_token = args.num_token

    dpi = args.dpi

    logger = get_logger()

    en_bert_path = root_dir /\
        f'output/dynamic/{pp}-en_{langs}_{emb_type}-bert_{num_token}.pkl'
    if en_bert_path.exists():
        logger.info(f'load {en_bert_path}')
        with open(en_bert_path, 'rb') as f:
            en_bert = pkl.load(f)
        sorted_en_embed, en_id2word, en_word2id, \
            sorted_bert_embed, bert_id2token, bert_token2id, \
            bert_sents, bert_org_token2ids, both_swap_corr_matrix = en_bert
    else:
        # fastText
        if emb_type == 'cc':
            en_dump_path = root_dir /\
                f'output/crosslingual/{langs}/cc.en.300_dic_and_emb.pkl'
        else:
            en_dump_path = root_dir /\
                f'output/crosslingual/{langs}/wiki.multi.en_dic_and_emb.pkl'

        if not en_dump_path.exists():
            raise FileNotFoundError(f'{en_dump_path} does not exist! Run '
                                    'crosslingual_save_embeddings.py first.')

        logger.info(f'en_dump_path: {en_dump_path}')
        with open(en_dump_path, 'rb') as f:
            en_word2id, en_id2word, _, en_pca_embed, en_ica_embed = pkl.load(f)

        # BERT
        bert_dump_path = root_dir /\
            f'output/dynamic/bert-pca-ica-{num_token}.pkl'
        logger.info(f'bert_dump_path: {bert_dump_path}')
        with open(bert_dump_path, 'rb') as f:
            tokens_sents_embeds = pkl.load(f)
        bert_id2token, bert_sents, _, bert_pca_embed, bert_ica_embed = \
            tokens_sents_embeds

        if pca:
            en_embed = en_pca_embed
            bert_embed = bert_pca_embed
        else:
            en_embed = en_ica_embed
            bert_embed = bert_ica_embed

        en_embed = pos_direct(en_embed)
        bert_embed = pos_direct(bert_embed)

        en_dim = en_embed.shape[1]
        bert_dim = bert_embed.shape[1]
        logger.info(f'en_embed.shape: {en_embed.shape}')
        logger.info(f'bert_embed.shape: {bert_embed.shape}')

        bert_org_token2ids = defaultdict(list)
        bert_token2id = {}
        for i, token_c in enumerate(bert_id2token):
            # remove number
            token = '_'.join(token_c.split('_')[:-1])
            bert_org_token2ids[token].append(i)
            bert_token2id[token_c] = i

        pairs = []
        weights = []
        for i in range(len(en_embed)):
            word = en_id2word[i]
            if word in bert_org_token2ids:
                assert len(bert_org_token2ids[word]) > 0
                w = 1 / len(bert_org_token2ids[word])
                for j in bert_org_token2ids[word]:
                    pairs.append((i, j))
                    weights.append(w)

        assert len(pairs) == len(weights)
        logger.info(f'the number of pairs: {len(pairs)}')

        en_vs = []
        bert_vs = []

        for i, j in pairs:
            en_vs.append(en_embed[i])
            bert_vs.append(bert_embed[j])

        en_vs = np.array(en_vs)
        bert_vs = np.array(bert_vs)

        cands = []
        corr_matrix = [[] for _ in range(en_dim)]
        logger.info('start calculating corr_matrix...')
        weights = pd.Series(weights)
        for i in tqdm(range(en_dim)):
            ax1 = en_vs[:, i]
            ax1 = pd.Series(ax1)
            for j in range(bert_dim):
                ax2 = bert_vs[:, j]
                ax2 = pd.Series(ax2)
                r = WeightedCorr(x=ax1, y=ax2, w=weights)(method='pearson')
                corr_matrix[i].append(r)
                cands.append((r, i, j))
        logger.info('finish calculating corr_matrix!')
        corr_matrix = np.array(corr_matrix)
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
        assert len(rij) == en_dim

        # index mapping
        j2k = {}
        for k, (r, i, j) in enumerate(sorted(rij, key=lambda x: x[2])):
            j2k[j] = k

        tmp_corr_matrix = np.zeros((en_dim, en_dim))
        for j in range(bert_dim):
            if j in j2k:
                k = j2k[j]
                tmp_corr_matrix[:, k] = corr_matrix[:, j]
        corr_matrix = tmp_corr_matrix

        sorted_en_embed = np.zeros_like(en_embed)
        sorted_bert_embed = np.zeros_like(bert_embed[:, :en_dim])
        for idx, (r, i, j) in enumerate(rij):
            sorted_en_embed[:, idx] = en_embed[:, i]
            sorted_bert_embed[:, idx] = bert_embed[:, j]

        swap_corr_matrix = np.zeros((en_dim, en_dim))
        for _, i, j in rij:
            k = j2k[j]
            swap_corr_matrix[k, :] = corr_matrix[i, :]

        both_swap_corr_matrix = swap_corr_matrix.copy()
        both_swap_corr_matrix = \
            both_swap_corr_matrix[
                np.argsort(-np.diag(swap_corr_matrix)), :]

        for i in tqdm(range(en_dim)):
            argmax = i + np.argmax(both_swap_corr_matrix[i][i:])
            tmp = both_swap_corr_matrix[:, i].copy()
            both_swap_corr_matrix[:, i] = both_swap_corr_matrix[:, argmax]
            both_swap_corr_matrix[:, argmax] = tmp

        # save
        en_bert = (sorted_en_embed, en_id2word, en_word2id,
                   sorted_bert_embed, bert_id2token, bert_token2id,
                   bert_sents, bert_org_token2ids, both_swap_corr_matrix)

        with open(en_bert_path, 'wb') as f:
            pkl.dump(en_bert, f)

    output_path = root_dir / 'output/dynamic/figures/axis_corr/'\
        f'{pp}-axis_corr-en_{langs}_{emb_type}-bert_{num_token}-{dpi}dpi.png'
    (root_dir / 'output/dynamic/figures/axis_corr'
     ).mkdir(exist_ok=True, parents=True)
    logger.info(f'output_path: {output_path}')
    save_corr_matrix_fig(both_swap_corr_matrix, output_path,
                         'fastText', 'BERT', 100, pp=pp, dpi=dpi)


if __name__ == '__main__':
    main()
