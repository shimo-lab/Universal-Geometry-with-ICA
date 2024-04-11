import argparse
import pickle as pkl
from pathlib import Path

import inflect
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from tqdm import tqdm

from utils import get_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Show embeddings heatmap.')

    # root directory
    parser.add_argument('--root_dir', type=str, default='/working')

    # language
    parser.add_argument('--langs', type=str, default='en-es-ru-ar-hi-zh-ja')

    # bert vocabularies
    parser.add_argument('--num_token', type=int, default=100000)

    # embedding
    parser.add_argument('--emb_type', type=str, default='cc')
    parser.add_argument('--pca', action='store_true')

    # figure
    parser.add_argument('--dpi', type=int, default=150)
    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()


def sub_figure(name, embed, wids, id2word, ax, cmap, show_word, vmin,
               show_title, cb_ax=None):

    cbar = cb_ax is not None
    g = sns.heatmap(embed[wids], yticklabels=[
        id2word[wid] for wid in wids], cmap=cmap,
        ax=ax, vmin=vmin, vmax=1.0, cbar_ax=cb_ax, cbar=cbar)

    g.tick_params(left=False, bottom=True, labelsize=32)

    if show_word:
        ax.set_yticklabels(g.get_yticklabels(),
                           rotation=0, fontsize=20)
    else:
        ax.set_yticklabels([])

    if show_title:
        ax.set_title(name, fontsize=50)

    if not show_word:
        ax.set_xticks(range(0, len(embed[wids][0]), 20))
        ax.set_xticklabels(
            range(0, len(embed[wids][0]), 20), rotation=0, fontsize=30)

    s = 1 if show_word else 0.05
    padding = 0.001
    lw = 5 if show_word else 1

    # broadening the range of Axes by padding
    if show_word:
        ax.set_xlim(-len(embed[wids]) * padding,
                    len(embed[wids][0]) * (s + 3 * padding))
        ax.set_ylim(len(embed[wids]) * (s + 3 * padding),
                    -len(embed[wids]) * 3 * padding)
    else:
        ax.set_xlim(-len(embed[wids][0]) * 2 * padding, len(embed[wids][0]))
        ax.set_ylim(len(embed[wids]), -len(embed[wids]) * 2 * padding)

    if cbar:
        cbar = g.collections[0].colorbar
        cbar.ax.tick_params(labelsize=40)

    # convert the coordinate
    trans = ax.transData

    # coordinate of the rectangle
    x = -padding * len(embed[wids][0])
    y = -padding * len(embed[wids])
    width = len(embed[wids][0]) * (s + padding)
    height = len(embed[wids]) * (s + padding)

    # draw the border of the rectangle
    if show_word:
        lines = [Line2D([x, x + width], [y, y],
                        lw=lw, color='black'),
                 Line2D([x + width, x + width], [y, y + height],
                        lw=1.5 * lw, color='black'),
                 Line2D([x + width, x], [y + height, y + height],
                        lw=lw, color='black'),
                 Line2D([x, x], [y + height, y],
                        lw=lw, color='black')]
        for line in lines:
            ax.add_line(line)
    else:
        rect = Rectangle((x, y), width, height, fill=False, edgecolor='black',
                         lw=lw, transform=trans,
                         joinstyle='miter', capstyle='butt')
        ax.add_patch(rect)


def heatmap(src, tgt, sorted_en_embed, sorted_bert_embed,
            en_words, bert_words,
            en_id2word, bert_id2token, en_word2id, bert_token2id,
            img_path, words_per_axis, K, dpi=None):

    col = 2
    figx = 5 * col + 2
    figy = 5 * 5
    wspace = 0.8

    top = 0.95
    bottom = 0.03
    fig = plt.figure(figsize=(figx, figy))
    gs1 = gridspec.GridSpec(1, col, figure=fig,
                            width_ratios=[1, 1],
                            wspace=wspace, bottom=0.83, top=top)
    gs2 = gridspec.GridSpec(1, col, figure=fig,
                            width_ratios=[1, 1],
                            wspace=wspace, bottom=bottom, top=0.8)
    fig.subplots_adjust(left=0.16, right=0.825)
    gss = [gs1, gs2]

    cb_ax = fig.add_axes([.86, bottom, .05, top-bottom])
    cb_ax.tick_params(labelsize=30)

    cmap = 'magma_r'
    vmin = -0.1
    num_words = 5 * words_per_axis
    for rx in range(2):

        if rx == 0:
            show_word = False
            show_title = True

        else:
            show_word = True
            show_title = False

        if rx == 1:
            en_words = en_words[:num_words]
            sorted_en_embed = sorted_en_embed[:, :words_per_axis]

            bert_words = bert_words[:num_words * K]
            sorted_bert_embed = sorted_bert_embed[:, :words_per_axis]

        en_wids = [en_word2id[word] for word in en_words]

        if rx == 1:
            cb_ax = None
        ax_src = fig.add_subplot(gss[rx][0])
        sub_figure(src, sorted_en_embed, en_wids,
                   en_id2word, ax_src, cmap, show_word, vmin,
                   show_title, cb_ax)

        bert_wids = [bert_token2id[word] for word in bert_words]
        ax_tgt = fig.add_subplot(gss[rx][1])
        sub_figure(tgt, sorted_bert_embed, bert_wids,
                   bert_id2token, ax_tgt, cmap, show_word, vmin, show_title)

    if Path(img_path).suffix == '.png':
        assert dpi is not None
        plt.savefig(img_path, dpi=dpi)
    else:
        plt.savefig(img_path)


def main():

    args = parse_args()
    root_dir = Path(args.root_dir)

    args = parse_args()
    root_dir = Path(args.root_dir)

    langs = args.langs

    emb_type = args.emb_type
    pca = args.pca
    pp = 'pca' if pca else 'ica'

    num_token = args.num_token

    dpi = args.dpi
    seed = args.seed
    np.random.seed(seed)

    axis_num = 100
    K = 3
    words_per_axis = 5

    logger = get_logger()

    en_bert_path = root_dir /\
        f'output/dynamic/{pp}-en_{langs}_{emb_type}-bert_{num_token}.pkl'
    if not en_bert_path.exists():
        raise FileNotFoundError(
            f'{en_bert_path} does not exist! '
            'Run dynamic_show_axis_corr.py first.')

    logger.info(f'load {en_bert_path}')
    with open(en_bert_path, 'rb') as f:
        en_bert = pkl.load(f)
    sorted_en_embed, en_id2word, en_word2id, \
        sorted_bert_embed, bert_id2token, bert_token2id, \
        _, bert_org_token2ids, _ = en_bert

    normed_sorted_en_embed = sorted_en_embed / \
        np.linalg.norm(sorted_en_embed, axis=1, keepdims=True)

    normed_sorted_bert_embed = sorted_bert_embed / \
        np.linalg.norm(sorted_bert_embed, axis=1, keepdims=True)

    logger.info('start calculating...')
    p = inflect.engine()
    en_words = []
    bert_words = []
    for idx in tqdm(list(range(axis_num))):
        en_axis = normed_sorted_en_embed[:, idx]
        bert_axis = normed_sorted_bert_embed[:, idx]

        wids = np.argsort(-en_axis)
        en_tmp = []
        bert_tmp = []
        for wi in wids:
            en_word = en_id2word[wi]
            if en_word in en_tmp or en_word in en_words:
                continue

            # plural check
            flag = p.singular_noun(en_word)

            if flag is not False:
                en_singluar_word = p.singular_noun(en_word)
                if en_singluar_word in en_tmp or \
                        en_singluar_word in en_words:
                    continue
            else:
                en_plural_word = p.plural_noun(en_word)
                if en_plural_word in en_tmp or \
                        en_plural_word in en_words:
                    continue

            if en_id2word[wi] not in bert_org_token2ids:
                continue
            tids = bert_org_token2ids[en_id2word[wi]]
            if len(tids) < K:
                continue

            # randomly sample K words
            sampled_ids = np.random.choice(tids, K, replace=False)
            ids = np.argsort(
                -np.array([bert_axis[id] for id in sampled_ids]))
            random_bert_words = [bert_id2token[sampled_ids[id]] for id in ids]
            en_tmp.append(en_word)
            bert_tmp += random_bert_words

            if len(en_tmp) == 5:
                break

        en_words += en_tmp
        bert_words += bert_tmp

        if len(en_words) == 5 * axis_num:
            break

    normed_sorted_en_embed = normed_sorted_en_embed[:, :axis_num]
    normed_sorted_bert_embed = normed_sorted_bert_embed[:, :axis_num]

    img_path = root_dir / 'output/dynamic/figures/embeddings_heatmap/'\
        f'{pp}-embeddings_heatmap-en_{langs}_{emb_type}-'\
        f'bert_{num_token}-{dpi}dpi.png'
    (root_dir / 'output/dynamic/figures/embeddings_heatmap').mkdir(
        parents=True, exist_ok=True)
    logger.info(f'img_path: {img_path}')
    heatmap('fastText', 'BERT',
            normed_sorted_en_embed, normed_sorted_bert_embed,
            en_words, bert_words,
            en_id2word, bert_id2token, en_word2id, bert_token2id,
            img_path, words_per_axis, K, dpi)


if __name__ == '__main__':
    main()
