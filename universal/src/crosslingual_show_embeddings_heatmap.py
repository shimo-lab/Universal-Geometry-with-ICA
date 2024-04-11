import argparse
import pickle as pkl
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.stats import pearsonr
from tqdm import tqdm

from utils import (get_font_prop, get_lang_name, get_logger,
                   get_top_words_and_ids, pos_direct)

warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description='show embeddings heatmap')

    # root directory
    parser.add_argument('--root_dir', type=str, default='/working')

    # language
    parser.add_argument('--src', type=str, default='en')
    parser.add_argument('--tgts', type=str, default='es-ru-ar-hi-zh-ja')

    # embedding
    parser.add_argument('--emb_type', type=str, default='cc')
    parser.add_argument('--pca', action='store_true')

    # figure
    parser.add_argument('--dpi', type=int, default=150)

    return parser.parse_args()


def sub_figure(lang, embed, wids, id2word, ax, cmap, show_word, vmin,
               show_title, show_rectangle, fig, n, root_dir, cb_ax=None):

    font_prop = get_font_prop(lang, root_dir)

    cbar = cb_ax is not None
    g = sns.heatmap(embed[wids], yticklabels=[
        id2word[wid] for wid in wids], cmap=cmap,
        ax=ax, vmin=vmin, vmax=1.0, cbar_ax=cb_ax, cbar=cbar)

    g.tick_params(left=False, bottom=True, labelsize=25)

    if show_word:
        yticklabels = g.get_yticklabels()
        if lang == 'ar':
            # Arabic is written from right to left
            yticklabels = [yticklabel.get_text()[::-1]
                           for yticklabel in yticklabels]
        ax.set_yticklabels(yticklabels,
                           rotation=0, fontproperties=font_prop)
    else:
        ax.set_yticklabels([])

    if show_title:
        ax.text(0.5, 1.1, get_lang_name(lang), fontsize=50,
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes)

    if not show_word:
        ax.set_xticks(range(0, len(embed[wids][0]), 20))
        ax.set_xticklabels(range(0, len(embed[wids][0]), 20), rotation=0)

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

    # coordinate of the rectangle
    x = -padding * len(embed[wids][0])
    y = -padding * len(embed[wids])
    width = len(embed[wids][0]) * (s + padding)
    height = len(embed[wids]) * (s + padding)

    # Draw the border of the rectangle
    lines = [Line2D([x, x + width], [y, y],
                    lw=lw, color='black'),
             Line2D([x + width, x + width], [y, y + height],
                    lw=lw, color='black'),
             Line2D([x + width, x], [y + height, y + height],
                    lw=lw, color='black'),
             Line2D([x, x], [y + height, y],
                    lw=lw, color='black')]
    for line in lines:
        ax.add_line(line)


def heatmap(src, tgts, sorted_src_embed, sorted_tgt_embeds,
            src_words, tgt_words_list,
            src_id2word, tgt_id2words, src_word2id, tgt_word2ids,
            img_path, dpi, root_dir):

    # show src and tgt embeddings
    col = 1 + len(tgts.split('-'))
    figx = 5 * col
    figy = 5.5 * 2
    n = len(src_words)
    wspace = 0.5

    fig = plt.figure(figsize=(figx, figy))
    gs1 = gridspec.GridSpec(1, col, figure=fig, width_ratios=[1]*col,
                            wspace=wspace, bottom=0.525, top=0.9)
    gs2 = gridspec.GridSpec(1, col, figure=fig, width_ratios=[1]*col,
                            wspace=wspace, bottom=0.05, top=0.45)
    fig.subplots_adjust(left=0.05, right=0.925)
    gss = [gs1, gs2]

    cb_ax = fig.add_axes([.95, 0.05, .015, .85])
    cb_ax.tick_params(labelsize=30)

    cmap = 'magma_r'
    vmin = -0.1
    words_per_axis = 5
    num_words = 5 * words_per_axis
    for rx in range(2):

        if rx == 0:
            # show 100 axis
            show_word = False
            show_title = True
            show_rectangle = True

        else:
            # show 5 axis with 5 words for each axis
            show_word = True
            show_title = False
            show_rectangle = False

        if rx == 1:
            src_words = src_words[:num_words]
            sorted_src_embed = sorted_src_embed[:, :words_per_axis]

            tgt_words_list = [tgt_words[:num_words]
                              for tgt_words in tgt_words_list]
            sorted_tgt_embeds = [sorted_tgt_embed[:, :words_per_axis]
                                 for sorted_tgt_embed in sorted_tgt_embeds]

        src_wids = [src_word2id[word] for word in src_words]

        if rx == 1:
            cb_ax = None
        # sub_figure for each language
        ax_src = fig.add_subplot(gss[rx][0])
        sub_figure(src, sorted_src_embed, src_wids,
                   src_id2word, ax_src, cmap, show_word, vmin,
                   show_title, show_rectangle, fig, n, root_dir, cb_ax)

        for tx, (tgt, tgt_words) in enumerate(
                zip(tgts.split('-'), tgt_words_list)):
            tgt_wids = [tgt_word2ids[tx][word] for word in tgt_words]
            ax_tgt = fig.add_subplot(gss[rx][tx+1])
            sub_figure(tgt, sorted_tgt_embeds[tx], tgt_wids,
                       tgt_id2words[tx], ax_tgt, cmap, show_word, vmin,
                       show_title, show_rectangle, fig, n, root_dir)

    if Path(img_path).suffix == '.png':
        assert dpi is not None
        plt.savefig(img_path, dpi=dpi)
    else:
        plt.savefig(img_path)


def main():
    args = parse_args()
    root_dir = Path(args.root_dir)

    src = args.src
    tgts = args.tgts
    tgt_list = tgts.split('-')
    langs = f'{src}-{tgts}'

    emb_type = args.emb_type
    assert emb_type in ['cc', 'muse']
    pca = args.pca
    # post-processing
    pp = 'pca' if pca else 'ica'

    dpi = args.dpi

    logger = get_logger()
    logger.info(f'langs: {langs}')

    dumped_path = root_dir /\
        f'output/crosslingual/{langs}/axis_matching_{pp}.pkl'
    if dumped_path.exists():
        with open(dumped_path, 'rb') as f:
            _, _, normed_sorted_src_embed, normed_sorted_tgt_embeds, \
                src_id2word, tgt_id2words, \
                src_word2id, tgt_word2ids, sw2lang2tw = \
                pkl.load(f)
    else:

        if emb_type == 'cc':
            src_dump_path = root_dir /\
                f'output/crosslingual/{langs}/cc.{src}.300_dic_and_emb.pkl'
        else:
            src_dump_path = root_dir /\
                f'output/crosslingual/{langs}/wiki.multi.{src}_dic_and_emb.pkl'

        logger.info(f'src_dump_path: {src_dump_path}')

        with open(src_dump_path, 'rb') as f:
            src_word2id, src_id2word, _, src_pca_embed, src_ica_embed = \
                pkl.load(f)

        if pca:
            src_embed = src_pca_embed
        else:
            src_embed = src_ica_embed

        src_embed = pos_direct(src_embed)
        _, dim = src_embed.shape

        logger.info('load target dictionary and embeddings.')

        sw2lang2tw = defaultdict(lambda: defaultdict(list))
        # 0th element is the original index
        i_rs = [[i] for i in range(dim)]
        tgts_embed = []
        tgt_id2words = []
        tgt_word2ids = []

        for tgt in tgt_list:
            pairs = []
            src2tgt_dict_path = root_dir /\
                f'data/crosslingual/MUSE/dictionaries/{src}-{tgt}.txt'
            logger.info(f'src2tgt_dict_path: {src2tgt_dict_path}')
            with open(src2tgt_dict_path, 'r') as f:
                for line in f.readlines():
                    pairs.append(line.strip().split())

            if emb_type == 'cc':
                tgt_dump_path = root_dir /\
                    f'output/crosslingual/{langs}/cc.{tgt}.300_dic_and_emb.pkl'
            else:
                tgt_dump_path = root_dir / f'output/crosslingual/{langs}/'\
                    f'wiki.multi.{tgt}_dic_and_emb.pkl'
            logger.info(f'tgt_dump_path: {tgt_dump_path}')

            with open(tgt_dump_path, 'rb') as f:
                tgt_word2id, tgt_id2word, _, tgt_pca_embed, tgt_ica_embed = \
                    pkl.load(f)

            tgt_id2words.append(tgt_id2word)
            tgt_word2ids.append(tgt_word2id)

            if pca:
                tgt_embed = tgt_pca_embed
            else:
                tgt_embed = tgt_ica_embed

            tgt_embed = pos_direct(tgt_embed)

            # use translation pairs if both words are in the vocabularies
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

                    sw2lang2tw[s][tgt].append(t)

            src_vs = np.array(src_vs)
            tgt_vs = np.array(tgt_vs)

            logger.info(
                f'use pairs: {len(src_vs)}/{len(pairs)} - '
                f'{100*len(src_vs)/len(pairs):.2f}%')

            # calculate correlation for all axis pairs
            cands = []
            logger.info('start calculating corr_matrix...')
            for i in tqdm(range(dim)):
                ax1 = src_vs[:, i]
                for j in range(dim):
                    ax2 = tgt_vs[:, j]
                    r = pearsonr(ax1, ax2)[0]
                    cands.append((r, i, j))
            cands.sort(reverse=True)

            used_i = set()
            used_j = set()

            # greedy matching
            corr_W = np.zeros((dim, dim))
            for r, i, j in cands:
                if i in used_i or j in used_j:
                    continue
                used_i.add(i)
                used_j.add(j)
                corr_W[j, i] = 1
                i_rs[i].append(r)

            # permute target embeddings for alignment with source embeddings
            tgt_permu_embed = tgt_embed @ corr_W
            tgts_embed.append(tgt_permu_embed)

        # sort by the sum of correlation coefficients
        i_rs.sort(key=lambda irs: sum([r for r in irs[1:]]), reverse=True)

        sorted_src_embed = np.zeros_like(src_embed)
        sorted_tgt_embeds = []
        for _ in range(len(tgt_list)):
            sorted_tgt_embeds.append(np.zeros_like(tgt_embed))
        for idx, ir in enumerate(i_rs):
            # 0th element is the original index
            i = ir[0]
            sorted_src_embed[:, idx] = src_embed[:, i]
            for tx, tgt_embed in enumerate(tgts_embed):
                # note that the target embeddings are already permuted
                sorted_tgt_embeds[tx][:, idx] = tgt_embed[:, i]

        # nomalize embeddings to enhance the interpretability
        normed_sorted_src_embed = sorted_src_embed / np.linalg.norm(
            sorted_src_embed, axis=1, keepdims=True)
        normed_sorted_tgt_embeds = []
        for tx, sorted_tgt_embed in enumerate(sorted_tgt_embeds):
            normed_sorted_tgt_embed = sorted_tgt_embed / np.linalg.norm(
                sorted_tgt_embed, axis=1, keepdims=True)
            normed_sorted_tgt_embeds.append(normed_sorted_tgt_embed)

        logger.info('start saving sorted_embed...')
        sw2lang2tw = dict(sw2lang2tw)
        dump = (sorted_src_embed, sorted_tgt_embeds,
                normed_sorted_src_embed, normed_sorted_tgt_embeds,
                src_id2word, tgt_id2words,
                src_word2id, tgt_word2ids, sw2lang2tw)
        with open(dumped_path, 'wb') as f:
            pkl.dump(dump, f)

    # select top 100 axis and top 5 words for each axis
    logger.info('start calculating...')
    axis_num = 100
    src_words, _, tgt_words_list, _ = get_top_words_and_ids(
        src_id2word, tgt_id2words,
        src_word2id, tgt_word2ids,
        normed_sorted_src_embed, normed_sorted_tgt_embeds,
        sw2lang2tw, tgt_list, axis_num)

    normed_sorted_src_embed = normed_sorted_src_embed[:, :axis_num]
    normed_sorted_tgt_embeds = [normed_sorted_tgt_embed[:, :axis_num]
                                for normed_sorted_tgt_embed in
                                normed_sorted_tgt_embeds]

    output_dir = root_dir / 'output/crosslingual/figures/embeddings_heatmap/'
    output_dir.mkdir(parents=True, exist_ok=True)
    img_path = output_dir /\
        f'{pp}-embeddings_heatmap-{langs}-{emb_type}-{dpi}dpi.png'

    logger.info(f'start saving heatmap to {img_path}...')
    heatmap(src, tgts, normed_sorted_src_embed, normed_sorted_tgt_embeds,
            src_words, tgt_words_list,
            src_id2word, tgt_id2words, src_word2id, tgt_word2ids,
            img_path, dpi, root_dir)


if __name__ == '__main__':
    main()
