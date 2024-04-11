import argparse
import pickle as pkl
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

from utils import get_lang_name, get_logger

warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Show correlation scatter plot.')

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

    if not dumped_path.exists():
        raise FileNotFoundError(
            f'{dumped_path} not found. '
            'Run crosslingual_show_embeddings_heatmap.py first.')

    with open(dumped_path, 'rb') as f:
        sorted_src_embed, sorted_tgt_embeds, _, _, \
            src_id2word, tgt_id2words, \
            src_word2id, tgt_word2ids, sw2lang2tw = pkl.load(f)

    # src_color = 'deepskyblue'  # 'lightskyblue'
    # es, ru, ar, hi, zh, ja
    tgt_colors = [''] * len(tgt_list)
    tgt_colors[0] = 'crimson'  # 'lightsalmon'
    tgt_colors[1] = 'lightslategray'  # 'lightsteelblue'
    tgt_colors[2] = 'mediumorchid'  # 'thistle'
    tgt_colors[3] = 'goldenrod'  # 'khaki'
    tgt_colors[4] = 'pink'  # 'lightpink'
    tgt_colors[5] = 'limegreen'  # 'lightgreen'

    axis_num = 300
    ts = 50
    ls = 40

    # skewness
    fig, ax = plt.subplots(figsize=(10, 10))
    rss = []
    for i, tgt in enumerate(tgt_list):

        src2tgt_dict_path = root_dir /\
            f'data/crosslingual/MUSE/dictionaries/{src}-{tgt}.txt'
        with open(src2tgt_dict_path, 'r') as f:
            pairs = [line.strip().split() for line in f.readlines()]

        src_vs = []
        tgt_vs = []
        for s, t in pairs:
            if s in src_word2id and t in tgt_word2ids[i]:
                src_id = src_word2id[s]
                tgt_id = tgt_word2ids[i][t]
                src_vs.append(sorted_src_embed[src_id])
                tgt_vs.append(sorted_tgt_embeds[i][tgt_id])
        src_vs = np.array(src_vs)
        tgt_vs = np.array(tgt_vs)
        logger.info(f'use pairs: {len(src_vs)}/{len(pairs)}')

        rs = []
        for j in range(axis_num):
            rs.append(
                pearsonr(src_vs[:, j],
                         tgt_vs[:, j])[0])

        ax.scatter(np.arange(axis_num),
                   rs,
                   color=tgt_colors[i],
                   label=get_lang_name(tgt), alpha=0.66, s=150)

        rss.append(rs)

    rss = np.array(rss)
    rs_means = np.mean(rss, axis=0)
    assert sorted(rs_means, reverse=True) == list(rs_means)

    ax.plot(np.arange(axis_num),
            rs_means,
            color='black', label='average',
            linewidth=3.0, linestyle='--')

    # x tick is [0, 100, 200, 300]
    ax.set_xlabel('axis', fontsize=ls+5)
    if axis_num == 300:
        ax.set_xticks(np.arange(0, 301, 100))
        ax.set_ylim(-0.1, 0.81)
        ax.set_yticks(np.arange(0, 0.81, 0.2))
    else:
        ax.set_xticks(np.arange(0, axis_num + 1, 20))

    ax.set_title("Pearson's $r$", fontsize=ts)

    ax.tick_params(labelsize=ls)
    if not pca:
        ax.legend(loc='upper right', fontsize=ls-8)
    fig.tight_layout()

    output_dir = root_dir / 'output/crosslingual/figures/corr_scatter'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir /\
        f'{pp}-corr_scatter-{langs}-{emb_type}-{dpi}dpi.png'
    fig.savefig(output_path, dpi=dpi)


if __name__ == '__main__':
    main()
