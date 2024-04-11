import argparse
import pickle as pkl
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import get_lang_name, get_logger

warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Show log skew and kurtosis.')

    # root directory
    parser.add_argument('--root_dir', type=str, default='/working')

    # language
    parser.add_argument("--src", type=str, default="en")
    parser.add_argument("--tgts", type=str, default="es-ru-ar-hi-zh-ja")

    # embedding
    parser.add_argument("--emb_type", type=str, default="cc")
    parser.add_argument("--pca", action='store_true')

    # figure
    parser.add_argument("--dpi", type=int, default=150)

    return parser.parse_args()


def skew_sort(vecs):
    # positive direction
    skews = np.mean(vecs**3, axis=0)
    assert skews.all() >= 0
    # sort by skewness
    vecs = vecs[:, np.argsort(-skews)]
    return vecs


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
        corr_sorted_src_embed, corr_sorted_tgt_embeds, _, _, \
            src_id2word, tgt_id2words, \
            src_word2id, tgt_word2ids, sw2lang2tw = pkl.load(f)

    # skew sort
    sorted_src_embed = skew_sort(corr_sorted_src_embed)
    sorted_tgt_embeds = [skew_sort(corr_sorted_tgt_embed)
                         for corr_sorted_tgt_embed in corr_sorted_tgt_embeds]

    # setting
    ts = 50
    ls = 40
    axis_num = sorted_src_embed.shape[1]
    src_color = 'deepskyblue'  # 'lightskyblue'
    tgt_colors = [''] * len(tgt_list)
    # es, ru, ar, hi, zh, ja
    tgt_colors[0] = 'crimson'  # 'lightsalmon'
    tgt_colors[1] = 'lightslategray'  # 'lightsteelblue'
    tgt_colors[2] = 'mediumorchid'  # 'thistle'
    tgt_colors[3] = 'goldenrod'  # 'khaki'
    tgt_colors[4] = 'pink'  # 'lightpink'
    tgt_colors[5] = 'limegreen'  # 'lightgreen'
    output_dir = root_dir / 'output/crosslingual/figures/log_skew_and_kurt'
    output_dir.mkdir(parents=True, exist_ok=True)

    lang2data = defaultdict(list)

    # skewness
    fig, ax = plt.subplots(figsize=(10, 10))
    skewness = np.mean(sorted_src_embed**3, axis=0)
    lang2data[src].append(skewness)
    ax.plot(np.arange(axis_num),
            np.log(skewness+1),
            color=src_color, label=get_lang_name(src),
            linewidth=5.0)
    for i, tgt in enumerate(tgt_list):
        skewness = np.mean(
            sorted_tgt_embeds[i]**3, axis=0)
        lang2data[tgt].append(skewness)
        ax.plot(np.arange(axis_num),
                np.log(skewness+1),
                color=tgt_colors[i], label=get_lang_name(tgt),
                linewidth=5.0)
    ax.set_xlabel('axis', fontsize=ls+5)
    ax.set_xticks(np.arange(0, axis_num+1, axis_num//3))
    ax.set_title('log(skewness+1)', fontsize=ts)
    ax.tick_params(labelsize=ls)
    ax.legend(loc='upper right', fontsize=ls-8)
    fig.tight_layout()
    fig.savefig(output_dir / f'{pp}-log_skew-{langs}-{emb_type}-{dpi}dpi.png',
                dpi=dpi)

    # kurtosis
    fig, ax = plt.subplots(figsize=(10, 10))
    kurtosis = np.mean(sorted_src_embed**4, axis=0)-3
    lang2data[src].append(kurtosis)
    ax.plot(np.arange(axis_num),
            np.log((np.mean(sorted_src_embed**4, axis=0)-3)+1),
            color=src_color, label=get_lang_name(src),
            linewidth=5.0)
    for i, tgt in enumerate(tgt_list):
        kurtosis = np.mean(sorted_tgt_embeds[i]**4, axis=0)-3
        lang2data[tgt].append(kurtosis)
        ax.plot(np.arange(axis_num),
                np.log((np.mean(sorted_tgt_embeds[i]**4, axis=0)-3)+1),
                color=tgt_colors[i], label=get_lang_name(tgt),
                linewidth=5.0)
    ax.set_xlabel('axis', fontsize=ls+5)
    ax.set_xticks(np.arange(0, axis_num+1, axis_num//3))
    ax.set_title('log(kurtosis+1)', fontsize=ts)
    ax.tick_params(labelsize=ls)
    fig.tight_layout()
    fig.savefig(output_dir / f'{pp}-log_kurt-{langs}-{emb_type}-{dpi}dpi.png',
                dpi=dpi)

    rows = []
    for lang, data in lang2data.items():
        skew = data[0]
        kurt = data[1]
        rows.append({
            'lang': lang,
            'skew_mean': f'{np.mean(skew):.2f}',
            'skew_median': f'{np.median(skew):.2f}',
            'kurt_mean': f'{np.mean(kurt):.2f}',
            'kurt_median': f'{np.median(kurt):.2f}',
        })
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / f'{pp}-log_skew_and_kurt-{langs}-{emb_type}.csv',
              index=False)


if __name__ == '__main__':
    main()
