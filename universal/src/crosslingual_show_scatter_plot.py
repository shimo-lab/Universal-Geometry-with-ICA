import argparse
import pickle as pkl
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
from adjustText import adjust_text

from utils import (get_font_prop, get_lang_name, get_logger,
                   get_top_words_and_ids)

warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description='Show scatterplot.')

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


def draw_scatterplot(root_dir, lang, color, xs, ys, axis_idx, axis_jdx,
                     idx_id2word, jdx_id2word, min_x, max_x, min_y, max_y,
                     save_path, pca, dpi):

    font_prop = get_font_prop(lang, root_dir)

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.subplots_adjust(left=0.15, right=0.95, bottom=0.1, top=0.9)
    ax.scatter(xs, ys, color=color, s=20)

    ts = 16
    texts = []
    for id_, word in idx_id2word.items():
        ax.scatter(xs[id_], ys[id_], color='black', s=40)
        texts.append(ax.text(xs[id_], ys[id_], word,
                             fontproperties=font_prop, fontsize=ts))

    for id_, word in jdx_id2word.items():
        ax.scatter(xs[id_], ys[id_], color='black', s=40)
        texts.append(ax.text(xs[id_], ys[id_], word,
                             fontproperties=font_prop, fontsize=ts))

    adjust_text(texts, force_pull=(0.1, 0.1),
                force_text=(0.05, 0.05),
                force_explode=(0.005, 0.005),
                expand_axes=False)

    fs = 25
    ax.set_xlabel(f'axis {axis_idx}', fontsize=fs)
    ax.set_ylabel(f'axis {axis_jdx}', fontsize=int(fs*1.1))

    ls = 25
    ax.tick_params(labelsize=ls)

    # title
    ts = 40
    ax.set_title(f'{get_lang_name(lang)}', fontsize=ts, pad=20)

    if pca:
        ax.set_xlim(min_x*1.05, max_x*1.05)
        ax.set_ylim(min_y*1.05, max_y*1.05)
    else:
        ax.set_xlim(min_x, max_x*1.3)
        ax.set_ylim(min_y, max_y*1.3)

    # axis
    ax.axhline(0, color='black', lw=1)
    ax.axvline(0, color='black', lw=1)

    plt.savefig(save_path, dpi=dpi)


def draw(root_dir, src, tgts,
         sorted_src_embed, sorted_tgt_embeds,
         src_id2word, tgt_id2words,
         src_word_ids, tgt_word_ids_list,
         output_dir, pca, dpi):

    src_color = 'lightskyblue'
    tgt_colors = [''] * len(tgts.split('-'))
    # es, ru, ar, hi, zh, ja
    tgt_colors[0] = 'lightsalmon'
    tgt_colors[1] = 'lightsteelblue'
    tgt_colors[2] = 'thistle'
    tgt_colors[3] = 'khaki'
    tgt_colors[4] = 'lightpink'
    tgt_colors[5] = 'lightgreen'

    for axis_idx in range(5):
        for axis_jdx in range(axis_idx + 1, 5):
            print(f'axis {axis_idx} - axis {axis_jdx}')

            (output_dir / f'axis{axis_idx}-axis{axis_jdx}').mkdir(
                exist_ok=True, parents=True)

            if pca:
                max_x = sorted_src_embed[:, axis_idx].max()
                max_y = sorted_src_embed[:, axis_jdx].max()
                for tx, tgt in enumerate(tgts.split('-')):
                    max_x = max(
                        max_x, sorted_tgt_embeds[tx][:, axis_idx].max())
                    max_y = max(
                        max_y, sorted_tgt_embeds[tx][:, axis_jdx].max())

            else:
                # ica embedding is spiky, so we need to adjust the range
                max_x = -10**10
                for i in range(axis_idx * 5, (axis_idx + 1) * 5):
                    id_ = src_word_ids[i]
                    max_x = max(max_x, sorted_src_embed[id_, axis_idx])

                max_y = -10**10
                for i in range(axis_jdx * 5, (axis_jdx + 1) * 5):
                    id_ = src_word_ids[i]
                    max_y = max(max_y, sorted_src_embed[id_, axis_jdx])

                for tx, tgt in enumerate(tgts.split('-')):
                    for i in range(axis_idx * 5, (axis_idx + 1) * 5):
                        id_ = tgt_word_ids_list[tx][i]
                        max_x = max(
                            max_x, sorted_tgt_embeds[tx][id_, axis_idx])
                    for i in range(axis_jdx * 5, (axis_jdx + 1) * 5):
                        id_ = tgt_word_ids_list[tx][i]
                        max_y = max(
                            max_y, sorted_tgt_embeds[tx][id_, axis_jdx])

            min_x = sorted_src_embed[:, axis_idx].min()
            min_y = sorted_src_embed[:, axis_jdx].min()
            for tx, tgt in enumerate(tgts.split('-')):
                min_x = min(min_x, sorted_tgt_embeds[tx][:, axis_idx].min())
                min_y = min(min_y, sorted_tgt_embeds[tx][:, axis_jdx].min())

            xs = sorted_src_embed[:, axis_idx]
            ys = sorted_src_embed[:, axis_jdx]

            save_path = output_dir / f'axis{axis_idx}-axis{axis_jdx}' /\
                f'{src}-{axis_idx}-{axis_jdx}.png'

            idx_id2word = {}
            for i in range(axis_idx * 5, (axis_idx + 1) * 5):
                idx_id2word[src_word_ids[i]] = \
                    f'{i} {src_id2word[src_word_ids[i]]}'
            jdx_id2word = {}
            for i in range(axis_jdx * 5, (axis_jdx + 1) * 5):
                jdx_id2word[src_word_ids[i]] = \
                    f'{i} {src_id2word[src_word_ids[i]]}'
            draw_scatterplot(root_dir, src, src_color, xs, ys,
                             axis_idx, axis_jdx, idx_id2word, jdx_id2word,
                             min_x, max_x, min_y, max_y, save_path, pca, dpi)

            for tx, tgt in enumerate(tgts.split('-')):
                xs = sorted_tgt_embeds[tx][:, axis_idx]
                ys = sorted_tgt_embeds[tx][:, axis_jdx]
                save_path = output_dir / f'axis{axis_idx}-axis{axis_jdx}' /\
                    f'{tgt}-{axis_idx}-{axis_jdx}.png'
                idx_id2word = {}
                for i in range(axis_idx * 5, (axis_idx + 1) * 5):
                    idx_id2word[tgt_word_ids_list[tx][i]] = \
                        f'{i} {tgt_id2words[tx][tgt_word_ids_list[tx][i]]}'
                jdx_id2word = {}
                for i in range(axis_jdx * 5, (axis_jdx + 1) * 5):
                    jdx_id2word[tgt_word_ids_list[tx][i]] = \
                        f'{i} {tgt_id2words[tx][tgt_word_ids_list[tx][i]]}'
                draw_scatterplot(root_dir, tgt, tgt_colors[tx], xs, ys,
                                 axis_idx, axis_jdx, idx_id2word, jdx_id2word,
                                 min_x, max_x, min_y, max_y,
                                 save_path, pca, dpi)


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
        sorted_src_embed, sorted_tgt_embeds, \
            normed_sorted_src_embed, normed_sorted_tgt_embeds, \
            src_id2word, tgt_id2words, \
            src_word2id, tgt_word2ids, sw2lang2tw = pkl.load(f)

    logger.info('start calculating...')
    axis_num = 5
    _, src_word_ids, _, tgt_word_ids_list = get_top_words_and_ids(
        src_id2word, tgt_id2words,
        src_word2id, tgt_word2ids,
        normed_sorted_src_embed, normed_sorted_tgt_embeds,
        sw2lang2tw, tgt_list, axis_num)

    output_dir = root_dir / f'output/crosslingual/figures/scatter_plot/{langs}'
    draw(root_dir, src, tgts, sorted_src_embed, sorted_tgt_embeds,
         src_id2word, tgt_id2words, src_word_ids, tgt_word_ids_list,
         output_dir, pca, dpi)


if __name__ == '__main__':
    main()
