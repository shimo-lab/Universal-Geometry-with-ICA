import argparse
import pickle as pkl
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text

from utils import get_font_prop, get_logger, get_top_words_and_ids

warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Show scatterplot projection.')

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


def draw_scatter_proj(output_path, lang, word2id, top_words,
                      normed_sorted_embed, axis_num,
                      color, pca, suf='png', root_dir=None, dpi=150):

    font_prop = get_font_prop(lang, root_dir)

    top_ids = [word2id[word] for word in top_words]

    proj_matrix = []
    for axis_idx in range(axis_num):
        theta = 2 * np.pi * axis_idx / axis_num
        proj_matrix.append((np.cos(theta), np.sin(theta)))
    proj_matrix = np.array(proj_matrix)
    proj_matrix /= np.linalg.norm(proj_matrix,
                                  axis=1)[:, None]  # (axis_num, 2)

    normed_sorted_embed = normed_sorted_embed[:, :axis_num]
    proj_embed = np.dot(normed_sorted_embed, proj_matrix)  # (n, 2)
    abs_max_x = 0.8
    abs_max_y = 0.7

    _, ax = plt.subplots(figsize=(10, 9))

    # axis_name = ['first name',
    #              'ships-and-sea',
    #              'country names',
    #              'plants',
    #              'meals']

    for axis_idx in range(axis_num):
        # draw axis
        theta = 2 * np.pi * axis_idx / axis_num
        x = abs_max_x * np.cos(theta) * 1.1
        y = abs_max_y * np.sin(theta) * 1.1
        point = {'start': (0, 0), 'end': (x, y)}
        ax.annotate('', xy=point['end'], xytext=point['start'],
                    arrowprops=dict(shrink=0, width=0.1, headwidth=8,
                                    headlength=10, connectionstyle='arc3',
                                    facecolor='gray', edgecolor='gray')
                    )
        tx = x * 1.05
        ty = y * 1.05

        ax.text(tx, ty, f'{axis_idx}', fontsize=25,
                ha='center', va='center', color='gray')

    ax.scatter(proj_embed[:, 0],
               proj_embed[:, 1],
               s=50, marker='o', color=color)

    ax.scatter(proj_embed[top_ids, 0],
               proj_embed[top_ids, 1],
               s=100, marker='o',
               color='black')

    fs = 16
    if axis_num <= 5:
        # paper setting
        if lang == 'ar':
            texts = [ax.text(proj_embed[id_][0],
                             proj_embed[id_][1],
                             f'{i} {top_words[i][::-1]}',
                             ha='center', va='center',
                             fontsize=fs, color='black',
                             fontproperties=font_prop
                             ) for i, id_ in enumerate(top_ids)]
        elif lang != 'en':
            texts = [ax.text(proj_embed[id_][0],
                             proj_embed[id_][1],
                             f'{i} {top_words[i]}',
                             ha='center', va='center',
                             fontsize=fs, color='black',
                             fontproperties=font_prop
                             ) for i, id_ in enumerate(top_ids)]
        else:
            # avoid overlapping for English, not necessary.
            texts = []
            for axis_idx in range(axis_num):
                if not pca and axis_idx == 2:
                    dx = 0.15
                    dy = 0
                else:
                    dx = 0
                    dy = 0
                mean_x = np.mean(
                    proj_embed[top_ids[axis_idx * 5: (axis_idx + 1) * 5], 0])
                mean_y = np.mean(
                    proj_embed[top_ids[axis_idx * 5: (axis_idx + 1) * 5], 1])
                for i in range(5):
                    idx = axis_idx * 5 + i
                    id_ = top_ids[idx]
                    sx = (-1) * int(proj_embed[id_, 0] < mean_x)
                    sy = (-1) * int(proj_embed[id_, 1] < mean_y)

                    texts.append(ax.text(proj_embed[id_][0] + sx * dx,
                                         proj_embed[id_][1] + sy * dy,
                                         f'{idx} {top_words[idx]}',
                                         ha='center', va='center',
                                         fontsize=fs, color='black',
                                         fontproperties=font_prop))

        # adjust text position
        adjust_text(texts, force_pull=(0.1, 0.1),
                    force_text=(0.05, 0.05),
                    force_explode=(0.005, 0.005),
                    expand_axes=False)
    else:
        # note that ignore the order of characters for Arabic
        texts = [ax.text(proj_embed[id_][0],
                         proj_embed[id_][1],
                         f'{i}',
                         ha='center', va='center',
                         fontsize=fs, color='black',
                         fontproperties=font_prop
                         ) for i, id_ in enumerate(top_ids)]
    ax.axis('equal')

    # save
    plt.xlim(-abs_max_x*1.2, abs_max_x*1.4)
    plt.ylim(-abs_max_y*1.2, abs_max_y*1.2)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.axis('off')

    if suf == 'png':
        plt.savefig(output_path, dpi=dpi)
    else:
        plt.savefig(output_path)
    plt.close()


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
        _, _, normed_sorted_src_embed, normed_sorted_tgt_embeds, \
            src_id2word, tgt_id2words, \
            src_word2id, tgt_word2ids, sw2lang2tw = pkl.load(f)

    logger.info('start calculating...')
    axis_num = 5
    src_words, _, tgt_words_list, _ = get_top_words_and_ids(
        src_id2word, tgt_id2words,
        src_word2id, tgt_word2ids,
        normed_sorted_src_embed, normed_sorted_tgt_embeds,
        sw2lang2tw, tgt_list, axis_num)

    src_color = 'lightskyblue'
    tgt_colors = [''] * len(tgt_list)
    # es, ru, ar, hi, zh, ja
    tgt_colors[0] = 'lightsalmon'
    tgt_colors[1] = 'lightsteelblue'
    tgt_colors[2] = 'thistle'
    tgt_colors[3] = 'khaki'
    tgt_colors[4] = 'lightpink'
    tgt_colors[5] = 'lightgreen'
    suf = 'png'
    output_dir = root_dir / 'output/crosslingual/figures/scatter_projection/'
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir /\
        f'{pp}-scatter_projection-{src}-'\
        f'{emb_type}-{axis_num}axis-{dpi}dpi.{suf}'
    logger.info(f'{src}: {output_path}')
    draw_scatter_proj(output_path, src, src_word2id, src_words,
                      normed_sorted_src_embed, axis_num,
                      src_color, pca, suf=suf, root_dir=root_dir, dpi=dpi)
    for tx, tgt in enumerate(tgts.split('-')):
        output_path = output_dir /\
            f'{pp}-scatter_projection-{tgt}-'\
            f'{emb_type}-{axis_num}axis-{dpi}dpi.{suf}'
        logger.info(f'{tgt}: {output_path}')
        draw_scatter_proj(output_path,
                          tgt, tgt_word2ids[tx], tgt_words_list[tx],
                          normed_sorted_tgt_embeds[tx], axis_num,
                          tgt_colors[tx], pca, suf=suf,
                          root_dir=root_dir, dpi=dpi)


if __name__ == '__main__':
    main()
