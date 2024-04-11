import argparse
import pickle as pkl
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from gensim.parsing.preprocessing import STOPWORDS
from tqdm import tqdm

from utils import get_logger, long_name2title, pos_direct, short_name2long_name
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

    # image model
    parser.add_argument("--bases", type=str,
                        default="vit-resmlp-swin-resnet-regnety")

    # figure
    parser.add_argument("--dpi", type=int, default=150)

    return parser.parse_args()


def save_corr_matrix_fig(corr_matrix, img_path, ref, tgt, dim=300,
                         pp='ica', dpi=None):

    corr_matrix = corr_matrix[:dim, :dim]

    labelsize = 50
    titlesize = 50
    fig, ax = plt.subplots(figsize=(15, 12))
    fig.subplots_adjust(left=0.15, right=0.925, bottom=0.16, top=0.95)

    if dim > 100:
        xticlabels = 50
        yticlabels = 50
    elif dim == 100:
        xticlabels = 20
        yticlabels = 20

    ax = sns.heatmap(corr_matrix, xticklabels=xticlabels,
                     yticklabels=yticlabels,
                     cmap='RdBu_r', vmin=-1.0, vmax=1.0, square=True,
                     cbar_kws={"shrink": 1.0,
                               "ticks": np.arange(-1, 1.25, 0.25)})
    ax.xaxis.labelpad = 20
    upper_pp = pp.upper()
    base_title = f'Permuted {upper_pp} Axis of {tgt}'
    ref_title = f'Permuted {upper_pp} Axis of {ref}'
    ax.set_xlabel(base_title, fontsize=titlesize)

    yticks = np.flip(np.arange(0, dim, yticlabels))
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks, rotation=0)

    ax.set_ylabel(ref_title, fontsize=titlesize)
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

    if Path(img_path).suffix == '.png':
        assert dpi is not None
        fig.savefig(img_path, dpi=dpi)
    else:
        fig.savefig(img_path)

    plt.close(fig)


def main():

    args = parse_args()

    root_dir = Path(args.root_dir)

    langs = args.langs

    emb_type = args.emb_type
    pca = args.pca
    pp = 'pca' if pca else 'ica'

    bases = args.bases
    base_list = []
    for base in bases.split('-'):
        base_list.append(short_name2long_name(base))
    assert len(base_list) > 1

    dpi = args.dpi

    logger = get_logger()
    logger.info(f'base_list: {base_list}')

    output_dir = root_dir / 'output/image'

    (output_dir / bases).mkdir(parents=True, exist_ok=True)
    dump_path = output_dir / bases /\
        f'{pp}-en_{langs}_{emb_type}-image_{bases}.pkl'
    if dump_path.exists():
        logger.info(f'{dump_path} already exists')
        with open(dump_path, 'rb') as f:
            dump = pkl.load(f)
        normed_sorted_ref_embed, normed_sorted_base_embeds, \
            normed_perm_en_embed, en_id2word, en_id2base_ids, en_word2id, \
            base_id2file_name, both_swap_corr_matrixs = dump
    else:

        # image model
        base_embeds = []
        for bx, base in enumerate(base_list):
            names_pca_ica_embeds_path = output_dir / f'{base}-pca_ica.pkl'
            logger.info(
                f'names_pca_ica_embeds_path: {names_pca_ica_embeds_path}')
            with open(names_pca_ica_embeds_path, 'rb') as f:
                base_id2file_name, _, pca_embed, ica_embed = pkl.load(f)
            if pca:
                base_embed = pos_direct(pca_embed)
            else:
                base_embed = pos_direct(ica_embed)

            logger.info(f'base_embed.shape: {base_embed.shape}')
            base_embeds.append(base_embed)

        # we can regard two same images as a pair:
        # (img1, img1), (img2, img2), ...
        # this is the same procedure as the following translation pairs:
        # ('cat', 'gato'), ('sea', 'mar'), ...
        image_pairs = [(i, i) for i in range(len(base_embeds[0]))]
        image_weights = [1/len(image_pairs) for _ in range(len(image_pairs))]
        logger.info(f'the number of image pairs: {len(image_pairs)}')

        ref_embed = base_embeds[0]
        base_embeds = base_embeds[1:]
        base_dims = []
        for base_embed in base_embeds:
            _, base_dim = base_embed.shape
            base_dims.append(base_dim)
        ref_dim = ref_embed.shape[1]
        logger.info(f'ref_dim: {ref_dim}')

        # calculate correlation matrix
        rlists = [[] for _ in range(ref_dim)]
        corr_matrixs = []
        rijs = []
        for bx, base_embed in enumerate(base_embeds):

            ref_vs = []
            base_vs = []

            for i, j in image_pairs:
                ref_vs.append(ref_embed[i])
                base_vs.append(base_embed[j])

            ref_vs = np.array(ref_vs)
            base_vs = np.array(base_vs)

            logger.info(f'ref_vs.shape: {ref_vs.shape}')
            logger.info(f'base_vs.shape: {base_vs.shape}')

            cands = []
            corr_matrix = [[] for _ in range(ref_dim)]
            logger.info('start calculating corr_matrix...')
            weights = pd.Series(image_weights)
            for i in tqdm(range(ref_dim)):
                ax1 = ref_vs[:, i]
                ax1 = pd.Series(ax1)
                for j in range(base_dims[bx]):
                    ax2 = base_vs[:, j]
                    ax2 = pd.Series(ax2)
                    # This is equivalent to the normal Pearson correlation
                    r = WeightedCorr(x=ax1, y=ax2, w=weights
                                     )(method='pearson')
                    corr_matrix[i].append(r)
                    cands.append((r, i, j))
            corr_matrix = np.array(corr_matrix)
            corr_matrixs.append(corr_matrix)
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
                rlists[i].append(r)
            assert len(rij) == min(ref_dim, base_dims[bx])
            rijs.append(rij)

        # we choose only the index
        # where all base models have the pair with the reference model.
        valid_is = []
        for i in range(ref_dim):
            if len(rlists[i]) == len(base_embeds):
                valid_is.append(i)

        image_min_dim = len(valid_is)
        logger.info(f'image_min_dim: {image_min_dim}')

        # here load fastText model for min_dim
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
        if pca:
            en_embed = pos_direct(en_pca_embed)
        else:
            en_embed = pos_direct(en_ica_embed)
        _, en_dim = en_embed.shape

        min_dim = min(image_min_dim, en_dim)
        logger.info(f'min_dim: {min_dim}')

        # index mapping
        # for ref model
        i2k = {}
        for k, i in enumerate(valid_is):
            i2k[i] = k
        # for base models
        j2ks = [{} for _ in range(len(base_embeds))]
        for bx, base_embed in enumerate(base_embeds):
            k = 0
            for (_, i, j) in sorted(rijs[bx], key=lambda x: x[2]):
                if i in valid_is:
                    j2ks[bx][j] = k
                    k += 1
        assert all(len(j2k) == len(i2k) for j2k in j2ks)

        perm_base_embeds = []
        # for base models and fastText model
        both_swap_corr_matrixs = []
        for bx, base_embed in enumerate(base_embeds):
            # define permutation matrix
            W = np.zeros((image_min_dim, ref_dim))
            for _, i, j in rijs[bx]:
                if i not in i2k:
                    continue
                assert j in j2ks[bx]
                W[j2ks[bx][j], i] = 1

            # compress the correlation matrix.
            # there are axes for both the reference model and the base model.
            tmp_corr_matrix = np.zeros(
                (image_min_dim, corr_matrixs[bx].shape[1]))
            for i in range(ref_dim):
                if i not in i2k:
                    continue
                tmp_corr_matrix[i2k[i], :] = corr_matrixs[bx][i, :]
            corr_matrixs[bx] = tmp_corr_matrix
            tmp_corr_matrix = np.zeros((image_min_dim, image_min_dim))
            for j in range(base_dims[bx]):
                if j not in j2ks[bx]:
                    continue
                tmp_corr_matrix[:, j2ks[bx][j]] = corr_matrixs[bx][:, j]
            corr_matrixs[bx] = tmp_corr_matrix
            assert corr_matrixs[bx].shape == (image_min_dim, image_min_dim)

            # permute the compressed correlation matrix
            swap_corr_matrix = np.zeros_like(corr_matrixs[bx])
            for _, i, j in rijs[bx]:
                if i not in i2k:
                    continue
                swap_corr_matrix[j2ks[bx][j], :] = \
                    corr_matrixs[bx][i2k[i], :]
            both_swap_corr_matrix = swap_corr_matrix.copy()
            both_swap_corr_matrix = both_swap_corr_matrix[
                np.argsort(-np.diag(swap_corr_matrix)), :]
            for i in tqdm(range(image_min_dim)):
                argmax = i + np.argmax(both_swap_corr_matrix[i][i:])
                tmp = both_swap_corr_matrix[:, i].copy()
                both_swap_corr_matrix[:, i] = \
                    both_swap_corr_matrix[:, argmax]
                both_swap_corr_matrix[:, argmax] = tmp
            both_swap_corr_matrixs.append(
                both_swap_corr_matrix[:min_dim, :min_dim])

            # compress base model embeddings
            compress_base_embed = np.zeros_like(
                base_embed[:, :image_min_dim])
            for j in range(base_dims[bx]):
                if j not in j2ks[bx]:
                    continue
                compress_base_embed[:, j2ks[bx][j]] = base_embed[:, j]

            # permute compressed base model embeddings
            perm_base_embed = compress_base_embed @ W
            perm_base_embeds.append(perm_base_embed)

        # sort the axis of the reference model and the base models
        # by the sum of the correlation coefficients.
        valid_i_rs = []
        for i, rlist in enumerate(rlists):
            if i not in valid_is:
                continue
            assert len(rlist) == len(base_embeds)
            assert len(valid_i_rs) == i2k[i]
            valid_i_rs.append((i, sum(rlist)))
        valid_i_rs.sort(key=lambda x: x[1], reverse=True)
        sorted_is = []
        for i, _ in valid_i_rs[:min_dim]:
            sorted_is.append(i)
        sorted_ref_embed = np.zeros_like(ref_embed[:, :min_dim])
        sorted_base_embeds = []
        for idx in range(len(base_embeds)):
            sorted_base_embeds.append(
                np.zeros_like(perm_base_embeds[idx][:, :min_dim]))
        logger.info(f'sorted_ref_embed.shape: {sorted_ref_embed.shape}')
        logger.info(
            f'sorted_base_embeds[0].shape: {sorted_base_embeds[0].shape}')
        for idx, i in enumerate(sorted_is):
            sorted_ref_embed[:, idx] = ref_embed[:, i]
            for bx, base_embed in enumerate(base_embeds):
                sorted_base_embeds[bx][:, idx] = perm_base_embeds[bx][:, i]

        # image reference model to fastText model

        # Imagenet label is sentence.
        # Split the label into words and prepare the label to words dictionary.
        label_path = root_dir / 'data/image/imagenet/LOC_synset_mapping.txt'
        logger.info(f'label_path: {label_path}')
        label2name_words = dict()
        labels = []
        label_names = []
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                line_split = line.split(' ')
                label = line_split[0]
                labels.append(label)
                names = ' '.join(line_split[1:]).split(',')
                label_names.append([name.strip().lower() for name in names
                                    if len(name.strip()) > 0])
                name_words = []
                for name in names:
                    for name_word in name.split(' '):
                        name_word = name_word.strip().lower()
                        if len(name_word) > 0:
                            name_words.append(name_word)
                name_words = list(set(name_words))
                label2name_words[label] = name_words

        # fastText
        # pairs for the word and the image whose label contains the word.
        # note that each label contains 100 images.
        en_pairs = []
        en_id2freq = defaultdict(int)
        en_id2base_ids = defaultdict(list)
        for lx, label in enumerate(labels):
            name_words = label2name_words[label]
            for name_word in name_words:
                if name_word in en_word2id:
                    if name_word in STOPWORDS:
                        continue
                    en_id = en_word2id[name_word]
                    en_id2freq[en_id] += 100
                    for idx in range(100):
                        base_id = 100 * lx + idx
                        en_pairs.append((en_id, base_id))
                        en_id2base_ids[en_id].append(base_id)
        Z = 0
        for _, freq in en_id2freq.items():
            Z += freq
        en_weights = []
        for en_idx, _ in en_pairs:
            en_weights.append(en_id2freq[en_idx] / Z)

        logger.info(f'the number of en_pairs: {len(en_pairs)}')
        logger.info(f'the number of en_id2base_ids: {len(en_id2base_ids)}')

        ref_vs = []
        en_vs = []
        for en_id, ref_id in en_pairs:
            ref_vs.append(sorted_ref_embed[ref_id])
            en_vs.append(en_embed[en_id])

        ref_vs = np.array(ref_vs)
        en_vs = np.array(en_vs)

        # calculate correlation matrix
        cands = []
        corr_matrix = [[] for _ in range(min_dim)]
        logger.info('start calculating corr_matrix...')
        weights = pd.Series(en_weights)
        assert len(weights) == len(ref_vs) == len(en_vs)
        for i in tqdm(range(min_dim)):
            ax1 = ref_vs[:, i]
            ax1 = pd.Series(ax1)
            for j in range(en_dim):
                ax2 = en_vs[:, j]
                ax2 = pd.Series(ax2)
                r = WeightedCorr(x=ax1, y=ax2, w=weights)(
                    method='pearson')
                corr_matrix[i].append(r)
                cands.append((r, i, j))
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
        assert len(rij) == min_dim

        # index mapping and define permutation matrix
        j2k = {}
        W = np.zeros((min_dim, min_dim))
        for k, (_, i, j) in enumerate(sorted(rij, key=lambda x: x[2])):
            j2k[j] = k
            W[k, i] = 1

        # compress the fastText model embeddings and permute them
        compress_en_embed = np.zeros_like(en_embed[:, :min_dim])
        for j in range(en_dim):
            if j not in j2k:
                continue
            compress_en_embed[:, j2k[j]] = en_embed[:, j]
        perm_en_embed = compress_en_embed @ W

        # sort the axis of the reference model, the base models,
        # and the fastText model by the sum of the correlation coefficients
        # for the reference model and the fastText model.
        sorted_ref_embed2 = np.zeros_like(ref_embed[:, :min_dim])
        sorted_base_embeds2 = []
        for idx in range(len(base_embeds)):
            sorted_base_embeds2.append(
                np.zeros_like(sorted_base_embeds[idx][:, :min_dim]))
        perm_en_embed2 = np.zeros_like(perm_en_embed[:, :min_dim])
        for idx, (_, i, j) in enumerate(sorted(rij, reverse=True)):
            sorted_ref_embed2[:, idx] = sorted_ref_embed[:, i]
            perm_en_embed2[:, idx] = perm_en_embed[:, i]
            for bx, base_embed in enumerate(base_embeds):
                sorted_base_embeds2[bx][:, idx] = \
                    sorted_base_embeds[bx][:, i]
        sorted_ref_embed = sorted_ref_embed2
        sorted_base_embeds = sorted_base_embeds2
        perm_en_embed = perm_en_embed2

        # compress the correlation matrix.
        tmp_corr_matrix = np.zeros((min_dim, min_dim))
        for j in range(en_dim):
            if j in j2k:
                tmp_corr_matrix[:, j2k[j]] = corr_matrix[:, j]
        corr_matrix = tmp_corr_matrix

        # permute the compressed correlation matrix
        swap_corr_matrix = np.zeros((min_dim, min_dim))
        for _, i, j in rij:
            swap_corr_matrix[j2k[j], :] = corr_matrix[i, :]
        both_swap_corr_matrix = swap_corr_matrix.copy()
        both_swap_corr_matrix = \
            both_swap_corr_matrix[
                np.argsort(-np.diag(swap_corr_matrix)), :]
        for i in tqdm(range(min_dim)):
            argmax = i + np.argmax(both_swap_corr_matrix[i][i:])
            tmp = both_swap_corr_matrix[:, i].copy()
            both_swap_corr_matrix[:, i] = \
                both_swap_corr_matrix[:, argmax]
            both_swap_corr_matrix[:, argmax] = tmp
        both_swap_corr_matrixs.append(both_swap_corr_matrix)

        # nomalize
        normed_sorted_ref_embed = sorted_ref_embed / np.linalg.norm(
            sorted_ref_embed, axis=1, keepdims=True)
        normed_sorted_base_embeds = []
        for _, sorted_base_embed in enumerate(sorted_base_embeds):
            normed_sorted_base_embed = sorted_base_embed / np.linalg.norm(
                sorted_base_embed, axis=1, keepdims=True)
            normed_sorted_base_embeds.append(normed_sorted_base_embed)
        normed_perm_en_embed = perm_en_embed / np.linalg.norm(
            perm_en_embed, axis=1, keepdims=True)

        # save
        dump = (normed_sorted_ref_embed,
                normed_sorted_base_embeds,
                normed_perm_en_embed,
                en_id2word, en_id2base_ids, en_word2id,
                base_id2file_name, both_swap_corr_matrixs)
        with open(dump_path, 'wb') as f:
            pkl.dump(dump, f)

    # plot corr_matrix
    ref = base_list[0]
    (output_dir / 'figures/axis_corr').mkdir(parents=True, exist_ok=True)

    # image model
    assert len(both_swap_corr_matrixs[:-1]) == len(base_list[1:])
    for bx, corr_matrix in enumerate(both_swap_corr_matrixs[:-1]):
        base = base_list[bx+1]
        tgt = long_name2title(base)
        img_path = output_dir / 'figures/axis_corr/'\
            f'{pp}-axis_corr-{bases}_{ref}_{base}-{dpi}dpi.png'
        logger.info(img_path)
        save_corr_matrix_fig(corr_matrix, img_path, long_name2title(ref),
                             tgt, dim=100, pp=pp, dpi=dpi)

    # fasttext model
    tgt = 'fastText'
    corr_matrix = both_swap_corr_matrixs[-1]
    img_path = output_dir / 'figures/axis_corr/'\
        f'{pp}-axis_corr-{langs}_{emb_type}-'\
        f'{bases}_{ref}-{dpi}dpi.png'
    logger.info(img_path)
    save_corr_matrix_fig(corr_matrix, img_path, long_name2title(ref),
                         tgt, dim=100, pp=pp, dpi=dpi)


if __name__ == '__main__':
    main()
