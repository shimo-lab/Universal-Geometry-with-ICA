import argparse
import io
import os
import pickle as pkl
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from tqdm import tqdm

from utils import get_logger, pos_direct

try:
    import faiss
    FAISS_AVAILABLE = True
    if not hasattr(faiss, 'StandardGpuResources'):
        sys.stderr.write('Impossible to import Faiss-GPU. '
                         'Switching to FAISS-CPU, '
                         'this will be slower.\n\n')
except ImportError:
    sys.stderr.write('Impossible to import Faiss library!! '
                     'this will be significantly slower.\n\n')
    FAISS_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(description='alignment task evaluation')

    # root directory
    parser.add_argument('--root_dir', type=str, default='/working')

    # language
    parser.add_argument('--src', type=str, default='en')
    parser.add_argument('--tgts', type=str, default='es-fr-de-it-ru')

    return parser.parse_args()


def load_dictionary(path, word2id1, word2id2, logger):
    '''
    Return a torch tensor of size (n, 2) where n is the size of the
    loader dictionary, and sort it by source word frequency.
    '''
    assert os.path.isfile(path)

    pairs = []
    not_found = 0
    not_found1 = 0
    not_found2 = 0

    with io.open(path, 'r', encoding='utf-8') as f:
        for index, line in enumerate(f):
            assert line == line.lower()
            parts = line.rstrip().split()
            if len(parts) < 2:
                logger.warning(f'Could not parse line {line} ({index})')
                continue
            word1, word2 = parts
            if word1 in word2id1 and word2 in word2id2:
                pairs.append((word1, word2))
            else:
                not_found += 1
                not_found1 += int(word1 not in word2id1)
                not_found2 += int(word2 not in word2id2)

    logger.info(f'Found {len(pairs)} pairs of words in the dictionary '
                f'({len(set([x for x, _ in pairs]))} unique). '
                f'{not_found} other pairs contained at least one unknown word '
                f'({not_found1} in lang1, {not_found2} in lang2)')

    # sort the dictionary by source word frequencies
    pairs = sorted(pairs, key=lambda x: word2id1[x[0]])
    dico = np.zeros((len(pairs), 2))
    for i, (word1, word2) in enumerate(pairs):
        dico[i, 0] = word2id1[word1]
        dico[i, 1] = word2id2[word2]

    return dico.astype(int)


def get_nn_avg_dist(emb, query, knn):
    '''
    Compute the average distance of the `knn` nearest neighbors
    for a given set of embeddings and queries.
    Use Faiss if available.
    '''
    if FAISS_AVAILABLE:
        emb_ = emb.astype(np.float32)
        query_ = query.astype(np.float32)
        if hasattr(faiss, 'StandardGpuResources'):
            # gpu mode
            res = faiss.StandardGpuResources()
            config = faiss.GpuIndexFlatConfig()
            config.device = 0
            index = faiss.GpuIndexFlatIP(res, emb_.shape[1], config)
        else:
            # cpu mode
            index = faiss.IndexFlatIP(emb_.shape[1])
        index.add(emb_)
        distances, _ = index.search(query_, knn)
        return distances.mean(1)
    else:
        bs = 1024
        all_distances = []
        emb = emb.transpose()
        for i in tqdm(list(range(0, query.shape[0], bs))):
            distances = query[i:i + bs] @ emb
            best_indices = np.argpartition(distances, knn, axis=1)[:, :knn]
            best_distances = []
            for j in range(len(best_indices)):
                best_distances.append(np.mean(distances[j][best_indices[j]]))
            all_distances += best_distances
        assert len(all_distances) == len(query)
        return np.array(all_distances)


def get_word_translation_accuracy(word2id1, emb1,
                                  word2id2, emb2,
                                  method, path, logger):
    '''
    Given source and target word embeddings, and a dictionary,
    evaluate the translation accuracy using the accuracy@k.
    '''
    dico = load_dictionary(path, word2id1, word2id2, logger)

    assert dico[:, 0].max() < emb1.shape[0]
    assert dico[:, 1].max() < emb2.shape[0]

    # nearest neighbors
    if method == 'nn':
        query = emb1[dico[:, 0]]
        scores = query @ emb2.T

    # contextual dissimilarity measure
    elif method.startswith('csls_knn_'):
        # average distances to k nearest neighbors
        knn = method[len('csls_knn_'):]
        assert knn.isdigit()
        knn = int(knn)
        average_dist1 = get_nn_avg_dist(emb2, emb1, knn)
        average_dist2 = get_nn_avg_dist(emb1, emb2, knn)

        # queries / scores
        query = emb1[dico[:, 0]]
        scores = query @ emb2.T
        scores = 2 * scores
        scores -= average_dist1[dico[:, 0]][:, np.newaxis]
        scores -= average_dist2[np.newaxis, :]

    else:
        raise Exception(f'Unknown method: {method}')

    results = []
    top_matches = np.argpartition(-scores, 10, axis=1)[:, :10]
    for i in range(len(top_matches)):
        top_matches[i] = top_matches[i][np.argsort(-scores[i][top_matches[i]])]

    for k in [1, 5, 10]:
        top_k_matches = top_matches[:, :k]
        _matching = np.sum(top_k_matches == dico[:, 1][:, np.newaxis], axis=1)
        # allow for multiple possible translations
        matching = {}
        for i, src_id in enumerate(dico[:, 0]):
            matching[src_id] = min(matching.get(src_id, 0) + _matching[i], 1)
        # evaluate accuracy@k
        accuracy_at_k = 100 * np.mean(list(matching.values()))
        logger.info(
            f'{len(matching)} source words - {method} - '
            f'Accuracy at k = {k}: {accuracy_at_k:.2f}')
        results.append((k, accuracy_at_k))

    return results


def main():
    args = parse_args()
    root_dir = Path(args.root_dir)

    src = args.src
    tgts = args.tgts
    langs = f'{src}-{tgts}'

    output_dir = root_dir / 'output/crosslingual/alignment_task'
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger()
    logger.info(f'args: {args}')

    data = []
    for emb_type in ('muse', 'cc'):
        logger.info(f'=============== emb_type: {emb_type} ===============')

        tgt_list = tgts.split('-')
        for tgt in tgt_list:

            logger.info(
                f'=============== src: {src}, tgt: {tgt} ===============')

            train_dict_path = root_dir /\
                f'data/crosslingual/MUSE/dictionaries/{src}-{tgt}.0-5000.txt'
            with open(train_dict_path, 'r') as f:
                pairs = [line.strip().split() for line in f.readlines()]

            for rot_type in ['original', 'randrot']:

                rot_str = '' if rot_type == 'original' else 'randrot.'

                if emb_type == 'cc':
                    src_dump_path = root_dir /\
                        f'output/crosslingual/{langs}/'\
                        f'{rot_str}cc.{src}.300_dic_and_emb.pkl'
                    tgt_dump_path = root_dir /\
                        f'output/crosslingual/{langs}/'\
                        f'{rot_str}cc.{tgt}.300_dic_and_emb.pkl'
                else:
                    src_dump_path = root_dir /\
                        f'output/crosslingual/{langs}/'\
                        f'{rot_str}wiki.multi.{src}_dic_and_emb.pkl'
                    tgt_dump_path = root_dir /\
                        f'output/crosslingual/{langs}/'\
                        f'{rot_str}wiki.multi.{tgt}_dic_and_emb.pkl'

                with open(src_dump_path, 'rb') as f:
                    src_word2id, src_id2word, src_raw_embed, \
                        src_pca_embed, src_ica_embed = pkl.load(f)

                with open(tgt_dump_path, 'rb') as f:
                    tgt_word2id, tgt_id2word, tgt_raw_embed, \
                        tgt_pca_embed, tgt_ica_embed = pkl.load(f)

                for method_type in [
                        'Procrustes', 'Least Square',
                        'ICA-Permutation', 'PCA-Permutation']:
                    logger.info(
                        f'rot_type: {rot_type} - '
                        f'method_type: {method_type}')

                    if 'Permutation' not in method_type:
                        src_embed = src_raw_embed
                        tgt_embed = tgt_raw_embed
                    elif 'ICA' in method_type:
                        src_embed = pos_direct(src_ica_embed)
                        tgt_embed = pos_direct(tgt_ica_embed)
                    elif 'PCA' in method_type:
                        src_embed = pos_direct(src_pca_embed)
                        tgt_embed = pos_direct(tgt_pca_embed)
                    _, dim = src_embed.shape

                    if 'Permutation' in method_type:
                        src_vs = []
                        tgt_vs = []
                        for s, t in pairs:
                            if s in src_word2id and t in tgt_word2id:
                                si = src_word2id[s]
                                ti = tgt_word2id[t]
                                sv = src_embed[si]
                                tv = tgt_embed[ti]

                                # nomalize
                                sv = sv / np.linalg.norm(sv)
                                tv = tv / np.linalg.norm(tv)

                                src_vs.append(sv)
                                tgt_vs.append(tv)

                        src_vs = np.array(src_vs)
                        tgt_vs = np.array(tgt_vs)

                        logger.info(
                            f'use pairs for swap: {len(src_vs)}/{len(pairs)} -'
                            f' {100*len(src_vs)/len(pairs):.2f}%')

                        cands = []
                        logger.info('start calculating corr_matrix...')
                        for i in tqdm(range(dim)):
                            ax1 = src_vs[:, i]
                            for j in range(dim):
                                ax2 = tgt_vs[:, j]
                                r = pearsonr(ax1, ax2)[0]
                                cands.append((r, i, j))
                        logger.info('finish calculating corr_matrix!')
                        cands.sort(reverse=True)

                        used_i = set()
                        used_j = set()

                        corr_W = np.zeros((dim, dim))
                        for r, i, j in cands:
                            if i in used_i or j in used_j:
                                continue
                            used_i.add(i)
                            used_j.add(j)
                            corr_W[i, j] = 1

                        src_embed = src_embed @ corr_W

                    else:
                        src_embed = src_embed - \
                            src_embed.mean(axis=0, keepdims=True)
                        tgt_embed = tgt_embed - \
                            tgt_embed.mean(axis=0, keepdims=True)
                        src_embed = src_embed / np.linalg.norm(
                            src_embed, axis=1, keepdims=True)
                        tgt_embed = tgt_embed / np.linalg.norm(
                            tgt_embed, axis=1, keepdims=True)

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
                            f'use pairs for proc: {len(src_vs)}/{len(pairs)} -'
                            f' {100*len(src_vs)/len(pairs):.2f}%')

                        if method_type == 'Least Square':
                            u, s, v_T = np.linalg.svd(
                                src_vs, full_matrices=False)
                            v = v_T.T
                            s_inv = np.diag(1 / s)
                            W = np.dot(np.dot(v, np.dot(s_inv, u.T)), tgt_vs)
                        else:
                            u, s, v_T = np.linalg.svd(np.dot(tgt_vs.T, src_vs))
                            v = v_T.T
                            W = np.dot(v, u.T)
                        src_embed = np.dot(src_embed, W)

                    val_dict_path = root_dir /\
                        'data/crosslingual/MUSE/dictionaries/'\
                        f'{src}-{tgt}.5000-6500.txt'
                    logger.info(f'eval data path: {val_dict_path}')

                    for method in ['csls_knn_10']:
                        results = get_word_translation_accuracy(
                            src_word2id, src_embed,
                            tgt_word2id, tgt_embed,
                            method=method,
                            path=val_dict_path,
                            logger=logger
                        )
                        for k, accuracy in results:
                            data.append(
                                (emb_type,
                                 src, tgt, rot_type, method_type,
                                 method, k, accuracy))

        df = pd.DataFrame(data, columns=[
            'cc_or_muse', 'src', 'tgt', 'rot_type', 'method_type',
            'metric', 'k', 'accuracy'])
        df.to_csv(output_dir / f'{src}-{tgts}.csv', index=False)


if __name__ == '__main__':
    main()
