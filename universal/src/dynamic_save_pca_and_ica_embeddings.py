import argparse
import pickle as pkl
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA, FastICA

from utils import get_logger


def parse_args():
    parser = argparse.ArgumentParser(description='alignment task evaluation')

    # root directory
    parser.add_argument('--root_dir', type=str, default='/working')

    # embedding
    parser.add_argument('--num_token', type=int, default=100000)

    # ica
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_iter', type=int, default=10000)
    parser.add_argument('--tol', type=float, default=1e-10)

    return parser.parse_args()


def main():

    args = parse_args()

    root_dir = Path(args.root_dir)

    num_token = args.num_token

    seed = args.seed
    max_iter = args.max_iter
    tol = args.tol

    logger = get_logger()

    data_path = root_dir / f'output/dynamic/bert-raw-{num_token}.pkl'

    with open(data_path, 'rb') as f:
        tokens_sents_embeds = pkl.load(f)
    all_tokens, all_sents, all_embeddings = tokens_sents_embeds

    # centering
    all_embeddings_ = all_embeddings - all_embeddings.mean(axis=0)

    rng = np.random.RandomState(seed)
    pca_params = {'random_state': rng}
    logger.info(f'pca_params: {pca_params}')
    pca = PCA(**pca_params)
    pca_embed = pca.fit_transform(all_embeddings_)
    pca_embed = pca_embed / pca_embed.std(axis=0)

    ica_params = {
        'n_components': None,
        'random_state': rng,
        'max_iter': max_iter,
        'tol': tol,
        'whiten': False
    }
    logger.info(f'ica_params: {ica_params}')
    ica = FastICA(**ica_params)
    ica.fit(pca_embed)
    R = ica.components_.T
    ica_embed = pca_embed @ R

    tokens_sents_embeds = (all_tokens, all_sents,
                           all_embeddings, pca_embed, ica_embed)
    output_dir = root_dir / 'output/dynamic'
    output_path = output_dir / f'bert-pca-ica-{num_token}.pkl'
    with open(output_path, 'wb') as f:
        pkl.dump(tokens_sents_embeds, f)


if __name__ == '__main__':
    main()
