import argparse
import pickle as pkl
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA, FastICA

from utils import check_model, get_logger


def parse_args():
    parser = argparse.ArgumentParser(description='save PCA and ICA embeddings')

    # root directory
    parser.add_argument('--root_dir', type=str, default='/working')

    # image model
    parser.add_argument("--base", type=str, default="resnet18")

    # ica
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_iter", type=int, default=10000)
    parser.add_argument("--tol", type=float, default=1e-10)

    return parser.parse_args()


def main():
    args = parse_args()

    root_dir = Path(args.root_dir)

    base = args.base
    check_model(base)

    seed = args.seed
    max_iter = args.max_iter
    tol = args.tol

    logger = get_logger()
    logger.info(f'base: {base}')
    output_dir = root_dir / 'output/image/'
    names_embeds_path = output_dir / f'{base}-raw.pkl'
    assert names_embeds_path.exists(), f'{names_embeds_path} does not exist.'

    with open(names_embeds_path, 'rb') as f:
        all_names, all_embeddings = pkl.load(f)
    logger.info(f'all_embeddings.shape: {all_embeddings.shape}')

    rng = np.random.RandomState(seed)

    # centering
    all_embeddings_ = all_embeddings - all_embeddings.mean(axis=0)

    # PCA
    pca_params = {'random_state': rng}
    logger.info(f'pca_params: {pca_params}')
    pca = PCA(**pca_params)
    pca_embed = pca.fit_transform(all_embeddings_)
    pca_embed = pca_embed / pca_embed.std(axis=0)

    # ICA
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
    R = ica.mixing_
    ica_embed = pca_embed @ R

    names_pca_ica_embeds = (all_names, all_embeddings, pca_embed, ica_embed)
    names_pca_ica_embeds_path = output_dir / f'{base}-pca_ica.pkl'

    logger.info(f'names_pca_ica_embeds_path: {names_pca_ica_embeds_path}')
    with open(names_pca_ica_embeds_path, 'wb') as f:
        pkl.dump(names_pca_ica_embeds, f)


if __name__ == '__main__':
    main()
