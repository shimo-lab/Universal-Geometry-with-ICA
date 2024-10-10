import os
import sys
import time
import random
import threading
import numpy as np
from sklearn.decomposition import PCA, FastICA
import pickle
from typing import List, Tuple
import fire

DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')

def load_data(data_dir: str) -> Tuple[dict, dict, np.ndarray, np.ndarray]:
    """
    Load necessary data from the specified directory.
    
    Args:
        data_dir (str): Path to the directory containing the data files.
    
    Returns:
        Tuple containing word-to-id dict, id-to-word dict, wordcount array, and word vectors matrix.
    """
    with open(os.path.join(data_dir, 'text8_word-to-id.pkl'), 'rb') as f:
        get_id = pickle.load(f)
    with open(os.path.join(data_dir, 'text8_id-to-word.pkl'), 'rb') as f:
        get_word = pickle.load(f)
    wordcount = np.fromfile(os.path.join(data_dir, 'text8_wordcount'), dtype=np.int32)
    mat_X = np.fromfile(os.path.join(data_dir, 'text8_sgns-Win_ep100')).reshape(len(wordcount), -1)
    return get_id, get_word, wordcount, mat_X

def sampling_words(count_p: np.ndarray, seed: int = 4, alpha: float = 1, min_count: int = 10, num_row: int = 100000) -> List[int]:
    """
    Sample words based on their frequency.
    
    Args:
        count_p (np.ndarray): Array of word counts.
        seed (int): Random seed for reproducibility.
        alpha (float): Power to raise the sampling weights to.
        min_count (int): Minimum count for a word to be considered.
        num_row (int): Number of words to sample.
    
    Returns:
        List of sampled word IDs.
    """
    random.seed(seed)
    np.random.seed(seed)
    possible_ids = np.where(count_p >= min_count)[0]
    wids = random.choices(list(possible_ids), weights=(count_p[possible_ids] ** alpha), k=num_row)
    wid_set = set(wids)
    new_wid = set()
    for wid in np.flip(np.argsort(count_p)):
        if wid not in wid_set:
            new_wid.add(wid)
        if (len(wid_set) + len(new_wid)) >= 30000:
            break
    wids += list(new_wid)
    return wids

def save_results(output_dir: str, list_wordid: List[int], mat_pca1: np.ndarray, mat_pca2: np.ndarray, R_ica: np.ndarray):
    """
    Save the results of the analysis.
    
    Args:
        output_dir (str): Directory to save the output files.
        list_wordid (List[int]): List of word IDs used in the analysis.
        mat_pca1 (np.ndarray): PCA1 matrix.
        mat_pca2 (np.ndarray): PCA2 matrix.
        R_ica (np.ndarray): ICA mixing matrix.
    """
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    os.makedirs(os.path.join(output_dir, 'ica_data'), exist_ok=True)
    
    with open(os.path.join(output_dir, 'ica_data', f'wids_{timestamp}.pkl'), 'wb') as f:
        pickle.dump(list_wordid, f)
    with open(os.path.join(output_dir, 'ica_data', f'pca1_{timestamp}.pkl'), 'wb') as f:
        pickle.dump(mat_pca1, f)
    with open(os.path.join(output_dir, 'ica_data', f'pca2_{timestamp}.pkl'), 'wb') as f:
        pickle.dump(mat_pca2, f)
    with open(os.path.join(output_dir, 'ica_data', f'R_ica_{timestamp}.pkl'), 'wb') as f:
        pickle.dump(R_ica, f)

def print_progress():
    """Print a dot every second to indicate progress."""
    while getattr(threading.current_thread(), "do_run", True):
        sys.stdout.write('.')
        sys.stdout.flush()
        time.sleep(1)

def run_ica(mat_X: np.ndarray, list_wordid: List[int], n_components: int = None, max_iter: int = 10000, tol: float = 1e-10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run ICA on the input matrix with a simple progress indicator and improved debugging.
    
    Args:
        mat_X (np.ndarray): Input matrix.
        list_wordid (List[int]): List of word IDs to use.
        n_components (int): Number of ICA components.
        max_iter (int): Maximum number of iterations for ICA.
        tol (float): Tolerance for ICA convergence.
    
    Returns:
        Tuple containing PCA1, PCA2, and ICA matrices.
    """
    mat_S = mat_X[list_wordid]
    bar_mat = np.mean(mat_S, axis=0)
    mat_c = mat_S - bar_mat  # centering
    pca = PCA(whiten=False)
    mat_pca1 = pca.fit_transform(mat_c)
    mat_pca2 = mat_pca1 / np.std(mat_pca1, axis=0)  # whitening
    
    print(f"Starting ICA with max_iter={max_iter} and tol={tol}")
    print("Progress: ", end='', flush=True)
    
    progress_thread = threading.Thread(target=print_progress)
    progress_thread.daemon = True
    progress_thread.start()
    
    start_time = time.time()
    ica = FastICA(n_components=n_components, whiten=False, max_iter=max_iter, tol=tol)
    mat_ica2 = ica.fit_transform(mat_pca2)
    end_time = time.time()
    
    progress_thread.do_run = False
    progress_thread.join()
    
    print(f"\nICA completed in {end_time - start_time:.2f} seconds")
    print(f"Number of iterations: {ica.n_iter_}")
    print(f"Converged: {ica.n_iter_ < max_iter}")

    return mat_pca1, mat_pca2, ica.mixing_

def run_ica_main(
    data_dir: str,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    seed: int = 4,
    alpha: float = 1,
    min_count: int = 10,
    num_row: int = 100000,
    n_components: int = None,
    max_iter: int = 10000,
    tol: float = 1e-10
):
    """
    Run ICA analysis on word embeddings with progress messages and improved debugging.
    
    Args:
        data_dir (str): Directory containing input data files.
        output_dir (str): Directory to save output files.
        seed (int): Random seed for reproducibility.
        alpha (float): Power to raise the sampling weights to.
        min_count (int): Minimum count for a word to be considered.
        num_row (int): Number of words to sample.
        n_components (int): Number of ICA components.
        max_iter (int): Maximum number of iterations for ICA.
        tol (float): Tolerance for ICA convergence.
    """
    print(f"Starting ICA analysis with the following parameters:")
    print(f"data_dir: {data_dir}")
    print(f"output_dir: {output_dir}")
    print(f"seed: {seed}")
    print(f"alpha: {alpha}")
    print(f"min_count: {min_count}")
    print(f"num_row: {num_row}")
    print(f"n_components: {n_components}")
    print(f"max_iter: {max_iter}")
    print(f"tol: {tol}")

    print("Loading data...")
    get_id, get_word, wordcount, mat_X = load_data(data_dir)
    
    print("Sampling words...")
    list_wordid = sampling_words(wordcount, seed, alpha, min_count, num_row)
    
    mat_pca1, mat_pca2, R_ica = run_ica(mat_X, list_wordid, n_components, max_iter, tol)
    
    print("Saving results...")
    save_results(output_dir, list_wordid, mat_pca1, mat_pca2, R_ica)
    
    print("ICA analysis completed and results saved.")

if __name__ == "__main__":
    fire.Fire(run_ica_main)

# Example usage:
# python ica.py --data_dir ../data/text8_sgns --max_iter 10