import argparse
import io
import pickle as pkl
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA, FastICA
from wordfreq import word_frequency

from utils import get_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Save PCA and ICA embeddings')

    # root directory
    parser.add_argument('--root_dir', type=str, default='/working')

    # language
    parser.add_argument('--lang', type=str, default='en')
    parser.add_argument('--src', type=str, default='en')
    parser.add_argument('--tgts', type=str, default='es-ru-ar-hi-zh-ja')

    # embedding
    parser.add_argument('--emb_type', type=str, default='cc')
    parser.add_argument('--vocab_size', type=int, default=50000)
    parser.add_argument('--dim', type=int, default=300)

    # ica
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_iter', type=int, default=10000)
    parser.add_argument('--tol', type=float, default=1e-10)

    # radom rotation flag
    parser.add_argument('--random_rotation', action='store_true')
    return parser.parse_args()


def load_txt_embeddings(args, logger):

    def get_cc_words(emb_path, dim):

        word_vec = []
        table_words = set()
        with io.open(emb_path, 'r', encoding='utf-8',
                     newline='\n', errors='ignore') as f:
            for i, line in enumerate(f):
                if i == 0:
                    split = line.split()
                    assert len(split) == 2
                    assert dim == int(split[1])
                else:
                    tokens = line.rstrip().split(' ')
                    word = tokens[0]
                    word = word.lower()
                    vec = list(map(float, tokens[1:]))
                    word_vec.append((word, vec))
                    table_words.add(word)
        return word_vec, table_words

    def get_muse_words(emb_path, dim):

        word_vec = []
        table_words = set()
        with io.open(emb_path, 'r', encoding='utf-8',
                     newline='\n', errors='ignore') as f:
            for i, line in enumerate(f):
                if i == 0:
                    split = line.split()
                    assert len(split) == 2
                    assert dim == int(split[1])
                else:
                    word, vect = line.rstrip().split(' ', 1)
                    word = word.lower()
                    vect = np.fromstring(vect, sep=' ')

                    # avoid to have null embeddings
                    if np.linalg.norm(vect) == 0:
                        vect[0] = 0.01
                    if word not in table_words:
                        if not vect.shape == (dim,):
                            raise RuntimeError(
                                f'Invalid dimension ({vect.shape[0]}) '
                                f'word {word} in line {i}.')
                        assert vect.shape == (dim,) and i > 0
                        word_vec.append((word, vect))
                        table_words.add(word)
        return word_vec, table_words

    def get_dic_words(lang, src, tgts, root_dir):
        tgt_list = tgts.split('-')
        assert lang == src or lang in tgt_list

        lang2words = defaultdict(set)
        for tgt in tgt_list:
            for num in ['0-5000', '5000-6500']:
                for tr in [f'{src}-{tgt}', f'{tgt}-{src}']:
                    src2tgt_dict_path = root_dir /\
                        f'data/crosslingual/MUSE/dictionaries/{tr}.{num}.txt'
                    with open(src2tgt_dict_path, 'r') as f:
                        pairs = [line.strip().split()
                                 for line in f.readlines()]
                    for s, t in pairs:
                        s = s.lower()
                        t = t.lower()
                        lang2words[src].add(s)
                        lang2words[tgt].add(t)
        return lang2words[lang]

    # load pretrained embeddings
    root_dir = Path(args.root_dir)
    lang = args.lang
    src = args.src
    tgts = args.tgts
    emb_type = args.emb_type
    emb_path = args.emb_path
    vocab_size = args.vocab_size
    dim = args.dim
    assert emb_path.exists()

    logger.info(f'Loading embeddings from {emb_path}.')
    if emb_type == 'cc':
        word_vec, table_words = get_cc_words(emb_path, dim)
    else:
        word_vec, table_words = get_muse_words(emb_path, dim)

    dic_words = get_dic_words(lang, src, tgts, root_dir)
    valid_words = set()
    logger.info(f'Found {len(dic_words)} dictionary words.')
    for w in dic_words:
        if w in table_words:
            valid_words.add(w)
    assert len(valid_words) <= vocab_size
    logger.info(f'Found {len(valid_words)} valid words in dictionary.')

    freq_path = root_dir /\
        f'output/crosslingual/word_freq/{emb_path.stem}_wordfreq.pkl'
    (root_dir / 'output/crosslingual/word_freq'
     ).mkdir(parents=True, exist_ok=True)

    if freq_path.exists():
        with open(freq_path, 'rb') as f:
            words_freqs = pkl.load(f)
        words, _ = words_freqs
    else:
        freq_word_list = []
        checked_words = set()
        for word in table_words:
            word = word.lower()
            freq = word_frequency(word, lang)
            if freq > 0 and word not in checked_words:
                checked_words.add(word)
                freq_word_list.append((freq, word))
        freq_word_list.sort(reverse=True)
        freqs = []
        words = []
        for freq, word in freq_word_list:
            freqs.append(freq)
            words.append(word)
        words_freqs = (words, freqs)
        with open(freq_path, 'wb') as f:
            pkl.dump(words_freqs, f)

    logger.info('choose words from frequency list.')
    for w in words:
        if w in table_words:
            valid_words.add(w)
        if len(valid_words) == vocab_size:
            break

    word2id = {}
    vectors = []

    logger.info('Building vocabulary.')
    for word, vec in word_vec:
        if word in valid_words and word not in word2id:
            word2id[word] = len(word2id)
            vectors.append(vec)
            if len(word2id) == vocab_size:
                break
    assert len(word2id) == vocab_size
    logger.info(f'Loaded {len(vectors)} pre-trained word embeddings.')

    # compute new vocabulary / embeddings
    id2word = {v: k for k, v in word2id.items()}
    embeddings = np.array(vectors)

    return id2word, word2id, embeddings


def main():
    args = parse_args()
    root_dir = Path(args.root_dir)
    lang = args.lang
    # We limit the vocabulary. The default value is 50,000.
    # The way of selecting words depends on the source and target languages.
    src = args.src
    tgts = args.tgts
    emb_type = args.emb_type
    vocab_size = args.vocab_size
    assert emb_type in ['cc', 'muse']
    random_rotation = args.random_rotation
    logger = get_logger()

    np.random.seed(args.seed)
    rng = np.random.RandomState(args.seed)

    if emb_type == 'cc':
        args.emb_path = root_dir /\
            f'data/crosslingual/157langs/vectors/cc.{lang}.300.vec'
    elif emb_type == 'muse':
        args.emb_path = \
            root_dir / f'data/crosslingual/MUSE/vectors/wiki.multi.{lang}.vec'
    else:
        raise ValueError(f'emb_type {emb_type} is not supported.')

    rand_str = 'randrot.' if random_rotation else ''
    emb_name = f'{rand_str}{Path(args.emb_path).stem}'
    output_dir = root_dir / f'output/crosslingual/{src}-{tgts}'
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f'Loading {vocab_size} embeddings.')
    id2word, word2id, embed = load_txt_embeddings(args, logger)
    logger.info('finish loading embeddings!')
    logger.info(f'embed.shape {embed.shape}')

    if random_rotation:
        U = np.random.normal(size=(args.dim, args.dim), scale=1/args.dim)
        V = np.random.normal(size=(args.dim, args.dim), scale=1/args.dim)
        Sigma = np.diag(np.random.exponential(size=args.dim))
        randmat = np.dot(U, np.dot(Sigma, V.T))
        embed = np.dot(embed, randmat)

    # centering
    embed_ = embed - embed.mean(axis=0)

    # PCA
    pca_params = {'random_state': rng}
    logger.info(f'pca_params: {pca_params}')
    pca = PCA(random_state=rng)
    pca_embed = pca.fit_transform(embed_)
    pca_embed = pca_embed / pca_embed.std(axis=0)

    # ICA
    ica_params = {
        'n_components': None,
        'random_state': rng,
        'max_iter': args.max_iter,
        'tol': args.tol,
        'whiten': False
    }
    logger.info(f'ica_params: {ica_params}')
    ica = FastICA(**ica_params)
    ica.fit(pca_embed)
    R = ica.components_.T
    ica_embed = pca_embed @ R

    # save embeddings
    logger.info('save embeddings...')
    dic_and_emb = (word2id, id2word, embed, pca_embed, ica_embed)
    emb_path = output_dir / f'{emb_name}_dic_and_emb.pkl'
    with open(emb_path, 'wb') as f:
        pkl.dump(dic_and_emb, f)


if __name__ == '__main__':
    main()
