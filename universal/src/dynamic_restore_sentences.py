import argparse
import pickle as pkl
from collections import defaultdict
from pathlib import Path

import numpy as np
from transformers import BertTokenizer

from utils import get_logger

# The distributed BERT embeddings do not contain text from the One Billion
# Word Benchmark. Instead, they store the line numbers of the sentences in
# the following file, and this script restores the tokens and sentences in
# the same way as dynamic_save_raw_embeddings.py.
NEWS_PATH = 'data/dynamic/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/news.en-00001-of-00100'  # noqa


def parse_args():
    parser = argparse.ArgumentParser(
        description='convert text to/from line numbers')

    # root directory
    parser.add_argument('--root_dir', type=str, default='/working')

    # embedding
    parser.add_argument('--num_token', type=int, default=100000)

    # 'restore' (line numbers -> text) or 'strip' (text -> line numbers)
    parser.add_argument('--mode', type=str, default='restore',
                        choices=['restore', 'strip'])

    return parser.parse_args()


def main():
    args = parse_args()
    root_dir = Path(args.root_dir)
    logger = get_logger()

    with open(root_dir / NEWS_PATH) as f:
        lines = [line.strip() for line in f.readlines()]

    output_dir = root_dir / 'output/dynamic'
    path = output_dir / f'bert-pca-ica-{args.num_token}.pkl'
    line_id_path = output_dir / f'bert-pca-ica-{args.num_token}-line_ids.pkl'

    if args.mode == 'restore':
        with open(line_id_path, 'rb') as f:
            line_ids, dtypes, embeds, pca_embed, ica_embed = pkl.load(f)

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        max_len = tokenizer.model_max_length - 2
        token_count = defaultdict(int)
        tokens = []
        sents = []
        for i in line_ids:
            sent = lines[i]
            token_list = ['[CLS]'] + tokenizer.tokenize(sent)[:max_len] \
                + ['[SEP]']
            for token in token_list:
                tokens.append(f'{token}_{token_count[token]}')
                token_count[token] += 1
                sents.append(sent)
        # the original arrays were made before truncation to num_token,
        # so their string lengths (dtypes) are restored explicitly
        tokens = np.array(tokens[:args.num_token], dtype=dtypes[0])
        sents = np.array(sents[:args.num_token], dtype=dtypes[1])

        with open(path, 'wb') as f:
            pkl.dump((tokens, sents, embeds, pca_embed, ica_embed), f,
                     protocol=4)
        logger.info(f'saved {path}')
    else:
        with open(path, 'rb') as f:
            tokens, sents, embeds, pca_embed, ica_embed = pkl.load(f)

        line2id = {}
        for i, line in enumerate(lines):
            line2id.setdefault(line, i)
        # each sentence starts with [CLS]
        line_ids = [line2id[sent] for token, sent in zip(tokens, sents)
                    if token.startswith('[CLS]_')]
        line_ids = np.array(line_ids, dtype=np.int32)

        with open(line_id_path, 'wb') as f:
            dtypes = (tokens.dtype.str, sents.dtype.str)
            pkl.dump((line_ids, dtypes, embeds, pca_embed, ica_embed), f,
                     protocol=4)
        logger.info(f'saved {line_id_path}')


if __name__ == '__main__':
    main()
