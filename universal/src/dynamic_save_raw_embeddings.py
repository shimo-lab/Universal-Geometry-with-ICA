import argparse
import pickle as pkl
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

from utils import get_logger


def parse_args():
    parser = argparse.ArgumentParser(description='alignment task evaluation')

    # root directory
    parser.add_argument('--root_dir', type=str, default='/working')

    # embedding
    parser.add_argument('--num_token', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=512)


def main():

    args = parse_args()

    root_dir = Path(args.root_dir)

    num_token = args.num_token
    batch_size = args.batch_size

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = get_logger()
    logger.info(f'device: {device}')
    logger.info('tokenizer loading...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    logger.info('model loading...')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.to(device)
    model.eval()

    def bert_encode(x, attention_mask):
        with torch.no_grad():
            result = model(x.to(device),
                           attention_mask=attention_mask.to(device))
        embeddings = result.last_hidden_state
        return embeddings

    def truncate(tokens):
        if len(tokens) > tokenizer.model_max_length - 2:
            tokens = tokens[0:(tokenizer.model_max_length - 2)]
        return tokens

    def padding(arr, pad_token, dtype=torch.long):
        lens = torch.LongTensor([len(a) for a in arr])
        max_len = lens.max().item()
        padded = torch.ones(len(arr), max_len, dtype=dtype) * pad_token
        mask = torch.zeros(len(arr), max_len, dtype=torch.long)
        for i, a in enumerate(arr):
            padded[i, :lens[i]] = torch.tensor(a, dtype=dtype)
            mask[i, :lens[i]] = 1
        return padded, lens, mask

    def prepare_text_sequences(arr, tokenize, numericalize, pad="[PAD]"):

        tokens = [["[CLS]"]+truncate(tokenize(a))+["[SEP]"] for a in arr]
        arr = [numericalize(a) for a in tokens]

        pad_token = numericalize([pad])[0]

        padded, lens, mask = padding(arr, pad_token, dtype=torch.long)

        return padded, lens, mask, tokens

    def get_bert_embedding(all_sens):

        padded_sens, lens, mask, token_lists = prepare_text_sequences(
            all_sens, tokenizer.tokenize, tokenizer.convert_tokens_to_ids)

        with torch.no_grad():
            torch_embedding_lists = \
                bert_encode(padded_sens, attention_mask=mask)

        numpy_embedding_lists = torch_embedding_lists.cpu().numpy()
        numpy_lens = lens.cpu().numpy()

        embedding_lists = []
        for embedding_list, len_, token_list in zip(
                numpy_embedding_lists, numpy_lens, token_lists):
            embedding_list = embedding_list[:len_]
            assert len(embedding_list) == len(token_list)
            embedding_lists.append(embedding_list)

        return embedding_lists, token_lists

    def loadFile(data_path):
        with open(data_path) as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
        return lines

    data_path = root_dir / 'data/dynamic/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/news.en-00001-of-00100'  # noqa
    sents = loadFile(data_path)

    logger.info(f'batch_size: {batch_size}')
    token_count = 0
    all_embeddings = []
    all_tokens = []
    all_sents = []
    for batch_start in tqdm(range(0, len(sents), batch_size)):
        batch_sents = sents[batch_start:batch_start+batch_size]
        try:
            batch_embeddings, batch_tokens = get_bert_embedding(batch_sents)
            all_embeddings += batch_embeddings
            all_tokens += batch_tokens
            assert len(batch_tokens) == len(batch_sents)
            for token_list, sent in zip(batch_tokens, batch_sents):
                token_count += len(token_list)
                tmp = [sent] * len(token_list)
                all_sents.append(tmp)
        except RuntimeError as e:
            logger.error(f'{e}, skip this batch')
            continue
        if token_count > num_token:
            logger.info(f'break at {batch_start}')
            break

    token_count = defaultdict(int)
    embeddings = []
    tokens = []
    sents = []
    logger.info('making org_matrix_and_words...')
    for embeds, token_list, sent_list in tqdm(
            zip(all_embeddings, all_tokens, all_sents)):
        for embed, token, sent in zip(embeds, token_list, sent_list):
            c = token_count[token]
            token_c = f'{token}_{c}'
            embeddings.append(embed)
            sents.append(sent)
            tokens.append(token_c)
            token_count[token] += 1

    all_embeddings = np.array(embeddings)[:num_token]
    all_tokens = np.array(tokens)[:num_token]
    all_sents = np.array(sents)[:num_token]

    logger.info(f'embeddings.shape: {all_embeddings.shape}')
    logger.info(f'number of tokens: {len(all_tokens)}')
    logger.info(f'number of sents: {len(all_sents)}')

    tokens_sents_embeds = (all_tokens, all_sents, all_embeddings)
    output_dir = root_dir / 'output/dynamic/'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'bert-raw-{num_token}.pkl'
    with open(output_path, 'wb') as f:
        pkl.dump(tokens_sents_embeds, f)


if __name__ == '__main__':
    main()
