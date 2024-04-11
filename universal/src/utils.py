import logging

import inflect
import matplotlib.font_manager as fm
import numpy as np
import scipy
from tqdm import tqdm


def get_logger(log_file=None, log_level=logging.INFO, stream=True):

    logger = logging.getLogger(__name__)
    handlers = []
    if stream:
        stream_handler = logging.StreamHandler()
        handlers.append(stream_handler)

    if log_file is not None:
        file_handler = logging.FileHandler(str(log_file), 'w')
        handlers.append(file_handler)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    logger.setLevel(log_level)

    return logger


def get_lang_name(lang):
    if lang == 'en':
        return 'English'
    elif lang == 'es':
        return 'Spanish'
    elif lang == 'fr':
        return 'French'
    elif lang == 'de':
        return 'German'
    elif lang == 'it':
        return 'Italian'
    elif lang == 'ru':
        return 'Russian'
    elif lang == 'ja':
        return 'Japanese'
    elif lang == 'zh':
        return 'Chinese'
    elif lang == 'ar':
        return 'Arabic'
    elif lang == 'ko':
        return 'Korean'
    elif lang == 'hi':
        return 'Hindi'
    else:
        raise ValueError(f'Unknown lang: {lang}')


def pos_direct(vecs):
    vecs = vecs * np.sign(scipy.stats.skew(vecs, axis=0))
    return vecs


def get_top_words_and_ids(src_id2word, tgt_id2words,
                          src_word2id, tgt_word2ids,
                          normed_sorted_src_embed, normed_sorted_tgt_embeds,
                          sw2lang2tw, tgt_list, axis_num):
    p = inflect.engine()
    n = len(src_id2word)
    src_words = []
    tgt_words_list = [[] for _ in range(len(tgt_list))]
    for idx in tqdm(list(range(axis_num))):
        src_axis = normed_sorted_src_embed[:, idx]
        tgt_axises = []
        for tx, _ in enumerate(tgt_list):
            tgt_axis = normed_sorted_tgt_embeds[tx][:, idx]
            tgt_axises.append(tgt_axis)
        assert len(src_axis) == n and len(tgt_axises[0]) == n

        wids = np.argsort(-src_axis)
        src_tmp = []
        tgt_tmps = [[] for _ in range(len(tgt_list))]
        for wi in wids:
            src_word = src_id2word[wi]
            if src_word in src_tmp or src_word in src_words:
                continue

            # plural check
            flag = p.singular_noun(src_word)

            if flag is not False:
                src_singluar_word = p.singular_noun(src_word)
                if src_singluar_word in src_tmp or \
                        src_singluar_word in src_words:
                    continue
            else:
                src_plural_word = p.plural_noun(src_word)
                if src_plural_word in src_tmp or \
                        src_plural_word in src_words:
                    continue

            tws_flag = True
            tws_list = []
            for tx, tgt in enumerate(tgt_list):
                if src_id2word[wi] not in sw2lang2tw:
                    continue
                tws = sw2lang2tw[src_id2word[wi]][tgt]
                if len(tws) == 0:
                    tws_flag = False
                    break
                tws_list.append(tws)

            if not tws_flag or len(tws_list) != len(tgt_list):
                continue

            max_tgt_words = [None for _ in range(len(tgt_list))]
            max_sims = [-10**10] * len(tgt_list)
            for tx, tws in enumerate(tws_list):
                for tgt_word in tws:
                    if tgt_word in tgt_tmps[tx] or \
                            tgt_word in tgt_words_list[tx]:
                        continue
                    if max_sims[tx] < tgt_axises[tx][
                            tgt_word2ids[tx][tgt_word]]:

                        max_sims[tx] = \
                            tgt_axises[tx][tgt_word2ids[tx][tgt_word]]
                        max_tgt_words[tx] = tgt_word

            max_tgt_word_flag = True
            for max_tgt_word in max_tgt_words:
                if max_tgt_word is None:
                    max_tgt_word_flag = False
                    break

            if not max_tgt_word_flag:
                continue

            src_tmp.append(src_word)
            for tx, max_tgt_word in enumerate(max_tgt_words):
                tgt_tmps[tx].append(max_tgt_word)

            if len(src_tmp) == 5:
                break

        assert len(src_tmp) == 5
        for tgt_tmp in tgt_tmps:
            assert len(tgt_tmp) == 5

        src_words += src_tmp
        for tx, tgt_tmp in enumerate(tgt_tmps):
            tgt_words_list[tx] += tgt_tmp

        if len(src_words) == 5 * axis_num:
            break

    src_word_ids = [src_word2id[src_word] for src_word in src_words]
    tgt_word_ids_list = []
    for tx, tgt_words in enumerate(tgt_words_list):
        tgt_word_ids = [tgt_word2ids[tx][tgt_word] for tgt_word in tgt_words]
        tgt_word_ids_list.append(tgt_word_ids)

    return src_words, src_word_ids, tgt_words_list, tgt_word_ids_list


def get_font_prop(lang, root_dir):

    if lang == 'ja' or lang == 'zh':
        font_path = root_dir /\
            'data/crosslingual/fonts/NotoSansCJKjp-Regular.otf'
        font_prop = fm.FontProperties(fname=font_path, size=13)
    elif lang == 'hi':
        font_path = root_dir /\
            'data/crosslingual/fonts/'\
            'NotoSansDevanagari-VariableFont_wdth,wght.ttf'
        font_prop = fm.FontProperties(fname=font_path, size=13)
    else:
        font_prop = fm.FontProperties(size=13)
    return font_prop


def short_name2long_name(short_name):
    if short_name == 'resnet':
        return 'resnet18'
    elif short_name == 'vit':
        return 'vit_base_patch32_224_clip_laion2b'
    elif short_name == 'swin':
        return 'swin_small_patch4_window7_224'
    elif short_name == 'regnety':
        return 'regnety_002'
    elif short_name == 'resmlp':
        return 'resmlp_12_224'


def long_name2title(long_name):
    if long_name == 'resnet18':
        return 'ResNet-18'
    elif long_name == 'vit_base_patch32_224_clip_laion2b':
        return 'ViT-Base'
    elif long_name == 'swin_small_patch4_window7_224':
        return 'Swin-S'
    elif long_name == 'regnety_002':
        return 'RegNetY-200MF'
    elif long_name == 'resmlp_12_224':
        return 'ResMLP-12'


def check_model(base):
    # we use 300 <= dim <= 1000 models
    # resnet18: 512
    # vit_base_patch32_224_clip_laion2b: 768
    # swin_small_patch4_window7_224: 768
    # regnety_002: 368
    # resmlp_12_224:384
    assert base in ('resnet18', 'vit_base_patch32_224_clip_laion2b',
                    'swin_small_patch4_window7_224', 'regnety_002',
                    'resmlp_12_224')
