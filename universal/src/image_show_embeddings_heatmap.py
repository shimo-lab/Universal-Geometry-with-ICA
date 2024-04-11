import argparse
import pickle as pkl
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from gensim.parsing.preprocessing import STOPWORDS
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from PIL import Image
from tqdm import tqdm

from utils import get_logger, long_name2title, short_name2long_name


def parse_args():
    parser = argparse.ArgumentParser(description='show embeddings heatmap')

    # root directory
    parser.add_argument('--root_dir', type=str, default='/working')

    # language
    parser.add_argument('--langs', type=str, default='en-es-ru-ar-hi-zh-ja')

    # embedding
    parser.add_argument('--emb_type', type=str, default='cc')
    parser.add_argument('--pca', action='store_true')

    # image model
    parser.add_argument(
        "--bases", type=str, default="vit-resmlp-swin-resnet-regnety")

    # figure
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=150)

    return parser.parse_args()


def sub_figure(name, embed, wids, ax, upper=False, cb_ax=None, id2word=None,
               embed_wids=None, images_per_label=1):

    if embed_wids is None:
        embed_wids = wids
    picked_embed = embed[embed_wids]

    if id2word is not None:
        y_labels = [id2word[wid] for wid in wids]
        g = sns.heatmap(picked_embed,
                        yticklabels=y_labels,
                        cmap='magma_r', ax=ax,
                        vmin=-0.1, vmax=1.0,
                        cbar_ax=cb_ax, cbar=cb_ax is not None)
        y_tick_interval = len(picked_embed) // len(y_labels)
        y_ticks = list(np.arange(0.5, len(picked_embed), y_tick_interval))
        if images_per_label > 1:
            y_ticks = [t + images_per_label // 2 for t in y_ticks]
        g.set_yticks(y_ticks)
        g.set_yticklabels(y_labels,
                          rotation=0, fontsize=18)
    else:
        g = sns.heatmap(picked_embed, cmap='magma_r', ax=ax, yticklabels=False,
                        vmin=-0.1, vmax=1.0, cbar_ax=cb_ax,
                        cbar=cb_ax is not None)

    if upper:
        # ax.set_title(name, fontsize=32)
        ax.text(0.5, 1.125, name, fontsize=32,
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes)
        x_tick_interval = 20
    else:
        x_tick_interval = 1

    # update the x-axis ticks
    g.set_xticks(np.arange(0.5, len(picked_embed[0]), x_tick_interval))

    # generate x-axis tick labels based on the tick positions
    xticklabels = range(0, len(picked_embed[0]), x_tick_interval)

    # set the x-axis tick labels
    g.set_xticklabels(xticklabels, rotation=0, fontsize=24)

    s = 25 * images_per_label / len(picked_embed) if upper else 1
    padding = 0.001
    lw = 1 if upper else 5

    # broadening the range of Axes by padding
    if upper:
        ax.set_xlim(-len(picked_embed[0]) * 2 * padding, len(picked_embed[0]))
        ax.set_ylim(len(picked_embed), -len(picked_embed) * 2 * padding)
    else:
        ax.set_xlim(-len(picked_embed[0]) * padding,
                    len(picked_embed[0]) * (s + padding))
        ax.set_ylim(len(picked_embed) * (s + padding),
                    -len(picked_embed) * padding)

    if cb_ax is not None:
        cbar = g.collections[0].colorbar
        cbar.ax.tick_params(labelsize=32)

    # convert the coordinate
    trans = ax.transData

    # coordinate of the rectangle
    x = -padding * len(picked_embed[0])
    y = -padding * len(picked_embed)
    width = len(picked_embed[0]) * (s + padding)
    height = len(picked_embed) * (s + padding)

    # draw the border using hlines and vlines
    if upper:
        rect = Rectangle((x, y), width, height, fill=False, edgecolor='black',
                         lw=lw, transform=trans,
                         joinstyle='miter', capstyle='butt')
        ax.add_patch(rect)
    else:
        lines = [Line2D([x, x + width], [y, y], lw=lw, color='black'),
                 Line2D([x + width, x + width], [y, y + height],
                        lw=lw, color='black'),
                 Line2D([x + width, x], [y + height, y + height],
                        lw=lw, color='black'),
                 Line2D([x, x], [y + height, y], lw=0.8 * lw, color='black')]
        for line in lines:
            ax.add_line(line)


def heatmap(labels_per_axis, images_per_label, base_list,
            normed_sorted_ref_embed,
            normed_sorted_base_embeds,
            normed_perm_en_embed,
            ref_lids,
            short_names, base_ids, en_words,
            en_id2word, en_word2id, img_path, logger, dpi=None):

    col = len(base_list) + 1
    figx = 4.25 * col
    figy = 5 * 2.5
    wspace = 0.45

    fig = plt.figure(figsize=(figx, figy))
    gs1 = gridspec.GridSpec(1, col, figure=fig,
                            width_ratios=[1]*col,
                            wspace=wspace, bottom=0.75, top=0.925)
    gs2 = gridspec.GridSpec(1, col, figure=fig,
                            width_ratios=[1]*col,
                            wspace=wspace, bottom=0.05, top=0.7)
    fig.subplots_adjust(left=0.1, right=0.92)
    gss = [gs1, gs2]

    cb_ax = fig.add_axes([.94, 0.05, .015, .875])
    cb_ax.tick_params(labelsize=30)

    sample_axis = 5

    for rx in tqdm(list(range(2))):

        if rx == 1:
            ref_lids = ref_lids[:sample_axis * labels_per_axis]
            base_ids = base_ids[
                :sample_axis * labels_per_axis * images_per_label]
            en_words = en_words[:sample_axis * labels_per_axis]

            for ref_lid, en_word in zip(ref_lids, en_words):
                logger.info(f'{short_names[ref_lid]} -> {en_word}')

            normed_sorted_ref_embed = normed_sorted_ref_embed[:, :sample_axis]
            normed_sorted_base_embeds = [
                normed_sorted_base_embed[:, :sample_axis]
                for normed_sorted_base_embed in normed_sorted_base_embeds]
            normed_perm_en_embed = normed_perm_en_embed[:, :sample_axis]

        en_wids = [en_word2id[word] for word in en_words]

        # plot image
        if rx == 0:
            # ref
            ax_ref = fig.add_subplot(gss[rx][0])
            sub_figure(long_name2title(base_list[0]),
                       normed_sorted_ref_embed, base_ids,
                       ax_ref, upper=True, cb_ax=cb_ax,
                       images_per_label=images_per_label)
            # base
            for bx, base in enumerate(base_list[1:]):
                ax_base = fig.add_subplot(gss[rx][bx+1])
                sub_figure(long_name2title(base),
                           normed_sorted_base_embeds[bx], base_ids,
                           ax_base, upper=True,
                           images_per_label=images_per_label)
            # fasttext
            ax_ref = fig.add_subplot(gss[rx][-1])
            sub_figure('fastText', normed_perm_en_embed, en_wids,
                       ax_ref, upper=True, cb_ax=cb_ax)

        elif rx == 1:
            # ref
            ax_ref = fig.add_subplot(gss[rx][0])
            sub_figure(long_name2title(base_list[0]),
                       normed_sorted_ref_embed, ref_lids,
                       ax_ref, id2word=short_names,
                       embed_wids=base_ids,
                       images_per_label=images_per_label)
            # base
            for bx, base in enumerate(base_list[1:]):
                ax_base = fig.add_subplot(gss[rx][bx+1])
                sub_figure(long_name2title(base),
                           normed_sorted_base_embeds[bx], base_ids,
                           ax_base, images_per_label=images_per_label)
            # fasttext
            ax_ref = fig.add_subplot(gss[rx][-1])
            sub_figure('fastText', normed_perm_en_embed, en_wids,
                       ax_ref, id2word=en_id2word)

    if Path(img_path).suffix == '.png':
        assert dpi is not None
        plt.savefig(img_path, dpi=dpi)
    else:
        plt.savefig(img_path)
    plt.close()


def sample_images(labels_per_axis, images_per_label,
                  base_ids, base_id2file_name, img_path, root_dir, dpi=None):

    margin = 25
    vertical_images = []
    images = []

    base_ids = base_ids[:5 * labels_per_axis * images_per_label]

    for idx, base_id in enumerate(base_ids):
        fig_path = root_dir / 'data/image/imagenet_100k/images/' /\
            base_id2file_name[base_id]
        # load image and convert to RGB format
        image = Image.open(fig_path).convert('RGB')
        # resize image
        image = image.resize((224, 224))
        # append image to images list
        images.append(image)
        if (idx + 1) % images_per_label == 0:
            space = np.full((margin, 224, 3), 255, dtype=np.uint8)
            images.append(space)
        # check if the images list contains
        # labels_per_axis * images_per_label images
        if (idx + 1) % (labels_per_axis * images_per_label) == 0:
            images.pop()
            # concatenate images vertically with a 10-pixel white space
            vertical_image = np.concatenate(images, axis=0)
            vertical_images.append(vertical_image)
            # add a 10-pixel white space horizontally
            space = np.full((vertical_image.shape[0], margin, 3), 255,
                            dtype=np.uint8)
            vertical_images.append(space)
            images = []

    assert len(images) == 0

    horizontal_image = np.concatenate(vertical_images[:-1], axis=1)
    h, w = horizontal_image.shape[:2]
    fig, ax = plt.subplots(figsize=(w//100, h//100))

    ax.set_xticks(np.arange(0.5, 5) * (224 + margin) - margin / 2)
    xticklabels = range(0, 5)
    ax.set_xticklabels(xticklabels, rotation=0, fontsize=70)

    ax.set_yticks(
        np.arange(0.5, 5) * (images_per_label * 224 + margin) - margin / 2)
    yticklabels = range(1, 6)
    ax.set_yticklabels(yticklabels, rotation=0, fontsize=70)

    # move y-axis ticks and label to the right
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")

    # ax.set_xlabel('Axis', fontsize=50)
    ax.set_ylabel('Top', fontsize=80)

    fig.subplots_adjust(left=0.05, right=0.825, bottom=0.025, top=0.95)
    ax.imshow(horizontal_image, vmin=0, vmax=255)

    if Path(img_path).suffix == '.png':
        assert dpi is not None
        plt.savefig(img_path, dpi=dpi)
    else:
        plt.savefig(img_path)
    plt.close()


def merge_images(himg_path, simg_path, output_path,
                 top_cut_ratio=0.1, bottom_cut_ratio=0.05):
    # load two images
    image1 = Image.open(himg_path)
    image2 = Image.open(simg_path)

    # get the width and height of the images
    width1, height1 = image1.size
    width2, height2 = image2.size

    # calculate the number of pixels to cut based on the top and bottom ratios
    top_cut_pixels = int(height2 * top_cut_ratio)
    bottom_cut_pixels = int(height2 * bottom_cut_ratio)

    # crop image2 from the top and bottom
    cropped_image2 = image2.crop(
        (0, top_cut_pixels, width2, height2 - bottom_cut_pixels))

    # get the width and height of the cropped image2
    cropped_width2, cropped_height2 = cropped_image2.size

    # calculate the scale factor based on
    # the height of the first image and the height of the cropped image2
    scale_factor = height1 / float(cropped_height2)

    # resize the cropped image2 based on the scale factor
    resized_image2 = cropped_image2.resize(
        (int(cropped_width2 * scale_factor), height1), Image.ANTIALIAS)

    # get the width of the resized image2
    resized_width2 = resized_image2.size[0]

    # create a new image with a white background
    padding = 10
    horizontal_concatenated_image = Image.new(
        'RGB', (width1 + resized_width2 + padding, height1), (255, 255, 255))
    horizontal_concatenated_image.paste(image1, (0, 0))
    horizontal_concatenated_image.paste(resized_image2, (width1 + padding, 0))

    # save
    if output_path.suffix == '.png':
        horizontal_concatenated_image.save(output_path)
    elif output_path.suffix == '.pdf':
        # resize
        horizontal_concatenated_image = horizontal_concatenated_image.resize(
            (int(horizontal_concatenated_image.size[0] * 0.25),
             int(horizontal_concatenated_image.size[1] * 0.25)),
            Image.ANTIALIAS)
        horizontal_concatenated_image.save(output_path, save_all=True)


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

    np.random.seed(args.seed)
    dpi = args.dpi

    logger = get_logger()
    logger.info(f'base_list: {base_list}')

    output_dir = root_dir / 'output/image'
    dump_path = output_dir / bases /\
        f'{pp}-en_{langs}_{emb_type}-image_{bases}.pkl'
    if not dump_path.exists():
        raise FileNotFoundError(f'{dump_path} does not exist! '
                                'Run image_show_axis_corr.py first.')
    with open(dump_path, 'rb') as f:
        dump = pkl.load(f)
    normed_sorted_ref_embed, normed_sorted_base_embeds, \
        normed_perm_en_embed, en_id2word, _, en_word2id, \
        base_id2file_name, _ = dump

    # labels
    label_path = root_dir / 'data/image/imagenet/LOC_synset_mapping.txt'
    logger.info(f'label_path: {label_path}')
    label2name_words = dict()
    labels = []
    label2names = dict()
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            line_split = line.split(' ')
            label = line_split[0]
            labels.append(label)
            names = ' '.join(line_split[1:]).split(',')
            label2names[label] = ' '.join(line_split[1:])
            name_words = []
            for name in names:
                for name_word in name.split(' '):
                    name_word = name_word.strip().lower()
                    if len(name_word) > 0:
                        name_words.append(name_word)
            name_words = list(set(name_words))
            label2name_words[label] = name_words

    # perm normed sorted base embeds
    tmp = []
    for bx, _ in enumerate(base_list[1:]):
        tmp.append(normed_sorted_base_embeds[bx])
    normed_sorted_base_embeds = tmp

    axis_num = 100
    logger.info(f'given axis_num: {axis_num}. check the dimension...')
    _, dim = normed_sorted_ref_embed.shape
    logger.info(f'dim: {dim}')
    axis_num = min(axis_num, dim)
    axis_num = min(axis_num, len(base_id2file_name))
    logger.info(f'valid axis_num: {axis_num}')

    logger.info('start calculating...')
    # note that the order of each axis is already sorted by the correlation
    images_per_label = 3
    labels_per_axis = 5
    short_names = dict()
    ref_lids = []
    base_ids = []
    en_words = []
    for idx in tqdm(list(range(axis_num))):
        ref_axis = normed_sorted_ref_embed[:, idx]
        base_axises = []
        for bx, _ in enumerate(base_list[1:]):
            base_axis = normed_sorted_base_embeds[bx][:, idx]
            base_axises.append(base_axis)
        en_axis = normed_perm_en_embed[:, idx]

        # sort ref_axis by the sum of the component for each label
        labelwise_ref_axis = [0] * (len(base_id2file_name) // 100)
        for base_id in range(len(base_id2file_name)):
            labelwise_ref_axis[base_id // 100] += ref_axis[base_id]
        lids = np.argsort(-np.array(labelwise_ref_axis))

        lid_tmp = []
        name_tmp = []
        base_tmp = []
        en_tmp = []
        for li in lids:
            if li in ref_lids:
                continue

            # check if we can use the label
            label = labels[li]
            unused_words = []
            for name_word in label2name_words[label]:
                if name_word in en_word2id and \
                        name_word not in STOPWORDS:
                    unused_words.append(name_word)
            if len(unused_words) == 0:
                continue
            lid_tmp.append(li)

            # use the first name as the short name for the label
            short_name = label2names[label].split(',')[0]
            name_tmp.append((li, short_name))

            # choose random images for each label
            bids = [100 * li + i for i in range(100)]
            sampled_ids = np.random.choice(
                bids, images_per_label, replace=False)
            tmp_ids = np.argsort(
                -np.array([ref_axis[id_] for id_ in sampled_ids]))
            sampled_ids = sampled_ids[tmp_ids]
            base_tmp.append(list(sampled_ids))

            # choose the word representing the label with the maximum component
            max_id = np.argmax(
                [en_axis[en_word2id[w]] for w in unused_words])
            en_tmp.append(unused_words[max_id])

            if len(en_tmp) == labels_per_axis:
                break

        if len(en_tmp) != labels_per_axis:
            break

        ref_lids += lid_tmp
        for li, short_name in name_tmp:
            short_names[li] = short_name
        base_ids += base_tmp
        en_words += en_tmp

        if len(en_words) == axis_num * labels_per_axis:
            break

    axis_num = len(en_words) // labels_per_axis
    logger.info(f'final axis_num: {axis_num}')
    base_ids = list(np.array(base_ids).flatten())

    normed_sorted_ref_embed = normed_sorted_ref_embed[:, :axis_num]
    normed_sorted_base_embeds = [normed_sorted_tgt_embed[:, :axis_num]
                                 for normed_sorted_tgt_embed in
                                 normed_sorted_base_embeds]
    normed_perm_en_embed = normed_perm_en_embed[:, :axis_num]

    (output_dir / 'figures/embeddings_heatmap'
     ).mkdir(exist_ok=True, parents=True)

    # save heatmap images
    himg_path = output_dir / 'figures/embeddings_heatmap/'\
        f'{pp}-embeddings_heatmap-{langs}_{emb_type}-{bases}-{dpi}dpi.png'
    logger.info(f'saving {himg_path}...')
    heatmap(labels_per_axis, images_per_label, base_list,
            normed_sorted_ref_embed,
            normed_sorted_base_embeds,
            normed_perm_en_embed,
            ref_lids,
            short_names, base_ids, en_words,
            en_id2word, en_word2id, himg_path, logger, dpi=dpi)

    # save sample images
    simg_path = output_dir / 'figures/embeddings_heatmap/'\
        f'{pp}-sample_images-{langs}_{emb_type}-{bases}-{dpi}dpi.png'
    logger.info(f'saving {simg_path}...')
    sample_images(labels_per_axis, images_per_label, base_ids,
                  base_id2file_name, simg_path, root_dir, dpi=dpi)

    # save merged images
    output_path = output_dir / 'figures/embeddings_heatmap/'\
        f'{pp}-merged_embeddings_heatmap-'\
        f'{langs}_{emb_type}-{bases}-{dpi}dpi.png'
    logger.info(f'saving {output_path}...')
    merge_images(himg_path, simg_path, output_path)


if __name__ == '__main__':
    main()
