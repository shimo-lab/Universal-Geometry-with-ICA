import argparse
import pickle as pkl
from pathlib import Path

import numpy as np
import torch
from timm.data.transforms_factory import create_transform
from torch.utils.data import DataLoader
from tqdm import tqdm

from modules import CustomImageDataset, TimmFeatureExtractor
from utils import check_model, get_logger


def parse_args():
    parser = argparse.ArgumentParser(description='save raw embeddings')

    # root directory
    parser.add_argument('--root_dir', type=str, default='/working')

    # image model
    parser.add_argument("--base", type=str, default="resnet18")
    parser.add_argument("--batch_size", type=int, default=1024)

    return parser.parse_args()


def main():
    args = parse_args()
    root_dir = Path(args.root_dir)
    base = args.base
    batch_size = args.batch_size
    check_model(base)

    logger = get_logger()
    logger.info(f'base: {base}')
    logger.info(f'batch_size: {batch_size}')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    data_dir = root_dir / 'data/image/imagenet_100k'
    image_root_dir = data_dir / 'images'

    imagenet_ds = CustomImageDataset(
        image_root_dir,
        transform=create_transform(input_size=224, is_training=False))
    imagenet_dl = DataLoader(imagenet_ds, batch_size=batch_size,
                             shuffle=False, num_workers=0)
    model = TimmFeatureExtractor(base, pretrained=True).to(device)

    all_embeds = []
    all_names = []
    for img, img_paths in tqdm(imagenet_dl):
        img = img.to(device)
        with torch.no_grad():
            out = model(img)
            embeds = out.cpu().numpy()
            for embed, img_path in zip(embeds, img_paths):
                img_path = img_path.replace(
                    str(root_dir / 'data/image/imagenet_100k/images/'), '')
                parent = Path(img_path).parent
                name = Path(img_path).name
                all_embeds.append(embed)
                all_names.append(f'{parent}/{name}')

    all_embeds = np.array(all_embeds)
    all_names = np.array(all_names)
    logger.info(f'all_embeds.shape: {all_embeds.shape}')

    output_dir = root_dir / 'output/image/'
    output_dir.mkdir(parents=True, exist_ok=True)
    names_embeds_path = output_dir / f'{base}-raw.pkl'
    with open(names_embeds_path, 'wb') as f:
        pkl.dump((all_names, all_embeds), f)


if __name__ == '__main__':
    main()
