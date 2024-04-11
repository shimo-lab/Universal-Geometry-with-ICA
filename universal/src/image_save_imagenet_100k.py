import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='alignment task evaluation')

    # root directory
    parser.add_argument('--root_dir', type=str, default='/working')

    # random seed
    parser.add_argument('--seed', type=int, default=0)


def main():

    args = parse_args()

    root_dir = Path(args.root_dir)

    np.random.seed(args.seed)

    data_dir = root_dir / 'data/image/imagenet'
    image_root_dir = data_dir / 'ILSVRC' / 'Data' / 'CLS-LOC' / 'train'
    label_file = data_dir / 'LOC_synset_mapping.txt'
    with open(label_file, 'r') as f:
        lines = f.readlines()
    lines = [line.strip().split(' ') for line in lines]
    labels = [line[0] for line in lines]

    output_root_dir = root_dir / 'data/image/imagenet_100k'
    output_dir = output_root_dir / 'images'
    output_dir.mkdir(parents=True, exist_ok=True)
    data = []
    for label in tqdm(labels):
        image_dir = image_root_dir / label
        image_paths = [x for x in image_dir.glob('*.JPEG')]
        hundred_image_paths = np.random.choice(image_paths, 100, replace=False)

        save_dir = output_dir / label
        save_dir.mkdir(parents=True, exist_ok=True)
        for image_path in hundred_image_paths:
            image_name = image_path.name
            save_path = save_dir / image_name
            shutil.copy(image_path, save_path)

            data.append([label, image_name])

    df = pd.DataFrame(data, columns=['label', 'image_name'])
    df.to_csv(output_root_dir / 'imagenet_100k.csv', index=False)


if __name__ == '__main__':
    main()
