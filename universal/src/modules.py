import timm
import torch.nn as nn
from timm.data import ImageDataset


class CustomImageDataset(ImageDataset):
    def __init__(self, root_dir, transform=None):
        super().__init__(
            root_dir,
            transform=transform,
        )

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        img_path = self.filename(idx, absolute=True)
        return img, img_path


class TimmFeatureExtractor(nn.Module):
    def __init__(self, base, pretrained):
        super().__init__()
        self.model = timm.create_model(base, pretrained=pretrained)
        self.base = base
        print(self.model)

        if 'resnet' in base:
            self.model = nn.Sequential(*list(self.model.children())[:-1])
        elif 'vit' in base or 'swin' in base or 'resmlp' in base:
            self.model.head = nn.Identity()
        elif 'regnety_002' in base:
            self.model.head.fc = nn.Identity()

    def forward(self, x):
        features = self.model(x)

        if 'resnet' in self.base:
            features = features.view(features.size(0), -1)

        return features
