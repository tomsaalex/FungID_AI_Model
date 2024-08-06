import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from PIL import Image


def process_labels(image_labels):
    text_labels = image_labels.iloc[:, 0]
    unique_labels = text_labels.unique()
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    #print(label_map)
    for idx, label in enumerate(unique_labels):
        print(idx, label)
    return label_map


class MO_106_Dataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, target_transform=None):
        self.image_labels = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.label_map = process_labels(self.image_labels)

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_labels.iloc[idx, 0], self.image_labels.iloc[idx, 1])

        image = Image.open(img_path)
        label = self.image_labels.iloc[idx, 0]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        numeric_label = self.label_map[label]

        #image = torch.from_numpy(image).type(torch.float32)

        return image, numeric_label
