from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
from torchvision.io import read_image

class WikiArts(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        ''''
        img_dir: images are stored in a directory img_dir
        annotations_file: labels are stored separately in a pickle file
        '''
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

        