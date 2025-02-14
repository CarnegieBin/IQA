import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms


class IQADataset(Dataset):
    def __init__(self, csv_path: str, folder_path: str, patch_num: int = 1, transform=None):
        data = pd.read_csv(csv_path)
        self.image_names = data['Image_name'].tolist()
        self.scores = data['mos'].tolist()
        self.folder_path = folder_path
        self.transform = transform

        if patch_num > 1:
            self.image_names = self.image_names * patch_num
            self.scores = self.scores * patch_num

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.folder_path, image_name)
        img = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        score = torch.tensor(self.scores[idx], dtype=torch.float32)

        return img, score


def get_dataloader(args):
    train_csv_path = '/data2/CarnegieBin_data/database/IQA/' + args.dataset + '/csv/train.csv'
    val_csv_path = '/data2/CarnegieBin_data/database/IQA/' + args.dataset + '/csv/val.csv'
    test_csv_path = '/data2/CarnegieBin_data/database/IQA/' + args.dataset + '/csv/test.csv'
    folder_path = '/data2/CarnegieBin_data/database/IQA/' + args.dataset + '/Images'
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor()
    ])
    test_transform = val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    train_dataset = IQADataset(csv_path=train_csv_path, folder_path=folder_path,
                               transform=train_transform, patch_num=args.patch_num)
    val_dataset = IQADataset(csv_path=val_csv_path, folder_path=folder_path,
                             transform=val_transform, patch_num=1)
    test_dataset = IQADataset(csv_path=test_csv_path, folder_path=folder_path,
                              transform=test_transform, patch_num=1)

    return train_dataset, val_dataset, test_dataset



