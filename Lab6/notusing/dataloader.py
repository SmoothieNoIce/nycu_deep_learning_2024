import json
import os
from glob import glob
import torch
from torch import stack
from torch.utils.data import Dataset as torchData

from torchvision.datasets.folder import default_loader as imgloader
from torch import stack
def labels_to_tensor(labels, label_to_idx):
    indices = [label_to_idx[label] for label in labels]
    return torch.tensor(indices, dtype=torch.long)


class Dataset_Iclevr(torchData):
    """
    Args:
        root (str)      : The path of your Dataset
        transform       : Transformation to your dataset
        mode (str)      : train, val, test
        partial (float) : Percentage of your Dataset, may set to use part of the dataset
    """

    def __init__(self, root, transform=None, mode="train", dataset_len=7, partial=1.0):
        super().__init__()
        assert mode in ["train", "val"], "There is no such mode !!!"
        self.img_folder = "iclevr"

        if mode == "train":
            f = open("train.json", "r")
            data = json.loads(f.read())
            self.prefix = "train"
        elif mode == "val":
            f = open("train.json", "r")
            data = json.loads(f.read())
            self.prefix = "val"
        else:
            raise NotImplementedError
        f.close()
        
        f = open("objects.json", "r")
        data_objects = json.loads(f.read())
        f.close()

        self.data = data
        self.images = list(data.keys())
        self.objects = data_objects
        
        self.transform = transform
        self.partial = partial
        self.dataset_len = len(data)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_folder, img_name)
        image = imgloader(img_path)
        labels = self.data[img_name]
        labels = list(set(labels))
        indices = [self.objects[item] for item in labels]
        tensor = torch.as_tensor(indices)

        if self.transform:
            image = self.transform(image)

        return image, tensor
