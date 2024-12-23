from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils import data
from torchvision.transforms import v2

def getData(root, mode):
    if mode == 'train':
        df = pd.read_csv(f'{root}/train.csv')
    elif mode == 'valid':
        df = pd.read_csv(f'{root}/valid.csv')
    else:
        df = pd.read_csv(f'{root}/test.csv')
    path = df['filepaths'].tolist()
    labels = df['labels'].tolist()
    label_id = df['label_id'].tolist()
    return path, labels, label_id

class BufferflyMothLoader(data.Dataset):
    def __init__(self, root, mode, batch_size = None):
        """
        Args:
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.img_name, self.labels, self.label_id = getData(root, mode)
        self.mode = mode
        #self.total_labels = len(set(self.labels))
        self.batch_size = batch_size
        print("> Found %d images..." % (len(self.img_name)))  

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        """something you should implement here"""

        """
           step1. Get the image path from 'self.img_name' and load it.
                  hint : path = root + self.img_name[index] + '.jpg'
           
           step2. Get the ground truth label from self.label
                     
           step3. Transform the .jpg rgb images during the training phase, such as resizing, random flipping, 
                  rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints. 
                       
                  In the testing phase, if you have a normalization process during the training phase, you only need 
                  to normalize the data. 
                  
                  hints : Convert the pixel value to [0, 1]
                          Transpose the image shape from [H, W, C] to [C, H, W]
                         
            step4. Return processed image and label
        """
        path = self.root + "/" + self.img_name[index]
        ground_truth_label = self.labels[index]
        ground_truth_label_id = self.label_id[index]

        img = Image.open(path)
        #img_np = np.array(img)
        #img_np = img_np.astype(np.float32)
        
        #img_transpose = np.transpose(img_np, (2,0,1))
        #img_torch = torch.from_numpy(img_np).cuda()
        #print(img_torch.shape)
        #print(img_torch.shape)
        if self.mode == 'train':
            transforms = v2.Compose([
                #v2.ToDtype(torch.float32, scale=True),
                v2.RandomResizedCrop(size=(224, 224), antialias=True),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomRotation(degrees=15),
                v2.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                v2.RandomZoomOut(),
                v2.RandomCrop(size=(224, 224),padding=25,fill=125,padding_mode="edge"),
                v2.ToTensor(),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            img_torch = transforms(img)
            #print(img_torch.shape)
            #plt.figure()
            #imshow(img, title='Content Image')
            #img = img_torch.unsqueeze(0)
            #img = img.requires_grad_(True)
            return img_torch, ground_truth_label_id
        else:
            transforms = v2.Compose([
                #v2.ToDtype(torch.float32, scale=True),
                v2.RandomResizedCrop(size=(224, 224), antialias=True),
                v2.ToTensor(),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            img_torch = transforms(img)
            return img_torch, ground_truth_label_id