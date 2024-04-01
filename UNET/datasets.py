import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms, datasets
import numpy as np
import os


class ChestXray(Dataset):
    def __init__(self, split, root_dir = '../data', eval = False):
        self.root_dir = root_dir
        self.split = split
        self.eval = eval
        self.imagePath = os.path.join(root_dir, 'images_' + self.split)
        self.maskPath = os.path.join(root_dir, 'mask_' +  self.split)
        self.trainImagesList = [img_name.split('.')[0] for img_name in os.listdir(self.imagePath)]
    def __len__(self):
        return len(self.trainImagesList)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.imagePath, self.trainImagesList[index] + '.png'))
        mask = np.load(os.path.join(self.maskPath, self.trainImagesList[index] + '.npy'))

        image = transforms.ToTensor()(image)
        mask = np.array(mask)
        mask = torch.from_numpy(mask)
        # print(f'Image given for training: {self.trainImagesList[index]}')
        mask = mask.type(torch.LongTensor)
        if self.eval:
            return image, mask, self.trainImagesList[index] + '.png'
        else:
            return image, mask
