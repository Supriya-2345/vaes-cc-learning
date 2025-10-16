import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import random



class Shapes(object):

    def __init__(self, dataset_zip=None):
        loc = 'data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
        if dataset_zip is None:
            self.dataset_zip = np.load(loc, encoding='latin1')
        else:
            self.dataset_zip = dataset_zip
        self.imgs = torch.from_numpy(self.dataset_zip['imgs']).float()

    def __len__(self):
        return self.imgs.size(0)

    def __getitem__(self, index):
        x = self.imgs[index].view(1, 64, 64)
        return x


class Dataset(object):
    def __init__(self, loc):
        self.dataset = torch.load(loc).float().div(255).view(-1, 1, 64, 64)

    def __len__(self):
        return self.dataset.size(0)

    @property
    def ndim(self):
        return self.dataset.size(1)

    def __getitem__(self, index):
        return self.dataset[index]


class Faces(Dataset):
    LOC = 'data/basel_face_renders.pth'

    def __init__(self):
        return super(Faces, self).__init__(self.LOC)


class Concon(Dataset):
    def __init__(self, root_dir='/u/student/2023/cs23mtech11019/data_small', transform=None):
    # def __init__(self, root_dir='/u/student/2023/cs23mtech11019/data/case_disjoint_main/train/images/t0', transform=None):    
        self.samples = []
        self.transform = transform if transform else transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # Convert to 1 channel
            transforms.Resize((64, 64)),                  # Resize to match dSprites
            transforms.ToTensor()                         # Convert to tensor (C x H x W)
        ])
        
        for label in [0, 1]:
            class_dir = os.path.join(root_dir, str(label))
            for fname in os.listdir(class_dir):
                fpath = os.path.join(class_dir, fname)
                if os.path.isfile(fpath) and fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((fpath, label))

        # Shuffle the dataset to mix class 0 and 1
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label = self.samples[index]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image, label

    