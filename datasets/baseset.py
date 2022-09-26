import numpy as np
import torch
from torchvision import transforms
# from .transforms.dispatcher import dispatcher

class JointCompose:
    '''
    Resembles

    https://pytorch.org/vision/stable/_modules/torchvision/transforms/transforms.html#Compose

    but it works with joint transformation (i.e., both images and label map)
    '''
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return (img, target)

class base_set(torch.utils.data.Dataset):
    '''
    An implementation of torch.utils.data.Dataset that supports various
    data transforms and augmentation.
    '''
    def __init__(self, dataset, split):
        '''
        Args:
            dataset: any object with __getitem__ and __len__ methods implemented.
                Object retruned from dataset[i] is expected to be (raw_tensor, label).
            split: ("train" or "test"). Specify dataset mode
            cfg: yacs root config node object.
        '''
        assert split in ["train", "test"]
        self.dataset = dataset
    
    def __getitem__(self, key):
        if isinstance(key, tuple):
            assert len(key) == 2
            assert isinstance(key[0], int)
            assert isinstance(key[1], dict)
            params = key[1]
        elif isinstance(key, int) or isinstance(key, np.integer):
            index = key
            params = {}
        else:
            raise NotImplementedError
        data, label, img_idxs = self.dataset[index]
        return (data, label, img_idxs)

    def __len__(self):
        return len(self.dataset)
