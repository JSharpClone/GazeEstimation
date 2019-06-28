import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

import numpy as np
import os
from torchvision.transforms import functional as tf
from PIL import Image
import h5py
import cv2

subject_num = 15 
subject_instance = [2927,
2904,
2916,
2929,
2860,
2870,
2877,
2843,
2767,
2719,
2194,
2262,
1601,
1498,
1500]

# w, h
screen_size = [(1280, 800), 
(1440, 900), 
(1280, 800),
(1440, 900),
(1280, 800), 
(1440, 900),
(1680, 1050),
(1440, 900),
(1440, 900),
(1440, 900),
(1440, 900),
(1280, 800),
(1280, 800),
(1280, 800),
(1440, 900)]



class Face2DGazeDataset(Dataset):
    def __init__(self, data_root, test_idx, mode='train', img_size=(224, 224)):
        super(Face2DGazeDataset, self).__init__()
        self.data_root = data_root
        self.mode = mode
        self.test_idx = test_idx
        self.img_size = img_size

        if mode == 'train':
            self.data_len = sum(subject_instance) - subject_instance[test_idx]

        elif mode == 'validation':
            self.data_len = subject_instance[test_idx]


    def __getitem__(self, idx):
        if self.mode == 'train':
            for i in range(subject_num):
                if i == self.test_idx:
                    continue
                if idx < subject_instance[i]:
                    subject_idx = i
                    break
                else:
                    idx -= subject_instance[i]
        elif self.mode == 'validation':
            subject_idx = self.test_idx
            idx = idx
            
        subject_name = 'p{:02d}'.format(subject_idx)
        subject_path = os.path.join(self.data_root, subject_name)
        annotation_file = os.path.join(subject_path, subject_name+'.txt')

        with open(annotation_file, 'r') as f:
            annotations = f.readlines()
        
        annotation = annotations[idx].split()
        img_path = os.path.join(subject_path, annotation[0])
        img = Image.open(img_path)
        img = img.convert('RGB').resize(self.img_size)
        img = tf.to_tensor(img)

        w, h = screen_size[subject_idx]
        label = [float(annotation[1])/w, float(annotation[2])/h] # w, h
        label = torch.FloatTensor(np.asarray(label))
        
        return img, label

    def __len__(self):
        return self.data_len

class Face3DGazeDataset(Dataset):
    def __init__(self, data_root, test_idx, mode='train', img_size=(224, 224)):
        super(Face3DGazeDataset, self).__init__()
        self.data_root = data_root
        self.test_idx = test_idx
        self.mode = mode
        self.img_size = img_size
        self.subject_num = 15
        self.subject_instance = 3000

        self.indices = list(range(self.subject_num*self.subject_instance))
        del self.indices[test_idx*self.subject_instance : (test_idx+1)*self.subject_instance]
            
    
    def __getitem__(self, idx):
        idx = self.indices[idx]
        subject_idx = idx // self.subject_instance
        instance_idx = idx % self.subject_instance

        subject_path = os.path.join(self.data_root, 'p{:02d}.mat'.format(subject_idx))
        with h5py.File(subject_path, 'r') as data:
            img = data['Data']['data'][instance_idx]
            label = data['Data']['label'][instance_idx][:2]
        
        img = Image.fromarray(img[:, :, ::-1], mode='RGB')
        img = img.convert('RGB').resize(self.img_size)
        img = tf.to_tensor(img)
        label = torch.FloatTensor(label)

        return img, label

    def __len__(self):
        return len(self.indices)

if __name__ == '__main__':
    # root = '/mnt/data/MPII/MPIIFaceGaze/'
    # dataset = Face2DGazeDataset(root, 0, mode='validation')
    # img, label = dataset[2000]
    # print(img.size())
    # print(len(dataset)) 
    root = '/mnt/data/MPII/MPIIFaceGaze_normalized/'
    dataset = Face3DGazeDataset(root, 0, mode='train')
    img, label = dataset[0]
    print(img)
    
