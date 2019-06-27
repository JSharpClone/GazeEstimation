import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
from torchvision.transforms import functional as tf
from PIL import Image

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



class FaceGazeDataset(Dataset):
    def __init__(self, data_root, test_idx, mode='train', img_size=(224, 224)):
        super(FaceGazeDataset, self).__init__()
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
        img.save('out.png')
        img = tf.to_tensor(img)

        w, h = screen_size[subject_idx]
        label = [float(annotation[1])/w, float(annotation[2])/h] # w, h
        label = torch.FloatTensor(np.asarray(label))
        
        return img, label

    def __len__(self):
        return self.data_len

if __name__ == '__main__':
    root = '/mnt/data/MPII/MPIIFaceGaze/'
    dataset = FaceGazeDataset(root, 0, mode='validation')
    img, label = dataset[2000]
    # print(img.size())
    # print(len(dataset))