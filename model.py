import torch
import torch.nn as nn
import torch.nn.functional as F
from data import FaceGazeDataset

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        self.fc1 = nn.Linear(40000, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), kernel_size=2, stride=2)
        x = F.relu(x, inplace=True)
        x = F.max_pool2d(self.conv2(x), kernel_size=2, stride=2)
        x = F.relu(x, inplace=True) 
        x = F.max_pool2d(self.conv3(x), kernel_size=2, stride=2)
        x = F.relu(x, inplace=True) 
        x = x.view(x.size(0), -1) # flatten
        x = F.relu(self.fc1(x), inplace=True)
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    root = '/mnt/data/MPII/MPIIFaceGaze/'
    dataset = FaceGazeDataset(root, 0, mode='validation')
    img, label = dataset[2000]
    img = img.unsqueeze(0)
    
    model = Model()
    y = model(img)
    print(a.size())

