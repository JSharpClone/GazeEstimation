from data import FaceGazeDataset
from model import Model

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn')



epochs = 50
device = 'cuda:0'
root = '/mnt/data/MPII/MPIIFaceGaze/'

model = Model().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_dataset = FaceGazeDataset(root, 0, mode='train')
valid_dataset = FaceGazeDataset(root, 0, mode='validation')

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=2)
valid_loader = DataLoader(valid_dataset, batch_size=512)


def train():
    model.train()

    loss_list = []

    with tqdm(total=len(train_dataset), ascii=True) as pbar:
        for img_batch, label_batch in iter(train_loader):
            img_batch = img_batch.to(device)
            label_batch = label_batch.to(device)

            optimizer.zero_grad()
            output_batch = model(img_batch)
            loss = criterion(output_batch, label_batch)
            loss.backward()
            optimizer.step()
            pbar.update(len(img_batch))
            pbar.set_description('Loss: {:.4f}'.format(loss.item()))
            loss_list.append(loss.item())
    
    return loss_list

def log(loss):
    fig, ax = plt.subplots()
    ax.plot(range(len(loss)), loss)
    fig.savefig('loss.png')

if __name__ == '__main__':
    loss = []
    for e in range(epochs):
        loss += train()
        log(loss)