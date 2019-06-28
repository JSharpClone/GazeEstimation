from data import Face2DGazeDataset, Face3DGazeDataset
from model import Model

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import os



epochs = 5
device = 'cuda:0'
root = '/mnt/data/MPII/MPIIFaceGaze_normalized/'

# model = Model().to(device)
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# train_dataset = Face2DGazeDataset(root, 0, mode='train')
# valid_dataset = Face2DGazeDataset(root, 0, mode='validation')

# train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=2)
# valid_loader = DataLoader(valid_dataset, batch_size=512)


def train(model, criterion, optimizer, train_loader, dataset_size):
    model.train()

    loss_list = []

    with tqdm(total=dataset_size, ascii=True) as pbar:
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

def validation(model, criterion, optimizer, valid_loader, dataset_size):
    model.eval()

    loss_list = []

    with tqdm(total=dataset_size, ascii=True) as pbar:
        for img_batch, label_batch in iter(valid_loader):
            img_batch = img_batch.to(device)
            label_batch = label_batch.to(device)

            output_batch = model(img_batch)
            loss = criterion(output_batch, label_batch)
            pbar.update(len(img_batch))
            pbar.set_description('Loss: {:.4f}'.format(loss.item()))
            loss_list.append(loss.item())
    
    return sum(loss_list)/len(loss_list)


def log(loss, save_path, test_idx):
    fig, ax = plt.subplots()
    ax.plot(range(len(loss)), loss)
    fig.savefig('loss_{}.png'.format(test_idx))

    save_path = os.path.join(save_path, str(test_idx))
    if not os.path.isdir(save_path):
        os.makedirs(save_path, exist_ok=True)
    torch.save(model, os.path.join(save_path, 'model.pth'))


if __name__ == '__main__':
    cross_valid_loss = []
    
    for test_idx in range(15):
        loss = []
        model = Model().to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        train_dataset = Face3DGazeDataset(root, test_idx, mode='train')
        valid_dataset = Face3DGazeDataset(root, test_idx, mode='validation')

        train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=2)
        valid_loader = DataLoader(valid_dataset, batch_size=512)

        for e in range(epochs):
            loss += train(model, criterion, optimizer, train_loader, len(train_dataset))
            log(loss, './model/3D', test_idx)

        valid_loss = validation(model, criterion, optimizer, valid_loader, len(valid_dataset))
        cross_valid_loss.append(valid_loss)
    
    for idx, loss in enumerate(cross_valid_loss):
        print('Valid idx: {}, Loss: {:.4f}'.format(idx, loss))

    # for test_idx in range(2):
    #     model = torch.load('./model/{}/model.pth'.format(test_idx))
        
    #     criterion = nn.MSELoss()

    #     valid_dataset = FaceGazeDataset(root, test_idx, mode='validation')
    #     valid_loader = DataLoader(valid_dataset, batch_size=32)

    #     model.eval()

    #     loss_list = []

    #     with tqdm(total=len(valid_dataset), ascii=True) as pbar:
    #         for img_batch, label_batch in iter(valid_loader):
    #             img_batch = img_batch.to(device)
    #             label_batch = label_batch.to(device)

    #             output_batch = model(img_batch)
    #             loss = criterion(output_batch, label_batch)
    #             pbar.update(len(img_batch))
    #             pbar.set_description('Loss: {:.4f}'.format(loss.item()))
    #             loss_list.append(loss.item())
    #     print('Test idx: {}, Loss: {}'.format(test_idx, sum(loss_list)/len(loss_list)))