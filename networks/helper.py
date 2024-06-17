from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletDataset(Dataset):
    def __init__(self, datasetA, datasetB):
        self.datasetA = datasetA
        self.datasetB = datasetB

    def __len__(self):
        return len(self.datasetA)

    def __getitem__(self, idx):
        anchor = self.datasetA[idx]
        positive_idx = torch.randint(0, len(self.datasetA), (1,)).item()
        while positive_idx == idx:
            positive_idx = torch.randint(0, len(self.datasetA), (1,)).item()
        positive = self.datasetA[positive_idx]
        
        negative_idx = torch.randint(0, len(self.datasetB), (1,)).item()
        negative = self.datasetB[negative_idx]
        
        return anchor, positive, negative

class ImgTripletDataset(Dataset):
    def __init__(self, datasetAImg, datasetAState, datasetBImg, datasetBState):
        self.datasetAImg = datasetAImg
        self.datasetAState = datasetAState
        self.datasetBImg = datasetBImg
        self.datasetBState = datasetBState

    def __len__(self):
        return len(self.datasetAImg)

    def __getitem__(self, idx):
        a_i = self.datasetAImg[idx]
        a_s = self.datasetAState[idx]
        positive_idx = torch.randint(0, len(self.datasetAImg), (1,)).item()
        while positive_idx == idx:
            positive_idx = torch.randint(0, len(self.datasetAImg), (1,)).item()
        p_i = self.datasetAImg[positive_idx]
        p_s = self.datasetAState[positive_idx]
        
        negative_idx = torch.randint(0, len(self.datasetBImg), (1,)).item()
        n_i = self.datasetBImg[negative_idx]
        n_s = self.datasetBState[negative_idx]
        
        return a_i, a_s, p_i, p_s, n_i, n_s

class TransitionDataset(Dataset):
    def __init__(self, s, a, s_n):
        self.s = s
        self.a = a
        self.s_n = s_n

    def __len__(self):
        return len(self.s)

    def __getitem__(self, idx):
        return self.s[idx], self.a[idx], self.s_n[idx]

class ImgTransitionDataset(Dataset):
    def __init__(self, i, i_n, s, s_n, a):
        self.i = i
        self.i_n = i_n
        self.s = s
        self.s_n = s_n
        self.a = a

    def __len__(self):
        return len(self.s)

    def __getitem__(self, idx):
        return self.i[idx], self.i_n[idx], self.s[idx], self.s_n[idx], self.a[idx]
    

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        positive_dist = F.pairwise_distance(anchor, positive)
        negative_dist = F.pairwise_distance(anchor, negative)
        losses = F.relu(positive_dist - negative_dist + self.margin)
        return losses.mean()
