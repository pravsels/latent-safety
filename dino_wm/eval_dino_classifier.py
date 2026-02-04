import argparse
import collections
import os
import pathlib
import sys
import numpy as np
import ruamel.yaml as yaml
import torch
from termcolor import cprint
import cv2
# add to os sys path
import sys
import matplotlib.pyplot as plt
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
dreamer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../model_based_irl_torch'))
sys.path.append(dreamer_dir)
env_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../real_envs'))
sys.path.append(env_dir)
print(dreamer_dir)
print(sys.path)
import model_based_irl_torch.dreamer.tools as tools
from model_based_irl_torch.dreamer.dreamer import Dreamer
from termcolor import cprint
from real_envs.env_utils import normalize_eef_and_gripper, unnormalize_eef_and_gripper, get_env_spaces
import pickle
from collections import defaultdict
from model_based_irl_torch.dreamer.tools import add_to_cache
from tqdm import tqdm, trange
from model_based_irl_torch.common.utils import to_np
import wandb
from torch.utils.data import Dataset, DataLoader
from dino_wm.config import get_dino_config

dino_cfg = get_dino_config()
if 'hub_source' in dino_cfg:
    dino = torch.hub.load(dino_cfg['hub_repo'], dino_cfg['model_name'],
                          source=dino_cfg['hub_source'], weights=dino_cfg['weights_path'])
else:
    dino = torch.hub.load(dino_cfg['hub_repo'], dino_cfg['model_name'])

import requests
from PIL import Image
from torchvision import transforms

import torch
from torch import nn
from torch.optim import AdamW

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import random
from dino_decoders_official import VQVAE

transform = transforms.Compose([           
                                transforms.Resize(256),                    
                                transforms.CenterCrop(MODEL_CONFIG['image_size'][0]),               
                                transforms.ToTensor(),                    
                                transforms.Normalize(                      
                                mean=[0.485, 0.456, 0.406],                
                                std=[0.229, 0.224, 0.225]              
                                )])


transform1 = transforms.Compose([           
                                transforms.Resize(520),
                                transforms.CenterCrop(518), #should be multiple of model patch_size                 
                                transforms.ToTensor(),                    
                                transforms.Normalize(mean=0.5, std=0.2)
                                ])



DINO_transform = transforms.Compose([           
                            transforms.Resize(MODEL_CONFIG['image_size'][0]),
                            #transforms.CenterCrop(MODEL_CONFIG['image_size'][0]), #should be multiple of model patch_size                 
                            
                            transforms.ToTensor(),])
norm_transform = transforms.Normalize(                      
                                mean=[0.485, 0.456, 0.406],                
                                std=[0.229, 0.224, 0.225]              
                                )

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from typing import Tuple, Optional
from test_loader import SplitTrajectoryDataset
from dino_models import Decoder, VideoTransformer, normalize_acs, normalize_states, batch_quat_to_rotvec, batch_rotvec_to_quat
from dino_wm.config import MODEL_CONFIG
import json
import os

def fail_loss(pred, fail_data):
    
    safe_data = torch.where(fail_data == 0.)
    unsafe_data = torch.where(fail_data == 1.)
    unsafe_data_weak = torch.where(fail_data == 2.)
    
    
    pos = pred[safe_data]
    neg = pred[unsafe_data]
    neg_weak = pred[unsafe_data_weak]
    
    

    gamma = 0.75
    lx_loss = (1/pos.size(0))*torch.sum(torch.relu(gamma - pos)) if pos.size(0) > 0 else 0. #penalizes safe for being negative
    lx_loss +=  (1/neg.size(0))*torch.sum(torch.relu(gamma + neg)) if neg.size(0) > 0 else 0. # penalizes unsafe for being positive
    lx_loss +=  (1/neg_weak.size(0))*torch.sum(torch.relu(neg_weak)) if neg_weak.size(0) > 0 else 0. # penalizes unsafe for being positive

    return lx_loss


def confusion(pred, fail_data):
    
    safe_data = torch.where(fail_data == 0.)
    unsafe_data = torch.where(fail_data == 1.)
    unsafe_data_weak = torch.where(fail_data == 2.)
    
    
    pos = pred[safe_data]
    neg = pred[unsafe_data]
    neg_weak = pred[unsafe_data_weak]
    
    TP = torch.sum(pos > 0).item()
    FN = torch.sum(pos < 0).item()
    FP = torch.sum(neg > 0).item()
    TN = torch.sum(neg < 0).item()

    return torch.tensor([TP, FN, FP, TN])

if __name__ == "__main__":

    use_amp = True
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    BS = 16
    BL= 4
    hdf5_file = '/data/ken/latent-unsafe-test/consolidated.h5'
    
    # Load state normalization stats
    dataset_stats_path = 'dataset_stats.json'  # Update this path as needed
    if not os.path.exists(dataset_stats_path):
        raise FileNotFoundError(
            f"Stats file '{dataset_stats_path}' not found! Please run scripts/compute_stats_json.py to generate it."
        )
    
    print(f"Loading dataset stats from {dataset_stats_path}")
    with open(dataset_stats_path, 'r') as f:
        stats = json.load(f)
    
    # Check for required keys
    required_keys = ["action_min", "action_max", "state_min", "state_max"]
    missing_keys = [k for k in required_keys if k not in stats]
    
    if missing_keys:
        raise ValueError(f"Stats file missing required keys: {missing_keys}")
    
    device = 'cuda:0'
    
    # Create tensors on device
    action_min = torch.tensor(stats['action_min']).float().to(device)
    action_max = torch.tensor(stats['action_max']).float().to(device)
    state_min = torch.tensor(stats['state_min']).float().to(device)
    state_max = torch.tensor(stats['state_max']).float().to(device)
    action_q02 = torch.tensor(stats['action_delta_q02']).float().to(device) if "action_delta_q02" in stats else None
    action_q98 = torch.tensor(stats['action_delta_q98']).float().to(device) if "action_delta_q98" in stats else None
    state_q02 = torch.tensor(stats['state_q02']).float().to(device) if "state_q02" in stats else None
    state_q98 = torch.tensor(stats['state_q98']).float().to(device) if "state_q98" in stats else None
    
    # Infer dimensions from stats
    state_dim = len(stats['state_min'])
    action_dim = len(stats['action_min'])
    
    print(f"Loaded state normalization stats from {dataset_stats_path}")
    print(f"Inferred state_dim={state_dim}, action_dim={action_dim} from dataset stats")

    expert_data = SplitTrajectoryDataset(hdf5_file, BL, split='train', num_test=0)

    expert_loader = iter(DataLoader(expert_data, batch_size=BS, shuffle=True))
    H = 3
   
    #decoder = Decoder().to(device)
    #decoder.load_state_dict(torch.load('checkpoints/best_decoder_10m.pth'))
    #decoder.eval()

    transition = VideoTransformer(
        state_dim=state_dim,  # Inferred from dataset stats
        action_dim=action_dim,  # Inferred from dataset stats
        num_frames=BL-1,
        **MODEL_CONFIG
    ).to(device)
    #transition.load_state_dict(torch.load('/home/kensuke/latent-safety/scripts/checkpoints/claude_zero_wfail20500_rotvec.pth'))
    transition.load_state_dict(torch.load('checkpoints/best_classifier.pth'))


    decoder = VQVAE().to(device)
    #decoder.load_state_dict(torch.load('/home/kensuke/latent-safety/scripts/checkpoints/best_decoder_10m.pth'))
    decoder.load_state_dict(torch.load('checkpoints/testing_decoder.pth'))
    decoder.eval()


    data = next(expert_loader)
    

         

    data1 = data['cam_zed_embd'].to(device)#[transition.get_dino_features(torch.tensor(data['agentview_image_norm']).to(device))
    data2 =  data['cam_rs_embd'].to(device)#transition.get_dino_features(torch.tensor(data['robot0_eye_in_hand_image_norm']).to(device))

    inputs1 = data1[:, :-1]
    output1 = data1[:, 1:]


    inputs2 = data2[:, :-1]
    output2 = data2[:, 1:]

    data_state = data['state'].to(device)
    norm_states = normalize_states(
        data_state, state_min, state_max, q02=state_q02, q98=state_q98
    )
    states = norm_states[:, :-1]
    output_state = norm_states[:, 1:]

    data_acs = data['action'].to(device)
    norm_acs = normalize_acs(
        data_acs, action_min, action_max, q02=action_q02, q98=action_q98
    )
    acs = norm_acs[:, :-1]

    print(data.keys())



    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
        pred1, pred2, pred_state, pred_fail = transition(inputs1, inputs2, states, acs)

    
    expert_loader = iter(DataLoader(expert_data, batch_size=BS, shuffle=True))
    best_eval = float('inf')
    best_fail= float('inf')
    iters = []
    eval_iter = 100000
    transition.eval()

    confusion_matrix = torch.tensor([0,0,0,0])
    for i in tqdm(range(eval_iter), desc="Training", unit="iter"):
        #if i % 30 == 0:
        #    expert_loader = iter(DataLoader(expert_data, batch_size=BS, shuffle=True))
        #    expert_loader_eval = iter(DataLoader(expert_data_eval, batch_size=BS, shuffle=True))
        #    expert_loader_imagine = iter(DataLoader(expert_data_imagine, batch_size=1, shuffle=True))

        data = next(expert_loader)

        data1 = data['cam_zed_embd'].to(device)
        data2 =  data['cam_rs_embd'].to(device)

        inputs1 = data1[:, :-1]
        output1 = data1[:, 1:]

        inputs2 = data2[:, :-1]
        output2 = data2[:, 1:]

        data_state = data['state'].to(device)
        norm_eval_states = normalize_states(
            data_state, state_min, state_max, q02=state_q02, q98=state_q98
        )
        states = norm_eval_states[:, :-1]
        output_state = norm_eval_states[:, 1:]

        data_acs = data['action'].to(device)
        norm_acs = normalize_acs(
            data_acs, action_min, action_max, q02=action_q02, q98=action_q98
        )
        acs = norm_acs[:, :-1]
        
        pred1, pred2, pred_state, pred_fail = transition(inputs1, inputs2, states, acs)

        '''rand_inp = torch.rand_like(pred1)
        rand_out = decoder(rand_inp[:,-1])[0]
        rand_out = rand_out.detach().cpu().numpy()
        rand_out = (rand_out * 255).clip(0, 255).astype(np.uint8)

        print(rand_out.shape)
        plt.imshow(rand_out.transpose(1,2,0))
        plt.savefig('rand_out.png')
        exit()'''
        confusion_matrix += confusion(pred_fail, data['failure'][:, 1:])
        print('TP', confusion_matrix[0].item(), 'FN', confusion_matrix[1].item(), 'FP', confusion_matrix[2].item(), 'TN', confusion_matrix[3].item())
        '''with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            pred1, pred2, pred_state, pred_fail = transition(inputs1, inputs2, states, acs)
            #im1_loss = nn.MSELoss()(pred1, output1)
            #im2_loss = nn.MSELoss()(pred2, output2)
            #state_loss = nn.MSELoss()(pred_state, output_state)
            failure_loss = fail_loss(pred_fail, data['failure'][:, 1:])
            loss = failure_loss #im1_loss + im2_loss + state_loss #+ failure_loss'''
        
        
        