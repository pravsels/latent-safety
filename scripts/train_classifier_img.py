import sys
import os

# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
print(sys.path)
import yaml


from generate_data import *
from networks.helper import *
from networks.mlp import *
from networks.cnn import ConvEncoderMLP



import torch.optim as optim
import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle 


def visualize(model, encoder, x_c,y_c,r, contrastive, mode, safe_img, safe, unsafe_img, unsafe):
  fig, ax = plt.subplots(figsize=(6, 6))
  tn, tp, fn, fp = 0,0,0,0
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  for i in range(len(safe_img)):
    # Evaluate the model for each point in the grid
    with torch.no_grad():
      model.eval()  # Set the model to evaluation mode
      img = torch.tensor([safe_img[i]/256.]).float().to(device)
      angle = torch.tensor(safe[i, [2]]).float().to(device)
      embed = encoder(img, angle) 
      scores = model(embed)   # Assuming z is a constant value for all points
    
    if scores > 0:
      tp += 1
      plt.quiver(safe[i,0], safe[i,1], torch.cos(safe[i,2]), torch.sin(safe[i,2]), color='green')
    else:
      fn += 1
      plt.quiver(safe[i,0], safe[i,1], torch.cos(safe[i,2]), torch.sin(safe[i,2]), color = 'purple')
  for i in range(len(unsafe_img)):
    # Evaluate the model for each point in the grid
    with torch.no_grad():
      model.eval()  # Set the model to evaluation mode
      img = torch.tensor([unsafe_img[i]/256.]).float().to(device)
      angle = torch.tensor(unsafe[i, [2]]).float().to(device)
      embed = encoder(img, angle) 
    if scores < 0:
      tn += 1
      plt.quiver(unsafe[i,0], unsafe[i,1], torch.cos(unsafe[i,2]), torch.sin(unsafe[i,2]), color='red')
    else:
      fp += 1
      plt.quiver(unsafe[i,0], unsafe[i,1], torch.cos(unsafe[i,2]), torch.sin(unsafe[i,2]), color = 'blue')

  axes = np.array([
        -1,
        1,
        -1,
        1,
    ])

  plt.xlabel('X')
  plt.ylabel('Y')

  tot = tp+fp+fn+tn
  title = 'tp: ' + str(tp) + ' tn: ' + str(tn) + ' fp: ' + str(fp) + ' fn: ' + str(fn)
  
  plt.title(title)
  #plt.contour(X, Y, scores, levels=[0], colors='black', linewidths=2, label='learned boundary')

  circle = plt.Circle((x_c, y_c), r, fill=False, color='blue', label = 'GT boundary')

  # Add the circle to the plot
  ax.add_patch(circle)
  ax.set_aspect('equal')
  plt.legend()

  #if safe is not None:
  #  plt.scatter(safe[:,0], safe[:,1], c = 'r')
  #  plt.scatter(unsafe[:,0], unsafe[:,1], c = 'b')

  title = ''
  if contrastive:
    title += '_contrastive'
  if mode == 'ring':
    title += '_ring'
  elif mode == 'local':
    title += '_local'
  plt.savefig('logs/classifier_img/'+title+'.png')
  plt.show()



def train_model(config, contrastive=False, mode=None):
  with open(config['lx_img_data'], 'rb') as f:
    dataset = pickle.load(f)
  safe_img,safe, unsafe_img, unsafe = dataset
  num_pts = len(safe_img)

  print(safe.size())
  print(unsafe.size())
  num_eval = int(0.2*num_pts)
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  triplet_dataset = ImgTripletDataset(safe_img[num_eval:], safe[num_eval:], unsafe_img[num_eval:], unsafe[num_eval:])
  eval_dataset = ImgTripletDataset(safe_img[:num_eval], safe[:num_eval], unsafe_img[:num_eval], unsafe[:num_eval])

  trainloader = DataLoader(triplet_dataset, batch_size=32, shuffle=True)
  testloader = DataLoader(eval_dataset, batch_size=int(0.2*num_pts), shuffle=True)
  criterion = ContrastiveLoss(0.1)

  act = 'SiLU'
  norm = True
  cnn_depth = 32
  kernel_size = 4 
  minres = 4
  img_size = 64
  input_shape = (img_size, img_size, 3)
  x_dim = 3 # x, y, cos(theta), sin(theta)
  u_dim = 1
  hidden_dim = 256
  #encoder = ConvEncoder(input_shape, cnn_depth, act, norm, kernel_size, minres)
  encoder = ConvEncoderMLP(input_shape, cnn_depth, act, norm, kernel_size, minres, out_dim = x_dim, in_dim=1, hidden_dim=hidden_dim, hidden_layer=2).to(device)
  encoder.load_state_dict(torch.load('logs/dynamics_img/encoder_img.pth'))

  model = MLP(x_dim, 1, hidden_dim).to(device)

  # Example settings
  epochs = 100
  learning_rate = 0.0001

  optimizer = optim.Adam(model.parameters(), lr=learning_rate)

  # Training loop
  best_loss = np.inf
  for epoch in range(epochs):
      model.train()
      for a_i, a_s, p_i, p_s, n_i, n_s in trainloader:
          optimizer.zero_grad()
          a_i = a_i.to(device)
          p_i = p_i.to(device)
          n_i = n_i.to(device)
          a_s = a_s.to(device)
          p_s = p_s.to(device)
          n_s = n_s.to(device)
          # Forward pass
          anchor_out = model(encoder(a_i/256., a_s[:, 2]).detach())
          positive_out = model(encoder(p_i/256., p_s[:, 2]).detach())
          negative_out = model(encoder(n_i/256., n_s[:, 2]).detach())
          # Compute loss
    
          loss =  torch.sum(torch.relu(-positive_out)) #penalizes safe for being positive
          loss += torch.sum(torch.relu(-anchor_out)) #penalizes safe for being positive
          loss +=  2*torch.sum(torch.relu(negative_out)) # penalizes unsafe for being negative
          
          if contrastive:
            crit_loss = criterion(anchor_out, positive_out, negative_out)
            loss+= crit_loss

          # Backward pass and optimization
          loss.backward()
          optimizer.step()
      model.eval()
      for a_i, a_s, p_i, p_s, n_i, n_s in testloader:
          # Forward pass
          # Forward pass
          a_i = a_i.to(device)
          p_i = p_i.to(device)
          n_i = n_i.to(device)
          a_s = a_s.to(device)
          p_s = p_s.to(device)
          n_s = n_s.to(device)
          # Forward pass
          anchor_out = model(encoder(a_i/256., a_s[:, 2]).detach())
          positive_out = model(encoder(p_i/256., p_s[:, 2]).detach())
          negative_out = model(encoder(n_i/256., n_s[:, 2]).detach())
          # Compute loss
    
          loss =  torch.sum(torch.relu(-positive_out)) #penalizes safe for being positive
          loss += torch.sum(torch.relu(-anchor_out)) #penalizes safe for being positive
          loss +=  2*torch.sum(torch.relu(negative_out)) # penalizes unsafe for being negative
          #if contrastive:
          #  loss += criterion(anchor_out, positive_out, negative_out)
          if loss < best_loss:
            best_loss = loss
            best_dict = model.state_dict()
      print('epoch: ', epoch, 'loss: ', best_loss.item())
          
  print(best_loss)
  model.load_state_dict(best_dict)
  return model, encoder, safe_img[:num_eval], safe[:num_eval], unsafe_img[:num_eval], unsafe[:num_eval]


def train_and_vis(x,y,r, config, contrastive=True, mode=None):
  model, encoder, safe_img, safe, unsafe_img, unsafe = train_model(config, contrastive, mode)
  visualize(model, encoder, x,y,r, contrastive, mode, safe_img, safe, unsafe_img, unsafe)
  torch.save(model.state_dict(), config['lx_img_path'])




if __name__=='__main__':
  config_path = '/home/kensuke/latent-safety/configs/config.yaml'
  with open(config_path, 'r') as file:
    config = yaml.safe_load(file)
  obs_x = config['obs_x']
  obs_y = config['obs_y']
  obs_r = config['obs_r']
  path = config['lx_img_path']
    
  train_and_vis(obs_x,obs_y, obs_r, config, contrastive=True, mode='ring')
