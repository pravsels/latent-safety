import sys
import os

# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import yaml

from generate_data import *

from networks.helper import *
from networks.mlp import *
from networks.cnn import *


import torch.optim as optim
import numpy as np
import torch
import matplotlib.pyplot as plt

def visualize(model, x_c,y_c,r, contrastive=False, mode=None, spectral=False, safe=None, unsafe=None):

  # Create a grid of x-y coordinates
  num_pts = 201
  x = np.linspace(-1, 1, num_pts)
  y = np.linspace(-1, 1, num_pts)
  X, Y = np.meshgrid(x, y)
  coordinates = np.stack((X, Y), axis=-1)  # Combine X and Y to create (x, y) coordinate pairs

  # Convert the coordinates to a tensor
  coordinates_tensor = torch.tensor(coordinates, dtype=torch.float32)

  # Reshape the tensor to [batch_size, 3] where z is a constant value for all points
  z_value = 0.0  # You can set z to any constant value for the heatmap on the x-y plane
  coordinates_tensor = coordinates_tensor.reshape(-1, 2)
  coordinates_tensor = torch.cat((coordinates_tensor, z_value * torch.ones(coordinates_tensor.shape[0], 1)), dim=1)
  print(coordinates_tensor.size())


  # Evaluate the model for each point in the grid
  with torch.no_grad():
      model.eval()  # Set the model to evaluation mode
      scores = model(coordinates_tensor)  # Assuming z is a constant value for all points

  # Plot the heatmap
  scores = scores.reshape(num_pts, num_pts)

  # Plot the heatmap
  fig, ax = plt.subplots(figsize=(6, 6))

  axes = np.array([
        -1,
        1,
        -1,
        1,
    ])

  im = ax.imshow(
          scores.T > 0.0,
          interpolation="none",
          extent=axes,
          origin="lower",
          cmap='seismic',
          alpha=0.5,
          zorder=-1,
    )

  nx = np.shape(scores)[0]
  ny = np.shape(scores)[1]
  tn, tp, fn, fp = 0,0,0,0

  it = np.nditer(scores, flags=["multi_index"])
  while not it.finished:
    idx = it.multi_index
    x_i = x[idx[0]]
    y_i = y[idx[1]]
    score_nn = scores[idx]
    score_gt = (x_i - x_c)**2 +(y_i - y_c)**2 - r**2
    if score_nn < 0 and score_gt < 0:
      tn += 1
    if score_nn > 0 and score_gt > 0:
      tp += 1
    if score_nn < 0 and score_gt > 0:
      fn += 1
    if score_nn > 0 and score_gt < 0:
      fp += 1
    it.iternext()  


  #heatmap = plt.pcolormesh(X, Y, scores, cmap='viridis')
  #plt.colorbar(heatmap, label='Score')
  plt.xlabel('X')
  plt.ylabel('Y')

  tot = tp+fp+fn+tn
  title = 'tp: ' + str(round(100*tp/tot,2)) + '% tn: ' + str(round(100*tn/tot,2)) + '%'+' fp: ' + str(round(100*fp/tot,2)) + '%'+' fn: ' + str(round(100*fn/tot,2))+'%'
  
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
    title += 'contrastive'
  if mode == 'ring':
    title += 'ring'
  elif mode == 'local':
    title += 'local'
  if spectral:
    title += 'spectral'
  plt.savefig('logs/classifier/'+title+'.png')
  plt.show()



def train_model(x_c, y_c, r, contrastive=False, mode=None, spectral=False):
  num_pts = 1000
  if mode=='ring':
    safe, unsafe = gen_data_ring(torch.Tensor([x_c, y_c, r]), num_pts)
  elif mode == 'local':
    safe, unsafe = gen_data_local(torch.Tensor([x_c, y_c, r]), num_pts)
  else:
    safe, unsafe = gen_data(torch.Tensor([x_c, y_c, r]), num_pts)

  print(safe.size())
  print(unsafe.size())
  triplet_dataset = TripletDataset(safe[int(0.2*num_pts):], unsafe[int(0.2*num_pts):])
  eval_dataset = TripletDataset(safe[:int(0.2*num_pts)], unsafe[:int(0.2*num_pts)])

  trainloader = DataLoader(triplet_dataset, batch_size=32, shuffle=True)
  testloader = DataLoader(eval_dataset, batch_size=int(0.2*num_pts), shuffle=True)
  criterion = ContrastiveLoss(0.1)
  if spectral:
    model = SpectralMLP(3,1,256, gamma=10)
  else:
    model = MLP(3, 1, 256)

  # Example settings
  epochs = 500
  learning_rate = 0.0001

  optimizer = optim.Adam(model.parameters(), lr=learning_rate)

  # Training loop
  best_loss = np.inf
  for epoch in range(epochs):
      model.train()
      for anchors, positives, negatives in trainloader:
          optimizer.zero_grad()
          
          # Forward pass
          anchor_out = model(anchors)
          positive_out = model(positives)
          negative_out = model(negatives)
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
      for anchors, positives, negatives in testloader:
          # Forward pass
          anchor_out = model(anchors)
          positive_out = model(positives)
          negative_out = model(negatives)
          # Compute loss
          loss =  torch.sum(torch.relu(-positive_out)) #penalizes safe for being positive
          loss +=  torch.sum(torch.relu(-anchor_out)) #penalizes safe for being positive
          loss +=  2*torch.sum(torch.relu(negative_out)) # penalizes unsafe for being negative
          #if contrastive:
          #  loss += criterion(anchor_out, positive_out, negative_out)
          if loss < best_loss:
            best_loss = loss
            best_dict = model.state_dict()
          
  print(best_loss)
  model.load_state_dict(best_dict)
  return model, safe[int(0.2*num_pts):], unsafe[int(0.2*num_pts):]


def train_and_vis(x,y,r, contrastive=True, mode=None, spectral=True, path = None):
  model, safe, unsafe = train_model(x,y,r,contrastive, mode, spectral)
  visualize(model,x,y,r, contrastive, mode, spectral, safe=safe, unsafe=unsafe)
  torch.save(model.state_dict(), path)




if __name__=='__main__':
    config_path = '/home/kensuke/latent-safety/configs/config.yaml'
    with open(config_path, 'r') as file:
      config = yaml.safe_load(file)
    obs_x = config['obs_x']
    obs_y = config['obs_y']
    obs_r = config['obs_r']
    torch.manual_seed(0)
    train_and_vis(obs_x, obs_y, obs_r, contrastive=True, mode='ring', spectral=False, path = config['lx_path'])
