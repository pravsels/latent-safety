import sys
import os

# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)


from generate_data import *
from networks.helper import *
from networks.mlp import *
from networks.cnn import *

import torch.optim as optim
import numpy as np
import torch
import matplotlib.pyplot as plt
import yaml



def visualize(model,u_max, v, dt):
  # Create a grid of x-y coordinates
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  for theta in [0, np.pi/3, np.pi/2]:
    for u in [-u_max, 0, u_max]:
      num_pts = 20
      x = np.linspace(-1, 1, num_pts)
      y = np.linspace(-1, 1, num_pts)
      X, Y = np.meshgrid(x, y)
      coordinates = np.stack((X, Y), axis=-1)  # Combine X and Y to create (x, y) coordinate pairs

      # Convert the coordinates to a tensor
      coordinates_tensor = torch.tensor(coordinates, dtype=torch.float32)

      # Reshape the tensor to [batch_size, 3] where z is a constant value for all points
      coordinates_tensor = coordinates_tensor.reshape(-1, 2)
      coordinates_tensor = torch.cat((coordinates_tensor, theta * torch.ones(coordinates_tensor.shape[0], 1)), dim=1)
      coordinates_tensor = torch.cat((coordinates_tensor, u/u_max * torch.ones(coordinates_tensor.shape[0], 1)), dim=1)

      # Evaluate the model for each point in the grid
      with torch.no_grad():
          model.eval()  # Set the model to evaluation mode
          states = model(coordinates_tensor.to(device)).detach().cpu()  # Assuming z is a constant value for all points
          states[:,:2] *= v*dt
          states[:,2] *= u_max*dt
          states += coordinates_tensor[:,:3]


          ctens2 = torch.cat((states, u/u_max * torch.ones(coordinates_tensor.shape[0], 1)), dim=1)
          states2 = (model(ctens2.to(device)).detach().cpu())
          states2[:,:2] *= v*dt
          states2[:,2] *= u_max*dt
          states2 += states[:,:3]
          states=states.numpy()
          states2=states2.numpy()

      
      # Plot the heatmap
      # Plot the heatmap
      fig, ax = plt.subplots(figsize=(6, 6))

      plt.quiver(coordinates_tensor[:,0], coordinates_tensor[:,1], np.cos(coordinates_tensor[:,2]), np.sin(coordinates_tensor[:,2]), color='r')
      plt.quiver(states[:,0], states[:,1], np.cos(states[:,2]), np.sin(states[:,2]), color='b')
      plt.quiver(states2[:,0], states2[:,1], np.cos(states2[:,2]), np.sin(states2[:,2]), color='b')
      #plt.scatter(coordinates_tensor[:,0], coordinates_tensor[:,1], c='b')
      #plt.scatter(states[:,0], states[:,1], c='r', marker='*')
      plt.xlabel('X')
      plt.ylabel('Y')

      title = 'eval_dyn, theta= pi*'+str(round(theta/np.pi,2))+', u=' +str(u)
      
      plt.title(title)


      # Add the circle to the plot

      plt.legend()



      plt.savefig('logs/dynamics/'+title+'.png')
      plt.show()

def train_dyn(x_min, x_max, y_min, y_max, u_max, dt, v, num_pts):
  x_dim = 3
  u_dim = 1

  s,a,sn = gen_data_dyn(x_min, x_max, y_min, y_max, u_max, dt, v, num_pts)
  

  train_dataset = TransitionDataset(s[int(0.2*num_pts):],a[int(0.2*num_pts):],sn[int(0.2*num_pts):])
  eval_dataset = TransitionDataset(s[:int(0.2*num_pts)],a[:int(0.2*num_pts)],sn[:int(0.2*num_pts)])

  trainloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
  testloader = DataLoader(eval_dataset, batch_size=int(0.2*num_pts), shuffle=True)
  
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model = MLP(x_dim+u_dim, x_dim, 256).to(device)

  # Example settings
  epochs = 1000
  learning_rate = 0.0001

  optimizer = optim.Adam(model.parameters(), lr=learning_rate)

  # Training loop
  best_loss = np.inf
  loss_fn = nn.MSELoss()
  for epoch in range(epochs):
      print('epoch: ', epoch)
      model.train()
      print('training')
      for s, a, s_next in trainloader:
          optimizer.zero_grad()
          s = s.to(device)
          a = a.to(device)/u_max
          s_next = s_next.to(device)
          # Forward pass
          # target is normalized difference
          target = s_next - s
          target[:, 0] /= dt*v
          target[:, 1] /= dt*v
          target[:, 2] /= dt*u_max 

          inp = torch.cat([s,a.unsqueeze(1)], dim=1)
          pred = model(inp)
          # Compute loss
          loss =  loss_fn(pred, target)
          # Backward pass and optimization
          loss.backward()
          optimizer.step()
      model.eval()
      print('eval')
      eval_loss = 0
      for s, a, s_next in testloader:
        s = s.to(device)
        a = a.to(device)/u_max
        s_next = s_next.to(device)
        # target is normalized difference
        target = s_next - s
        target[:, 0] /= dt*v
        target[:, 1] /= dt*v
        target[:, 2] /= dt*u_max
        inp = torch.cat([s,a.unsqueeze(1)], dim=1)
        pred = model(inp)
        # Compute loss
        eval_loss +=  loss_fn(pred, target)

      if loss < best_loss:
        best_loss = loss
        best_dict = model.state_dict()
      print('eval_loss: ', loss )
          
  print(best_loss)
  model.load_state_dict(best_dict)
  return model


def train_and_vis(x_min, x_max, y_min, y_max, u_max, dt, v, num_pts, path):
  model = train_dyn(x_min, x_max, y_min, y_max, u_max, dt, v, num_pts)
  visualize(model, u_max, v, dt)
  torch.save(model.state_dict(), path)


  
if __name__=='__main__':
    config_path = '/home/kensuke/latent-safety/configs/config.yaml'
    with open(config_path, 'r') as file:
      config = yaml.safe_load(file)
    
    x_min = config['x_min']
    x_max = config['x_max']
    y_min = config['y_min']
    y_max = config['y_max']
    u_max = config['u_max']
    dt = config['dt']
    v = config['speed']
    dt = config['dt']
    path = config['dyn_path']
  
    num_pts = config['num_pts']
    torch.manual_seed(0)
    train_and_vis(x_min, x_max, y_min, y_max, u_max, dt, v, num_pts, path)
