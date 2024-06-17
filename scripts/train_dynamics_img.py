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
import pickle


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



      plt.savefig('logs/dynamics_img/'+title+'.png')
      plt.show()

def train_dyn(x_min, x_max, y_min, y_max, u_max, dt, v):
  

  with open('datasets/dyn_data.pkl', 'rb') as f:
    dataset = pickle.load(f)
  i,i_n, s, s_n, a = dataset
  num_pts = len(i)

  num_eval = int(0.2*num_pts)
  print(num_pts)
  train_dataset = ImgTransitionDataset(i[num_eval:], i_n[num_eval:], s[num_eval:], s_n[num_eval:], a[num_eval:])
  eval_dataset = ImgTransitionDataset(i[:num_eval], i_n[:num_eval], s[:num_eval], s_n[:num_eval], a[:num_eval])


  trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
  testloader = DataLoader(eval_dataset, batch_size=int(0.2*num_pts), shuffle=True)
  
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  
  

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
  dynamics = MLP(x_dim+u_dim, x_dim, hidden_dim).to(device)

  #print(encoder.outdim)
  # Example settings
  epochs = 1000
  learning_rate = 0.0001

  optimizer = optim.Adam(list(encoder.parameters()) + list(dynamics.parameters()), lr=learning_rate)

  # Training loop
  best_loss = np.inf
  loss_fn = nn.MSELoss()
  for epoch in range(epochs):
      print('epoch: ', epoch)
      encoder.train()
      dynamics.train()
      print('training')
      for i, i_n, s,s_n, a in trainloader:
          optimizer.zero_grad()
          #print(i.size())
          i = i.to(device)
          i_n = i_n.to(device)
          s = s.to(device)
          a = a.to(device)/u_max
          s_n = s_n.to(device)
        
          # observation reconstruction
          latent = encoder(i/256., s[:,2])
          latent_n = encoder(i_n/256., s_n[:,2])
          recon_loss = loss_fn(latent, s) + loss_fn(latent_n, s_n) #reconstruct gt state from img
          
          # dynamics prediction
          target = latent_n - latent
          target = latent_n - latent
          target[:, 0] /= dt*v
          target[:, 1] /= dt*v
          target[:, 2] /= dt*u_max 
          inp = torch.cat([latent.detach(), a.unsqueeze(1)], dim=1)
          latent_pred = dynamics(inp)          
          dyn_loss =  loss_fn(latent_pred, target.detach())
          # Backward pass and optimization
          
          loss = recon_loss + dyn_loss 
          #print('recon_loss: ', recon_loss.item())#, 'dyn_loss: ', dyn_loss.item())
          loss.backward()
          optimizer.step()
      encoder.eval()
      dynamics.eval()      
      print('eval')
      eval_loss = 0
      for i, i_n, s,s_n, a in testloader:
        i = i.to(device)
        i_n = i_n.to(device)
        s = s.to(device)
        a = a.to(device)/u_max
        s_n = s_n.to(device)
        latent = encoder(i/256., s[:,2])
        latent_n = encoder(i_n/256., s_n[:,2])

        #s_target = torch.cat([s[:,:2], torch.cos(s[:,2]).unsqueeze(1), torch.sin(s[:,2]).unsqueeze(1)], dim=1).to(device)
        #s_n_target = torch.cat([s_n[:,:2], torch.cos(s_n[:,2]).unsqueeze(1), torch.sin(s_n[:,2]).unsqueeze(1)], dim=1).to(device)
        recon_loss = loss_fn(latent, s) + loss_fn(latent_n, s_n) #reconstruct gt state from img
        
        
        # Forward pass
        # target is normalized difference
        target = latent_n - latent
        target[:, 0] /= dt*v
        target[:, 1] /= dt*v
        target[:, 2] /= dt*u_max 

        
        inp = torch.cat([latent,a.unsqueeze(1)], dim=1)
        latent_pred = dynamics(inp)          
        # Compute loss
        dyn_loss =  loss_fn(latent_pred.detach(), target.detach())
        # Backward pass and optimization
        eval_loss += recon_loss+ dyn_loss
        print('eval recon_loss: ', recon_loss.item(), 'eval dyn_loss: ', dyn_loss.item())

      if eval_loss < best_loss:
        best_loss = eval_loss
        best_enc_dict = encoder.state_dict()
        best_dyn_dict = dynamics.state_dict()
      print('eval_loss: ', eval_loss )
          
  print(best_loss)
  encoder.load_state_dict(best_enc_dict)
  dynamics.load_state_dict(best_dyn_dict)
  return encoder, dynamics


def train_and_vis(x_min, x_max, y_min, y_max, u_max, dt, v):
  encoder, dynamics = train_dyn(x_min, x_max, y_min, y_max, u_max, dt, v)
  visualize(dynamics, u_max, v, dt)
  torch.save(dynamics.state_dict(), 'logs/dynamics_img/dynamics_img.pth')
  torch.save(encoder.state_dict(), 'logs/dynamics_img/encoder_img.pth')


  
if __name__=='__main__':
    dt = 0.05
    u_max = 1.25
    x_min = -1.1
    x_max = 1.1
    y_min = -1.1
    y_max = 1.1
    v = 1
    torch.manual_seed(0)
    train_and_vis(x_min, x_max, y_min, y_max, u_max, dt, v)
