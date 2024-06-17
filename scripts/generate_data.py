import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import yaml
import io
from PIL import Image
import numpy as np
import pickle
def gen_data(fail, num_pts):
  length = 0
  unsafe = None
  safe = None
  while length < num_pts:
    data = torch.rand((num_pts, 3))*2 - 1
    distances_squared = (data[:, 0] - fail[0])**2 + (data[:, 1] - fail[1])**2
    indices = torch.where(distances_squared < fail[2]**2)
    check = torch.where(((data[:,0]-fail[0])**2 + (data[:,1]-fail[1])**2) < fail[2]**2)
    check2 = torch.where(((data[:,0]-fail[0])**2 + (data[:,1]-fail[1])**2) >= fail[2]**2)
    if unsafe is None:
      unsafe = data[check]
    else:
      unsafe = torch.cat([unsafe, data[check]], dim=0)
    unsafe = unsafe[:int(num_pts/2),:]

    if safe is None:
      safe = data[check2]
    else:
      safe = torch.cat([safe, data[check2]], dim=0)
    safe = safe[:int(num_pts/2),:]

    length = safe.size(0) + unsafe.size(0)
  print(safe.size(0))
  print(unsafe.size(0))
  return safe, unsafe

def gen_data_ring(fail, num_pts):
  length = 0
  unsafe = None
  safe = None
  while length < num_pts:
    data = torch.rand((num_pts, 3))*2 - 1
    distances_squared = (data[:, 0] - fail[0])**2 + (data[:, 1] - fail[1])**2
    indices = torch.where(distances_squared < fail[2]**2)
    check = torch.where(((data[:,0]-fail[0])**2 + (data[:,1]-fail[1])**2) < fail[2]**2)
    check2 = torch.where(((data[:,0]-fail[0])**2 + (data[:,1]-fail[1])**2) > (fail[2]**2)/2)
    check3 = torch.where(((data[:,0]-fail[0])**2 + (data[:,1]-fail[1])**2) > (fail[2]**2))
    check4 = torch.where(((data[:,0]-fail[0])**2 + (data[:,1]-fail[1])**2) < (fail[2]**2)*2)
    intersection_mask = torch.isin(check[0], check2[0])
    
    if unsafe is None:
      unsafe = data[check[0][intersection_mask]]
    else:
      unsafe = torch.cat([unsafe, data[check[0][intersection_mask]]], dim=0)
    unsafe = unsafe[:int(num_pts/2),:]


    intersection_mask2 = torch.isin(check3[0], check4[0])

    if safe is None:
      safe = data[check3[0][intersection_mask2]]
    else:
      safe = torch.cat([safe, data[check3[0][intersection_mask2]]], dim=0)
    safe = safe[:int(num_pts/2),:]
    length = safe.size(0) + unsafe.size(0)
  return safe, unsafe

def gen_data_local(fail, num_pts):
  length = 0
  unsafe = None
  safe = None
  while length < num_pts:
    data = torch.rand((num_pts, 3))*2 - 1
    distances_squared = (data[:, 0] - fail[0])**2 + (data[:, 1] - fail[1])**2
    indices = torch.where(distances_squared < fail[2]**2)
    check = torch.where(((data[:,0]-fail[0])**2 + (data[:,1]-fail[1])**2) < fail[2]**2)
    check2 = torch.where(((data[:,0]-fail[0])**2 + (data[:,1]-fail[1])**2) > (fail[2]**2)/2)
    check3 = torch.where(((data[:,0]-fail[0])**2 + (data[:,1]-fail[1])**2) > (fail[2]**2))
    check4 = torch.where(((data[:,0]-fail[0])**2 + (data[:,1]-fail[1])**2) < (fail[2]**2)*2)
    intersection_mask = torch.isin(check[0], check2[0])
    
    if unsafe is None:
      unsafe = data[check]
    else:
      unsafe = torch.cat([unsafe, data[check]], dim=0)
    unsafe = unsafe[:int(num_pts/2),:]


    intersection_mask2 = torch.isin(check3[0], check4[0])

    if safe is None:
      safe = data[check3[0][intersection_mask2]]
    else:
      safe = torch.cat([safe, data[check3[0][intersection_mask2]]], dim=0)
    safe = safe[:int(num_pts/2),:]
    length = safe.size(0) + unsafe.size(0)
  return safe, unsafe



def gen_data_dyn(x_min, x_max, y_min, y_max, u_max, dt, v, num_pts):
  states = torch.rand(num_pts, 3)
  states[:, 0] *= x_max - x_min
  states[:, 1] *= y_max - y_min
  states[:, 0] += x_min
  states[:, 1] += y_min 
  states[:, 2] *= 2*torch.pi
  
  states_next = torch.clone(states)
  acs = torch.rand(num_pts)*(2*u_max) - u_max
  
  states_next[:,0] += v*dt*torch.cos(states[:, 2])
  states_next[:,1] += v*dt*torch.sin(states[:, 2])
  states_next[:,2] += dt*acs

  return states, acs, states_next


def gen_data_dyn_img(x_min, x_max, y_min, y_max, u_max, dt, v, num_pts):
  dpi=64
  x_max -= 0.1
  y_max -= 0.1
  x_min += 0.1
  y_min += 0.1
  states = torch.rand(num_pts, 3)
  states[:, 0] *= x_max - x_min
  states[:, 1] *= y_max - y_min
  states[:, 0] += x_min
  states[:, 1] += y_min 
  states[:, 2] *= 2*torch.pi
  
  states_next = torch.clone(states)

  random_integers = torch.randint(0, 3, (num_pts,))

  # Map 0 to -1, 1 to 0, and 2 to 1
  mapping = torch.tensor([-u_max, 0, u_max])
  acs = mapping[random_integers]
  
  
  states_next[:,0] += v*dt*torch.cos(states[:, 2])
  states_next[:,1] += v*dt*torch.sin(states[:, 2])
  states_next[:,2] += dt*acs
  center = (0.0, 0.0)
  radius = 0.5
  init_imgs = []
  next_imgs = []
  for i in range(num_pts):
    fig,ax = plt.subplots()
    plt.xlim([-1.1, 1.1])
    plt.ylim([-1.1, 1.1])
    plt.axis('off')
    fig.set_size_inches( 1, 1 )
    # Create the circle patch
    circle = patches.Circle(center, radius, edgecolor=(1,0,0), facecolor='none')
    # Add the circle patch to the axis
    ax.add_patch(circle)
    plt.quiver(states[i, 0], states[i, 1], dt*v*torch.cos(states[i,2]), dt*v*torch.sin(states[i,2]), angles='xy', scale_units='xy', minlength=0,width=0.05, scale=0.2,color=(0,0,1), zorder=3)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    buf = io.BytesIO()
    plt.savefig('logs/tests/test.png', dpi=dpi)
    plt.savefig(buf, format='png', dpi=dpi)
    buf.seek(0)

    # Load the buffer content as an RGB image
    img = Image.open(buf).convert('RGB')
    img_array = np.array(img)
    plt.close()
    fig, ax = plt.subplots()
    plt.xlim([-1.1, 1.1])
    plt.ylim([-1.1, 1.1])
    plt.axis('off')
    fig.set_size_inches( 1, 1 )
    circle = patches.Circle(center, radius, edgecolor=(1,0,0), facecolor='none')
    # Add the circle patch to the axis
    ax.add_patch(circle)
    plt.quiver(states_next[i, 0], states_next[i, 1], dt*v*torch.cos(states_next[i,2]), dt*v*torch.sin(states_next[i,2]), angles='xy', scale_units='xy', minlength=0,width=0.05, scale=0.2,color=(0,0,1),zorder=3)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    buf2 = io.BytesIO()
    plt.savefig('logs/tests/test_next.png', dpi=dpi)
    plt.savefig(buf2, format='png', dpi = dpi)
    buf2.seek(0)

    # Load the buffer content as an RGB image
    img2 = Image.open(buf2).convert('RGB')
    img_array2 = np.array(img2)
    plt.close()

    init_imgs.append(img_array)
    next_imgs.append(img_array2)
    

  with open('datasets/dyn_data.pkl', 'wb') as f:
    pickle.dump([init_imgs, next_imgs, states, states_next, acs], f)


  return states, acs, states_next

def gen_data_ring_img(fail, num_pts):
  length = 0
  unsafe = None
  safe = None
  center = (fail[0], fail[1])
  radius = fail[2]
  dpi = 64
  while length < num_pts:
    print('length: ', length)
    data = torch.rand((num_pts, 3))*2 - 1
    distances_squared = (data[:, 0] - fail[0])**2 + (data[:, 1] - fail[1])**2
    indices = torch.where(distances_squared < fail[2]**2)
    check = torch.where(((data[:,0]-fail[0])**2 + (data[:,1]-fail[1])**2) < fail[2]**2)
    check2 = torch.where(((data[:,0]-fail[0])**2 + (data[:,1]-fail[1])**2) > (fail[2]**2)/2)
    check3 = torch.where(((data[:,0]-fail[0])**2 + (data[:,1]-fail[1])**2) > (fail[2]**2))
    check4 = torch.where(((data[:,0]-fail[0])**2 + (data[:,1]-fail[1])**2) < (fail[2]**2)*2)
    intersection_mask = torch.isin(check[0], check2[0])
    
    if unsafe is None:
      unsafe = data[check[0][intersection_mask]]
    else:
      unsafe = torch.cat([unsafe, data[check[0][intersection_mask]]], dim=0)
    unsafe = unsafe[:int(num_pts/2),:]


    intersection_mask2 = torch.isin(check3[0], check4[0])

    if safe is None:
      safe = data[check3[0][intersection_mask2]]
    else:
      safe = torch.cat([safe, data[check3[0][intersection_mask2]]], dim=0)
    safe = safe[:int(num_pts/2),:]
    length = safe.size(0) + unsafe.size(0)

  safe_imgs = []
  unsafe_imgs = []
  for i in range(safe.size(0)):
    fig,ax = plt.subplots()
    plt.xlim([-1.1, 1.1])
    plt.ylim([-1.1, 1.1])
    plt.axis('off')
    fig.set_size_inches( 1, 1 )
    # Create the circle patch
    circle = patches.Circle((center), radius, edgecolor=(1,0,0), facecolor='none')
    # Add the circle patch to the axis
    ax.add_patch(circle)
    plt.quiver(safe[i, 0], safe[i, 1], dt*v*torch.cos(safe[i,2]), dt*v*torch.sin(safe[i,2]), angles='xy', scale_units='xy', minlength=0,width=0.05, scale=0.2,color=(0,0,1), zorder=3)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    buf = io.BytesIO()
    plt.savefig('logs/tests/safe.png', dpi=dpi)
    plt.savefig(buf, format='png', dpi=dpi)
    buf.seek(0)

    # Load the buffer content as an RGB image
    img = Image.open(buf).convert('RGB')
    safe_array = np.array(img)
    plt.close()
    safe_imgs.append(safe_array)

    fig,ax = plt.subplots()
    plt.xlim([-1.1, 1.1])
    plt.ylim([-1.1, 1.1])
    plt.axis('off')
    fig.set_size_inches( 1, 1 )
    # Create the circle patch
    circle = patches.Circle(center, radius, edgecolor=(1,0,0), facecolor='none')
    # Add the circle patch to the axis
    ax.add_patch(circle)
    plt.quiver(unsafe[i, 0], unsafe[i, 1], dt*v*torch.cos(unsafe[i,2]), dt*v*torch.sin(unsafe[i,2]), angles='xy', scale_units='xy', minlength=0,width=0.05, scale=0.2,color=(0,0,1), zorder=3)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    buf = io.BytesIO()
    plt.savefig('logs/tests/unsafe.png', dpi=dpi)
    plt.savefig(buf, format='png', dpi=dpi)
    buf.seek(0)

    # Load the buffer content as an RGB image
    img = Image.open(buf).convert('RGB')
    unsafe_array = np.array(img)
    plt.close()
    unsafe_imgs.append(unsafe_array)

  with open('datasets/classifier_data.pkl', 'wb') as f:
    pickle.dump([safe_imgs, safe, unsafe_imgs, unsafe], f)

  return safe, unsafe

if __name__=='__main__':      

    config_path = '/home/kensuke/latent-safety/configs/config.yaml'
    with open(config_path, 'r') as file:
      config = yaml.safe_load(file)
    num_pts = config['num_pts']
    x_min = config['x_min']
    x_max = config['x_max']
    y_min = config['y_min']
    y_max = config['y_max']
    u_max = config['u_max']
    dt = config['dt']
    v = config['speed']
    states, acs, states_next = gen_data_dyn_img(x_min, x_max, y_min, y_max, u_max, dt, v, num_pts)
    safe, unsafe = gen_data_ring_img(torch.Tensor([0.0, 0.0, 0.5]), num_pts)
