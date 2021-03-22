#!/usr/bin/env python
# coding: utf-8

# Estefania Rossich Molina - Hebrew University of Jerusalem
# Morse Potential estimation 

#PyTorch packages

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets
from torch import optim
from torch.autograd import Variable 
from torch.utils import data

from math import e

import sklearn
from sklearn.model_selection import train_test_split
    
import numpy as np

#Other packages 
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.collections import LineCollection
from time import process_time
import time

torch.manual_seed(1) #For reproducibility
np.random.seed(1)

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[2]:


initial = 0.5; final = 4; step = 0.0001  #0.00001 gives 35,000 that I found to give good fit of Morse Potential. 

points = int((final-initial)/step)
print(points)

R_int = torch.arange(initial, final, step).view(-1, 1) 
R_int = np.array(R_int)
H1_x = - R_int/2
H2_x =  R_int/2
X = (H1_x, H2_x)
X = np.transpose(X) 
X = X. reshape(points,2)  #important line: If I don't have this line, the shape is (1,points,2)


# In[3]:


bs= 100; lr = 0.0001; epochs = 15000

print("batch size:",  bs)

#Parameters of Morse Potential for the H2 molecule
De = 4.75 #eV
beta = -1.93 #1/angstrom
re = 0.741 #angstrom

#generated interatomic radio data:

points = int((final-initial)/step) #Define as integer because it is used in the architecture of the net

#Assuming the H2 molecule has its atoms on axis x, then we use the interatomic radio to define H1x and H2x
# H1x= - H2x 
# R_int = H2x - H1x = H2x - (-H2x) = 2 H2x, thus H2x= R_int/2 and H1x = - Rint/2

class Data(Dataset):
    
    # Constructor
    def __init__(self):

        self.R_int = torch.arange(initial, final, step).view(-1, 1) 
        self.X = X
        self.y = De *(1-e**(beta*(self.R_int - re)))**2 #Morse potential: V(r) = De (1-e^(-beta(r-re)))^2
        self.y = (self.y.reshape(points,1)) 
        self.len = self.X.shape[0]
        
    # Getter
    def __getitem__(self,index):    
        return self.X[index],self.y[index]
        
    # Get Length
    def __len__(self):
        return self.len


# In[4]:


#Dataset Generation and partition into train and test

dataset = Data() #this is the dataset i feed
training_dataset, test_dataset = sklearn.model_selection.train_test_split(dataset, train_size=0.8, test_size= 0.2, random_state=35) 


# In[5]:


#dataset and dataloader

dataset = training_dataset
train_loader = DataLoader(dataset = dataset, batch_size = bs, shuffle = True, drop_last=False)


# In[6]:


#Define the model
n_inp = 2
n_h = 30
n_y = 1
print ("NN Architecture: (%d:%d:%d)" %(n_inp, n_h, n_y))

class Net(nn.Module):
    
    #Constructor
    def __init__(self):
        
        super(Net,self).__init__()   

        self.fc1 = nn.Linear( n_inp, n_h) 
        self.fc2 = nn.Linear(n_h, n_y)
                
    def forward(self, x): 

        output = F.relu(self.fc1(x)) 
        output = self.fc2(output)
         
        return output
       
net= Net()


# In[7]:


#TRAINING LOOP 

optimizer = optim.SGD(net.parameters(), lr=lr)
criterion = nn.MSELoss()

iterations = 0
losses = np.array([])
accuracies = np.array([]) 
start_time = time.time()

inputs = np.array([]) 
predictions = np.array([]) 

for e in range(epochs):
    
    for data in (train_loader): 
        epoch_time = time.time()
        
        X,y = data
        output = net(X)  
        loss = criterion(output, y)
        net.zero_grad()                      #initialize grads
        loss.backward()                      #calc grads
        optimizer.step()                     #updates
        accuracy = (output/y)
       
    if e % 1000 == 999: 
        
        print('Epoch %d' %e ,'loss %.5f' % (loss.item())) 
      
        iterations +=1
        losses = np.append(losses, loss.item())
        accuracy = accuracy.detach().numpy()
        accuracies = np.append(accuracies, accuracy)
        output = output.detach()
        
        plt.plot(X[:,1], y, 'r.', label = 'Morse ')
        plt.plot(X[:,1], output, 'g.', label = 'NN predictions ') 
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(loc='lower right')
        plt.savefig('NEW-Trained-Morse-Potential-%.d.png' % e)
        plt.title("TRAINING for Morse Potential for H2 epoch %.d" %e)
        plt.show()
        plt.clf()
        
print("--- Training time: %.2f seconds ---" % (time.time() - start_time))
print ('final loss: %.5f' %loss) 

mean = np.mean (accuracies)
std = np.std (accuracies)
print ('Mean %.3f' %mean)
print ('std %.3f' %std)


# In[8]:


##print all the weights and bias of the net:
torch.save(net.state_dict(), 'net.pth') 


# In[9]:


# Plot the expected and predicted

output = output.detach()
#plt.plot(dataset.x.numpy(), dataset.f.numpy(), 'r', label = 'Morse Potential')

plt.plot(X[:,1], y, 'r.', label = 'Morse ')
plt.plot(X[:,1], output, 'g.', label = 'Train NN predictions ') 
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='lower right')
plt.savefig('Trained-Morse-Potential-%.d.png' % points)
plt.title("Pred. during TRAINING for Morse Potential for H2")
plt.show()
plt.clf()


# In[15]:


plt.show()
plt.style.use('seaborn')
plt.plot(np.arange(iterations),losses, '-b', label='TRAIN loss')
plt.legend(loc='upper right')
plt.xlabel('N of iterations')
plt.ylabel('loss')
plt.savefig('Training-loss.png')
plt.clf() #avoids plots being overwritten


# In[11]:


net = Net()
net.load_state_dict(torch.load('net.pth'))


# In[12]:


#Test: dataset and dataloader
dataset = test_dataset
test_loader = DataLoader(dataset = dataset, batch_size = bs, shuffle = True, drop_last=False)


# In[13]:


criterion = nn.MSELoss() #this line is needed for future, when I do not train, but only load trained model
accuracies = np.array([]) 
outputs_test = np.array([]) 

for data in (test_loader): 

        X ,y = data
        output = net(X)  
        loss = criterion(output, y)
        accuracy = (output/y)
        
        accuracy = accuracy.detach().numpy()
        output = output.detach()
        
plt.plot(X[:,1], y, 'r.', label = 'Morse ')
plt.plot(X[:,1], output, 'y.', label = 'Testing preds.')  
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='lower right')
plt.savefig('NEW-Test-Morse-Potential-%.d.png' % points)
plt.title("TEST for Morse Potential for H2 epoch %.d" %e)
plt.show()
plt.clf()

mean = np.mean (accuracy)
std = np.std (accuracy)
print ('Mean %.3f' %mean)
print ('std %.3f' %std)

