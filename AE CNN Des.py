# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 15:58:38 2023

@author: coolm
"""

import os
import torch
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split


plt.ion()   # interactive mode


class LabelsDataset(Dataset):
    """Face labels dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.labels_frame.iloc[idx, 0])
        img_name = str(img_name) + ".csv"
        
        img_err_name = img_name.replace(".csv","_err.csv")
        #print(img_name)
        image = (pd.read_csv(img_name).to_numpy())
        image_err = (pd.read_csv(img_err_name).to_numpy())
        image = [image.transpose()]
        image_err = [image_err.transpose()]
        image = np.concatenate((image,image_err),axis=0).astype(np.float)
        image = image.transpose((0,1,2))
        
        #image = image.reshape(12*128)

        
        labels = self.labels_frame.iloc[idx, 1:]
        labels = np.array([labels]).astype(int)
        labels[:] = [x if x != 42 else 0 for x in labels]
        labels[:] = [x if x != 52 else 0 for x in labels]
        labels[:] = [x if x != 62 else 0 for x in labels]
        labels[:] = [x if x != 67 else 0 for x in labels]
        labels[:] = [x if x != 90 else 1 for x in labels]
        labels[:] = [x if x != 95 else 0 for x in labels]
        #labels = labels.astype('float').reshape(-1, 2)
        sample = {'image': image, 'labels': labels}

        if self.transform:
            sample = self.transform(sample)

        return sample
    



class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        
        #print(image.shape)
        return (torch.from_numpy(image),
                torch.from_numpy(labels))





def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    #print(npimg)
    plt.imshow(npimg,aspect="auto")
    


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.ConstantPad2d(0,2),
            nn.Conv2d(128, 64, (1,2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 8, (1,2)),
            #nn.BatchNorm2d(16),
            nn.ReLU(),
            
            #nn.Conv2d(32, 16, (1,1)),
            #nn.BatchNorm2d(64),
            #nn.ReLU(),
            nn.MaxPool2d(2,return_indices=True),
            
            
        )
        
        self.flat = nn.Flatten()
        self.unflat = nn.Unflatten(1, [8,1,1])
        self.unpool = nn.MaxUnpool2d(2)
        
        
        self.decoder = nn.Sequential(
            
            #torch.Tensor.reshape(64,64,6,113),
            
            #nn.ConvTranspose2d(8, 16, (1,1)),
            #nn.BatchNorm2d(16),
            #nn.ReLU(),
            
            nn.ConvTranspose2d(8, 64, (1,2)),
            #nn.BatchNorm2d(8),
            #nn.ReLU(),
            
            nn.ConvTranspose2d(64, 128, (1,2)),
            #nn.BatchNorm2d(2),
            #nn.ReLU(),
            #nn.Dropout(0.33),
            #nn.Sigmoid()
        )
        
    def forward(self, x):
        encoded, indices = self.encoder(x)
        print(encoded.size())
        #print(encoded)
        flat_encoded = self.flat(encoded)
        #print(flat_encoded.size())
        unflat_encoded = self.unflat(flat_encoded)
        
        unpooled = self.unpool(unflat_encoded,indices)#,output_size=(64,8,1,2))
        
        decoded = self.decoder(unpooled)
        #print(decoded)
        return decoded


def convert_to_1D(In,In_err,Out,Out_err):
    
    Inmax = In.max().max()
    Inmin = In.min().min()
    Outmax = Out.max().max()
    Outmin = Out.min().min()
    
    if(Outmin >= Inmin):
        data_min = Outmin
        
    else:
        data_min = Inmin
        
    if(Outmax >= Inmax):
        data_max = Outmax
        
    else:
        data_max = Inmax
    
    
    ticks = [0,1,2,3,4,5]
    
    xlabs = ["u","g","r","i","z","Y"]
    
    fig, axes = plt.subplots(nrows=2, ncols=2)
    
    ax1 = plt.subplot(2,2,1)
    ax1.imshow(In,aspect="auto",vmin=data_min, vmax=data_max)
    plt.title("Original Image")
    plt.yticks(ticks,labels=xlabs)
    plt.setp(ax1.get_xticklabels(), visible=False)
    
    ax2 = plt.subplot(2,2,2)
    ax2.imshow(In_err,aspect="auto",vmin=data_min, vmax=data_max)
    plt.title("Error of Image")
    plt.yticks(ticks,labels=xlabs)
    plt.setp(ax2.get_xticklabels(), visible=False)
    
    ax3 = plt.subplot(2,2,3)
    ax3.imshow(Out,aspect="auto",vmin=data_min, vmax=data_max)
    plt.title("Decoded Image")
    plt.yticks(ticks,labels=xlabs)
    
    ax4 = plt.subplot(2,2,4)
    im = ax4.imshow(Out_err,aspect="auto",vmin=data_min, vmax=data_max)
    plt.title("Error of Image")
    plt.yticks(ticks,labels=xlabs)
    
    
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.show()
    


net = Net()
batch_size=128







data  = LabelsDataset(csv_file='C:/Users/coolm/OneDrive/Documents/Uni/Year 2/Summer Internship/DES/cutouts/info.csv',
                                    root_dir='C:/Users/coolm/OneDrive/Documents/Uni/Year 2/Summer Internship/DES/cutouts/',
                                    transform=ToTensor())

train_set, eval_set = train_test_split(data,train_size=0.66,random_state=42)

trainloader = DataLoader(train_set, batch_size=batch_size,
                        shuffle=True, num_workers=0)

validation_loader = DataLoader(eval_set, batch_size=batch_size,
                        shuffle=True, num_workers=0)

# print(trainloader)

classes = ("II","Iax","Ibc","Ia-91bg","Ia","SLSN-1")
"""
# get some random training images
dataiter = iter(trainloader)
#print(type(dataiter))
images, labels = next(dataiter)
#print(images.shape)
for i in range(4):
    image = images[i,0]
    # show images
    plt.subplot(2,4,i+1)
    imshow((image))
    # print labels
    #print(' '.join(f'{classes[i]}'))
for i in range(4):
    image = images[i,1]
    # show images
    plt.subplot(2,4,i+5)
    imshow((image))
    # print labels
"""



plt.show()
net = Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=5e-8)

all_models = np.array([])
all_optims = np.array([])


loss_store = np.array([])
vloss_store = np.array([])
acc_store = np.array([])
vacc_store = np.array([])
best_vloss = 1000000000
for epoch in range(5000):  # loop over the dataset multiple times
    print(epoch)
    correct = 0
    vcorrect = 0
    running_loss = 0.0
    net = net.train()
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        labels = labels.squeeze(axis=2)
        labels = labels.float()
        # zero the parameter gradients
        optimizer.zero_grad()
        
        inputs = inputs.float()
        
        #print(inputs)
        #print(inputs.size())
        inputs = inputs.transpose(1,3).transpose(2,3)  #(64,128,2,6)
        
        
        
        # forward + backward + optimize
        outputs = net(inputs)
        
        inputs = inputs.transpose(2,3).transpose(1,3)
        
        inputs_shape = inputs.numpy().shape

        no_inputs = inputs_shape[0]
        
        
        outputs = outputs.transpose(2,3).transpose(1,3)
        
        #print(inputs.size())
        #print(outputs.size())
        
        
        
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        
        #object_type = outputs.argmax(dim=1)
        
        outputs = outputs.detach().numpy()
        correct_array = (np.isclose(outputs,inputs,atol=0.1))
        #print(correct_array)
        # for j in range(len(correct_array)):
        #     if(correct_array[0,0,0,j] == True):
        #         correct+=1
                
        
        # print statistics
        running_loss += loss.item()
        #loss_store = np.append(loss_store,running_loss)
        
        
        
        #print(running_loss)
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
    
    running_vloss = 0.0
    net = net.eval()
    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            vlabels = vlabels.float()
            
            vinputs = vinputs.transpose(1,3).transpose(2,3)  #(64,128,2,6)
            
            
            
            # forward + backward + optimize
            voutputs = net(vinputs.float())
            
            vinputs = vinputs.transpose(2,3).transpose(1,3)
            
            inputs_shape = vinputs.numpy().shape

            no_inputs = inputs_shape[0]
            
            
            voutputs = voutputs.transpose(2,3).transpose(1,3)
            #print(voutputs)
            
            #print(type(vlabels))
            vloss = criterion(voutputs, vinputs)
            running_vloss += vloss
            
            #object_type = voutputs.argmax(dim=1)
            # print("object_type")
            # print(object_type)
            
            voutputs = voutputs.detach().numpy()
            vcorrect_array = (np.isclose(voutputs,vinputs,atol=0.1))
            
            # for j in range(len(vcorrect_array)):
            #     if(vcorrect_array[0,0,0,j] == True):
            #         vcorrect+=1
    
    avg_vloss = running_vloss / (i + 1)    
    if(avg_vloss < best_vloss):
        best_vloss = avg_vloss
        model_path = 'Models/model_{}'.format(epoch)
        optim_path = 'Models/op_model_{}'.format(epoch)
        torch.save(net.state_dict(), model_path)
        torch.save(optimizer.state_dict(),optim_path)
        #print(optimizer.state_dict())
        model_id = "model_{}".format(epoch)
        optim_id = "op_model_{}".format(epoch)
        
        all_models = np.append(all_models,model_id)
        all_optims  = np.append(all_optims,optim_id)
        
        
        I_in = vinputs.detach().numpy()[0,0,:,:]
        I_in_err = vinputs.detach().numpy()[0,1,:,:]
        
        
        I_out = voutputs[0,0,:,:]
        I_out_err = voutputs[0,1,:,:]
        convert_to_1D(I_in, I_in_err, I_out, I_out_err)
        
    
    
    loss_store = np.append(loss_store,running_loss)
    vloss_store = np.append(vloss_store,running_vloss)
    accuracy = 100 * correct / len(train_set)
    acc_store = np.append(acc_store,accuracy)
    vaccuracy = 100 * vcorrect / len(eval_set)

    vacc_store = np.append(vacc_store,vaccuracy)
    #print(vacc_store)
    #plt.ylim(0,2)
    plt.plot(loss_store)
    #plt.plot(vloss_store)
    plt.show()
plt.plot(acc_store)
plt.plot(vacc_store)
plt.show()
np.savetxt("New Models/model_info.csv", np.transpose([all_models,all_optims]),fmt="%s",delimiter=",",header="model_id,optim_id")
print('Finished Training')





















