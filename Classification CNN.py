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
                                str(self.labels_frame.iloc[idx, 0]))
        
        img_err_name = img_name.replace(".csv","_err.csv")
        #print(img_name)
        image = (pd.read_csv(img_name).to_numpy())
        image_err = (pd.read_csv(img_err_name).to_numpy())
        image = [image.transpose()]
        image_err = [image_err.transpose()]
        image = np.concatenate((image,image_err),axis=0).astype(np.float)
        image = image.transpose((0,1,2))
        #print(image.shape)
        labels = self.labels_frame.iloc[idx, 1:]
        labels = np.array([labels]).astype(int)
        labels[:] = [x if x != 42 else 0 for x in labels]
        labels[:] = [x if x != 52 else 1 for x in labels]
        labels[:] = [x if x != 62 else 2 for x in labels]
        labels[:] = [x if x != 67 else 3 for x in labels]
        labels[:] = [x if x != 90 else 4 for x in labels]
        labels[:] = [x if x != 95 else 5 for x in labels]
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
        image = image
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
        self.cnn = nn.Sequential(
            nn.ConstantPad2d(0,2),
            nn.Conv2d(2, 8, (2,8)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(num_features = 8),
            nn.Conv2d(8, 16, (1,5)),
            nn.ReLU(),
            nn.MaxPool2d((2)),
            nn.BatchNorm2d(16),
            
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(448, 120),
            nn.BatchNorm1d(120),

            nn.Dropout(0.33),
            nn.Linear(120,8), 
            #nn.Sigmoid()
        )
    def forward(self, x):

        x = self.cnn(x)
        return x
    





net = Net()
batch_size=64



data  = LabelsDataset(csv_file='Autoencoding/clustered_info.csv',
                                    root_dir='cutouts/',
                                    transform=ToTensor())

train_set, eval_set = train_test_split(data,train_size=0.66,random_state=42)

trainloader = DataLoader(train_set, batch_size=batch_size,
                        shuffle=True, num_workers=0)

validation_loader = DataLoader(eval_set, batch_size=batch_size,
                        shuffle=True, num_workers=0)
                                           
# print(trainloader)

classes = ("II","Iax","Ibc","Ia-91bg","Ia","SLSN-1")


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
plt.show()
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=5e-5)

all_models = np.array([])
all_optims = np.array([])


loss_store = np.array([])
vloss_store = np.array([])
acc_store = np.array([])
vacc_store = np.array([])
best_vloss = 1000000000
for epoch in range(500):  # loop over the dataset multiple times
    correct = 0
    vcorrect = 0
    running_loss = 0.0
    net = net.train()
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        labels = labels.squeeze(axis=2)
        labels = labels.long()
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        outputs = net(inputs.float())

        #outputs = outputs.flatten() 
        #outputs = outputs.long()

        labels = labels.flatten()
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        #object_type = outputs.argmax(dim=1)
        print(outputs)
        print(labels)
        outputs = outputs.detach().numpy()
        # correct_array = (np.isclose(outputs,labels,atol=0.1))
        # for j in range(len(correct_array)):
        #     if(correct_array[j] == True):
        #         correct+=1
                
        
        # print statistics
        running_loss += loss.item()
        #loss_store = np.append(loss_store,running_loss)
        
        
        
        print(running_loss)
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
    
    running_vloss = 0.0
    net = net.eval()
    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            vlabels = vlabels.long()
            voutputs = net(vinputs.float())
            #voutputs = voutputs.flatten()
            # print(voutputs)
            vlabels = vlabels.flatten()
            # print(vlabels)
            
            #print(type(vlabels))
            vloss = criterion(voutputs, vlabels)
            running_vloss += vloss
            
            #object_type = voutputs.argmax(dim=1)
            # print("object_type")
            # print(object_type)
            
            voutputs = voutputs.detach().numpy()
            # vcorrect_array = (np.isclose(voutputs,vlabels,atol=0.1))
            
            # for j in range(len(vcorrect_array)):
            #     if(vcorrect_array[j] == True):
            #         vcorrect+=1
    
    avg_vloss = running_vloss / (i + 1)    
    if(avg_vloss < best_vloss):
        best_vloss = avg_vloss
        model_path = 'Anomaly finding/Models/model_{}'.format(epoch)
        optim_path = 'Anomaly finding/Models/op_model_{}'.format(epoch)
        torch.save(net.state_dict(), model_path)
        torch.save(optimizer.state_dict(),optim_path)
        print(optimizer.state_dict())
        model_id = "model_{}".format(epoch)
        optim_id = "op_model_{}".format(epoch)
        
        all_models = np.append(all_models,model_id)
        all_optims  = np.append(all_optims,optim_id)
        
    
    
    
    loss_store = np.append(loss_store,running_loss)
    vloss_store = np.append(vloss_store,running_vloss)
    # accuracy = 100 * correct / len(train_set)
    # acc_store = np.append(acc_store,accuracy)
    # vaccuracy = 100 * vcorrect / len(eval_set)

    # vacc_store = np.append(vacc_store,vaccuracy)
    # print(vacc_store)
    plt.ylim(0,2)
    plt.plot(loss_store)
    plt.plot(vloss_store)
    plt.show()
# plt.plot(acc_store)
# plt.plot(vacc_store)
# plt.show()
np.savetxt("Anomaly finding/Models/model_info.csv", np.transpose([all_models,all_optims]),fmt="%s",delimiter=",",header="model_id,optim_id")
print('Finished Training')




















