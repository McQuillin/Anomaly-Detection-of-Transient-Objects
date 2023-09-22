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
        
        img_err_name = img_name.replace(".csv","_err.csv")
        #print(img_name)
        image = (pd.read_csv(img_name).to_numpy())
        image_err = (pd.read_csv(img_err_name).to_numpy())
        image = [image.transpose()]
        image_err = [image_err.transpose()]
        image = np.concatenate((image,image_err),axis=0).astype(np.float)
        image = image.transpose((0,1,2))
        
        #image = image.reshape(12*128)
        print(image.shape)
        
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
            
            nn.Conv2d(64, 16, (1,2)),
            #nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.Conv2d(16, 2, (1,1)),
            #nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,return_indices=True),
            
            
        )
        
        self.flat = nn.Flatten()
        self.unflat = nn.Unflatten(1, [2,1,2])
        self.unpool = nn.MaxUnpool2d(2)
        
        
        self.decoder = nn.Sequential(
            
            #torch.Tensor.reshape(64,64,6,113),
            
            nn.ConvTranspose2d(2, 16, (1,1)),
            #nn.BatchNorm2d(16),
            #nn.ReLU(),
            
            nn.ConvTranspose2d(16, 64, (1,2)),
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
        print(flat_encoded.size())
        unflat_encoded = self.unflat(flat_encoded)
        
        unpooled = self.unpool(unflat_encoded,indices)#,output_size=(64,8,1,2))
        
        decoded = self.decoder(unpooled)
        #print(decoded)
        return flat_encoded,decoded


def convert_to_1D(In,In_err,Out,Out_err,obj,trgt):
    
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
    plt.title("Id " + str(obj)+"  "+ "Type "+ str(trgt))
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
batch_size=20



data  = LabelsDataset(csv_file='C:/Users/coolm/OneDrive/Documents/Uni/Year 2/Summer Internship/Practice/ML on PLAsTICC data/info_3.csv',
                                    root_dir='C:/Users/coolm/OneDrive/Documents/Uni/Year 2/Summer Internship/Practice/ML on PLAsTICC data/cutouts',
                                    transform=ToTensor())

info_df = pd.read_csv('C:/Users/coolm/OneDrive/Documents/Uni/Year 2/Summer Internship/Practice/ML on PLAsTICC data/info_3.csv')

validation_loader = DataLoader(data, batch_size=batch_size,
                        shuffle=False, num_workers=0)

# print(trainloader)

classes = ("II","Iax","Ibc","Ia-91bg","Ia","SLSN-1")



plt.show()
net = Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=5e-8)

net = net.eval()

all_models = np.array([])
all_optims = np.array([])



#model_info = pd.read_csv("AE Models/model_info.csv")
#print(model_info)

# model_ids = model_info["model_id"]
# optim_ids = model_info["optim_id"]





model = "model_1310"
optim = "op_model_1310"
    
PATH = "Saved models/{}".format(model)
optim_PATH = "Saved models/{}".format(optim)
    
net.load_state_dict(torch.load(PATH))

optim_model = torch.load(optim_PATH)
optimizer.load_state_dict(optim_model)

columns = np.array([])


for i in range(4):
    column_num = "column_{}".format(i+1)
    columns = np.append(columns,column_num)

#print(columns)
data_store = pd.DataFrame(columns = columns)

with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            vlabels = vlabels.float()
            
            vinputs = vinputs.transpose(1,3).transpose(2,3)  #(64,128,2,6)
            
            
            
            # forward + backward + optimize
            encoded, voutputs = net(vinputs.float())
            
            vinputs = vinputs.transpose(2,3).transpose(1,3)
            
            inputs_shape = vinputs.numpy().shape

            no_inputs = inputs_shape[0]
            
            
            voutputs = voutputs.transpose(2,3).transpose(1,3)
            #print(voutputs)
            
            #print(type(vlabels))
            #vloss = criterion(voutputs, vinputs)
            
            
            #object_type = voutputs.argmax(dim=1)
            # print("object_type")
            # print(object_type)
            
            vinputs = vinputs.detach().numpy()
            voutputs = voutputs.detach().numpy()
            encoded = encoded.detach().numpy()
            
            encoded_df = pd.DataFrame(encoded,columns=columns)
            
            
            #vcorrect_array = (np.isclose(voutputs,vinputs,atol=0.1))
            #print(vinputs[0])
            #print(vinputs[1])

            
            for j in range(len(voutputs)):
                
                
                
                I_in = vinputs[j,0]
                I_in_err = vinputs[j,1]
            
            
                I_out = voutputs[j,0]
                I_out_err = voutputs[j,1]
                
                
                
                
                data_df = pd.DataFrame(voutputs[j,0])#,columns=["data_0","data_1","data_2","data_3","data_4","data_5"])
                print(data_df)
                data_err_df = pd.DataFrame(voutputs[j,1])
                print(data_err_df)
                
                data_df = data_df.transpose()
                
                data_err_df = data_err_df.transpose()
                
                j = 20*i + j
                
                k = info_df.iloc[j,0]
                l = info_df.iloc[j,1]
                
                convert_to_1D(I_in, I_in_err, I_out, I_out_err,k,l)
                
                data_df.to_csv("cutouts/{0}".format(k),index=False)
                
                k = k.replace(".csv","_err.csv")
                
                #data_err_df.to_csv("cutouts/{0}".format(k),index=False)
            data_store = pd.concat([data_store,encoded_df],ignore_index=True)


print(info_df)
print(data_store)
data_store = pd.concat([info_df,data_store],axis = 1,ignore_index=False)
                
print(data_store)
        

data_store.to_csv("encoded_data.csv",index=False)




















