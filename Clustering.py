# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 23:44:20 2023

@author: coolm

use clustering to create classes and then train classification cnn on them
follow up with anomaly detection
"""
import os
import numpy as np
import math
import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest


from sklearn.cluster import KMeans
from sklearn import manifold


encoded = pd.read_csv("encoded_data.csv")

class Load_data():
    
    def __init__(self,csv_file,root_dir):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        

        img_name = os.path.join(self.root_dir,
                                self.data.iloc[idx, 0])
        
        img_err_name = img_name.replace(".csv","_err.csv")
        #print(img_name)
        image = pd.read_csv(img_name)
        image_err = pd.read_csv(img_err_name)
        
        return image,image_err
        
    def output(self):
        image = self.image
        image_err = self.image_err
        return image,image_err


class plotter():
    
    def __init__(self,mean_data):
        self.data = mean_data

        
    def __len__(self):
        return len(self.data)
    
    def clustering(self):
        label_array = np.array([42,52,62,67,90,95])
        
        for k in range(6):
            i_plot_data = self.i_data[self.i_data["target_id"] == label_array[k]]
            j_plot_data = self.j_data[self.j_data["target_id"] == label_array[k]]
            
            
            
            
            plt.plot(i_plot_data.iloc[:,1],j_plot_data.iloc[:,1],markersize=1, marker="x",  mew=2,linestyle="None")
        plt.show()

    def run(self):
        self.d = {}
        for i in range(4):
            for j in range(4):
                self.i_data = self.data.iloc[:,[1,(i+1)]]
                self.j_data =self.data.iloc[:,[1,(j+1)]]
                #print(self.i_data)
                i_name = self.i_data.columns.values.tolist()[1]
                j_name = self.j_data.columns.values.tolist()[1]
                
                
                self.d["{0}".format(i)] = self.i_data
                
                """
                if(i_name == j_name):
                    continue
                else:
                    cluster.clustering(self)
            """
            
        plotter.visual_4D(self)
        return
    
    def visual_4D(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i, i_data in self.d.items():
            globals()[f'data_{i}']=i_data
            #print(data_0)
        encoded_fig = plt.figure()
        img = ax.scatter(data_3.iloc[:,1], data_1.iloc[:,1], data_2.iloc[:,1], c=data_0.iloc[:,0], cmap="Set1")
        fig.colorbar(img)
        plt.show()
        
        



class mean_shift():
    def __init__(self, data,n_clusters):
        self.data = data
        
        x = self.data.shape[0]
        y = self.data.shape[1]
        self.n_samples = x*y
        
        self.n_clusters = n_clusters
        
        
        
        
    def gaussian(d,bw): return np.exp(-0.5*((d/bw))**2) / (bw * math.sqrt(2*math.pi))

    def distance(x,X): return np.sqrt(((x-X)**2).sum(1))
    
    def meanshift_iter(X):
        
        # Loop through every point
        for i, x in enumerate(X):
            # Find distance from point x to every other point in X
            dist = mean_shift.distance(x, X)
            # Use gaussian to turn into array of weights
            weight = mean_shift.gaussian(dist, 2.5)
            # Weighted sum (see next section for details)
            X[i] = (weight[:,None]*X).sum(0) / weight.sum()
        return X

    def meanshift(self):
        data = self.data.copy()
        print(data)
        colnames = data.iloc[:,2:5].columns.values.tolist()
        X = data.iloc[:,2:5].to_numpy()
        # Loop through a few epochs
        # A full implementation would automatically stop when clusters are stable
        for it in range(5): output = mean_shift.meanshift_iter(X)
        
        output = pd.DataFrame(output, columns=colnames)
        print(output)
        self.output = pd.concat([data.iloc[:,0:2],output],axis=1)
        print(self.output)
        
        return self.output
    
    def plot_data(self):
        run = plotter(self.data)
        run.run()
        run = plotter(self.output)
        run.run()

def convert_to_1D(In,In_err,obj,trgt,chi2):
    
    Inmax = In.max().max()
    Inmin = In.min().min()
    
    ticks = [0,1,2,3,4,5]
    
    xlabs = ["u","g","r","i","z","Y"]
    
    fig, axes = plt.subplots(nrows=2, ncols=2)
    
    ax1 = plt.subplot(2,1,1)
    ax1.imshow(In,aspect="auto",vmin=Inmin,vmax=Inmax)
    plt.title("Id " + str(obj)+"  "+ "Type "+ str(trgt))
    plt.yticks(ticks,labels=xlabs)
    plt.setp(ax1.get_xticklabels(), visible=False)
    
    ax2 = plt.subplot(2,1,2)
    im = ax2.imshow(In_err,aspect="auto",vmin=Inmin,vmax=Inmax)
    plt.title("Reduced Chi2 " + str("{:.2f}".format(chi2)))
    plt.yticks(ticks,labels=xlabs)
    plt.setp(ax2.get_xticklabels(), visible=False)
    
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.show()



#mean = mean_shift(encoded,6) 
#mean.meanshift()
#mean.plot_data()
run = plotter(encoded)
run.run()
# print(encoded.iloc[:,2:6])
# kmeans = KMeans(init='k-means++', n_clusters=8, n_init=10)
# kmeans.fit(encoded.iloc[:,2:6])
# P = kmeans.predict(encoded.iloc[:,2:6])
# P = pd.Series(P)
# print(P)
# encoded["target_id"] = P
# run = plotter(encoded)
# run.run()
# encoded = encoded.iloc[:,0:2]
# print(encoded)
# encoded.to_csv("clustered_info.csv",index=False)

tree_data = encoded.iloc[:,2:6].to_numpy()
print(tree_data)
model = IsolationForest(n_estimators=5000,max_samples='auto', contamination=float(0.1),max_features=1.0)

model.fit(tree_data)
df = pd.DataFrame()
df['scores']=model.decision_function(tree_data)
df['anomaly']=model.predict(tree_data)
print(df)
encoded["target_id"]=model.decision_function(tree_data)
run = plotter(encoded)
run.run()


expected_data  = Load_data(csv_file='C:/Users/coolm/OneDrive/Documents/Uni/Year 2/Summer Internship/Practice/ML on PLAsTICC data/info_3.csv',
                                    root_dir='cutouts',)

info = pd.read_csv('C:/Users/coolm/OneDrive/Documents/Uni/Year 2/Summer Internship/Practice/ML on PLAsTICC data/info_3.csv')


for i in range((292)):
    expec_obj = expected_data.__getitem__(i)
    #print(expec_obj)
    
    expec_data, expec_err = expec_obj
    
    object_id,target_id = info.iloc[i]
    object_id = object_id.replace(".csv","")
    anscore = df.iloc[i,0]

    if df.iloc[i,1] <= 0:
        #convert_to_1D(expec_data.transpose(), expec_err.transpose(), object_id,target_id, anscore)
        None



