# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 10:06:08 2023

@author: coolm
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fulu


def load_data():
    info = pd.read_csv("Des data/DES data.csv")
    colnames = info.columns.values.tolist()
    print(colnames)
    d = {}
    
    for i in range(int(len(colnames)/2)):
        col = colnames[2*i+1]
        info_col = colnames[2*i+2]
        print(col)
        d["{0}".format(col)] = info.iloc[:,[(2*i+1),(2*i+2)]]
        #d["{0}".format(info_col)] = info.iloc[:,2*i+2]
        print(d["{0}".format(col)])
        #print(d["{0}".format(info_col)])
    return d
        

class gp():
    
    def __init__(self,data_dict):
        self.data_dict = data_dict
        
        
    def load_datafile(self):
        all_objs = np.array([])
        all_trgts = np.array([])
        for key,data in self.data_dict.items():
            datafiles = data.iloc[:,0]
            dataset = data.iloc[:,1]
            for i in range(len(datafiles)):
            #for i in range(20):
                datafile = datafiles.iloc[i]
                year_data = dataset.iloc[i]
                
                for j in range(1,6):
                    if(year_data == f"y{j}"):
                        
                        
                            filepath = f"Des data/Year {j}/"+str(datafile)
                            #print(filepath)
                            
                            self.file_data = pd.read_csv(filepath,sep=' ', skipinitialspace = True, 
                                               on_bad_lines="skip", comment = '#', header=46)
                            #print(self.file_data)
                            self.year = j
                            good_obj = gp.select_good_obj(self)
                            self.final_data = pd.DataFrame(columns = ["data_0","data_1","data_2","data_3"])
                            self.final_err_data = pd.DataFrame(columns = ["data_0","data_1","data_2","data_3"])
                            for key, dict_data in self.data_dict.items():
                                self.flux, self.flux_err, self.mjd, self.flt = dict_data
                                
                                gp.curve_fitting(self)
                                gp.prep_1D(self)
                                
                                self.final_data = pd.concat([self.final_data, self.flux_by_FLT])
                                self.final_err_data = pd.concat([self.final_err_data, self.flux_err_by_FLT])
                            
                            
                            gp.normalise(self)
                            object_id = datafile.replace(".dat","")
                            #gp.convert_to_1D(self, object_id)
                            #gp.curve_plotting(self)
                            gp.save_to_csv(self,object_id)
                            all_objs = np.append(all_objs,object_id)
                            all_trgts = np.append(all_trgts,j)
                            
                        
        np.savetxt("cutouts/info.csv", np.transpose([all_objs,all_trgts]),fmt="%s",delimiter=",",header="file_name,target_id")
    
    def select_good_obj(self):
        save_store = np.array([1])
        last_gap = 0
        mjd = self.file_data["MJD"].copy()
        
        #self.file_data.drop(self.file_data[self.file_data['PHOTPROB'] <= 0].index, inplace = True)
        file_data = self.file_data
        
        k=1
        for i in range(len(mjd)-1):
            if ((mjd.iloc[i+1]-mjd.iloc[i]) >= 30 or
                mjd.iloc[i+1] == mjd.iloc[-1]):
                temp = mjd.iloc[last_gap:i]
                last_gap = i
                
                for j in range(len(temp)):
                    save_store = np.append(save_store, k)
                k+=1
        
        save_store = np.append(save_store, k)
        save = pd.Series(save_store)
        file_data = self.file_data.assign(remove_index=save)
        data_dict = {}
        
       
                
        temp_data = file_data.drop(file_data[file_data['remove_index'] != self.year].index, inplace = False)
                
        flux = temp_data["FLUXCAL"]
        flux_err = temp_data["FLUXCALERR"]
        mjd = temp_data["MJD"]
        """
                t_intial = mjd.iloc[0]
                        #print(flux)
                        #print(flux_err)
                n_detect = len(flux)
                CSNR = np.sqrt(np.sum((flux**2)/flux_err**2))
                t_accept = t_intial + 30
                
                if(n_detect >= 5 and
                   CSNR > 50 and
                   t_accept >=30):
                            #print(temp_store)
                            
                
                else:
                            continue
                        """
        data_dict["year_{0}".format(i)] = flux,flux_err,mjd,temp_data["FLT"]
        
        self.data_dict = data_dict




    def curve_fitting(self):
        self.passband2lam = {"g" : 4826.85, "r" : 6223.24, "i" : 7545.98, "z" : 8590.90}
        
        pbs = self.flt.to_numpy()
        self.passbands = np.array([])
        
        for i in range(len(pbs)):
            a = str(pbs[i])
            self.passbands = np.append(self.passbands, a)
        
        
        #approximation
        aug = fulu.GaussianProcessesAugmentation(self.passband2lam)
        aug.fit(self.mjd, self.flux, self.flux_err, self.passbands)
        
        # augmentation
        self.t_aug, self.flux_aug, self.flux_err_aug, self.passband_aug = aug.augmentation(self.mjd.min(), self.mjd.max(), 128)
        
    
    def curve_plotting(self):    
        
        # visualization
        plotic = fulu.LcPlotter(self.passband2lam)
        plotic.plot_one_graph_all(t=self.mjd, flux=self.flux, flux_err=self.flux_err, passbands=self.passbands,
                                  t_approx=self.t_aug, flux_approx=self.flux_aug,
                                  flux_err_approx=self.flux_err_aug, passband_approx=self.passband_aug)
    
    
    def prep_1D(self):
        
        data_dict = {}
        data_err_dict = {}
        
        for i in range(4):
            temp_store = np.array([])
            for j in range(128):      
                temp_store = np.append(temp_store, self.flux_aug[j+(128*i)])
                
            data_dict["data_{0}".format(i)] = temp_store
            
        
        
        for i in range(4):
            temp_err_store = np.array([])
            for j in range(128):      
                temp_err_store = np.append(temp_err_store, self.flux_err_aug[j+(128*i)])
                
            data_err_dict["data_{0}".format(i)] = temp_err_store
        
        
        
        self.flux_by_FLT = pd.DataFrame.from_dict(data_dict,orient="columns")
        self.flux_err_by_FLT = pd.DataFrame.from_dict(data_err_dict,orient="columns")
        
        
        
    def convert_to_1D(self,object_id):
        
        ticks = [0,1,2,3]
        
        xlabs = ["g","r","i","z"]
        
        plt.subplot(1,2,1)
        
        plt.imshow(self.norm_data,aspect="auto")
        
        plt.title(str(object_id))
        
        plt.xticks(ticks,labels=xlabs)
        
        plt.subplot(1,2,2)
        
        plt.imshow(self.norm_err,aspect="auto")
        
        plt.title("error in "+str(object_id))
        
        plt.xticks(ticks,labels=xlabs)
        
        plt.show()
        


    
    def save_to_csv(self,key):

        
        key = key.replace("id_","")
        #print(type(key))
        
        
        
        #image_data,image_err = gp.normalise(self.flux_by_FLT,self.flux_err_by_FLT)
        
        
        
        #print(image_data)
        

                
                
        self.norm_data.to_csv("cutouts/{0}.csv".format(key),index=False)
        self.norm_err.to_csv("cutouts/{0}_err.csv".format(key),index=False)
        return key    
        

    def normalise(self):
        
        image_data = self.final_data.to_numpy().flatten()
        image_err = self.final_err_data.to_numpy().flatten()
        
        
        mean = image_data.mean()

        stand_dev = image_data.std()

        #print("metrics")
        #print(mean)
        #print(stand_dev)
        
        self.norm_data = (self.final_data - mean)/stand_dev
        #print(norm_data)
        
        data_max = self.final_data.max().max()
        
        norm_data_max = self.norm_data.max().max()
        #print(data_max)
        #print(norm_data_max)
        
        norm_const = data_max/norm_data_max
        #print(norm_const)
        
        self.norm_err = self.final_err_data/norm_const
        
        
        
                
data = load_data()

run = gp(data)

run.load_datafile()






















