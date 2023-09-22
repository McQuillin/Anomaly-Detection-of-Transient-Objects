# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 10:34:25 2023

@author: coolm



to do:
have fitted_curve output flux by filter/

remove undesired data from the training set/

plot each heat map/

use report metrics to refine data
-split up obs data into filter/
-set up Quaility metrics with iterative system/
-filter out all plots with lots of NaNs as they are erronious/

figure out how machine learning works (shouldn't be to hard? right')

impletement said ml and train it in plasticc data

test, hopefully works perfectly

Then train on relevent data

refine

upload to sever

"""

import numpy as np

import pandas as pd

import fulu

import scipy.special as spp

from matplotlib import pyplot as plt

from astropy.io import fits

import csv

import warnings
warnings.filterwarnings("ignore")


lightcurves = pd.read_csv("Data store/plasticc_train_lightcurves.csv.gz")
metadata = pd.read_csv("Data store/plasticc_train_metadata.csv.gz")

# lightcurves1 = pd.read_csv("Data store/plasticc_test_lightcurves_01.csv.gz")
# metadata1 = pd.read_csv("Data store/plasticc_test_metadata.csv.gz")

# lightcurves1["object_id"] = lightcurves1["object_id"] +600000
# metadata1["object_id"] = metadata1["object_id"] +600000

# lightcurves = pd.concat([lightcurves,lightcurves1],join = "inner")
# metadata = pd.concat([metadata,metadata1],join = "inner")


print(lightcurves)
print(metadata)

true_target_vals = [42,52,62,67,90,95] #different types of supernova identifiers


def format_data(lightcurves,metadata):
    #groups lightcurves by their id, also only selects lcs that meet set conditions
    lightcurves = lightcurves[lightcurves["detected_bool"].isin([1])]
    print(lightcurves)
    
    curve_dict={}
    
    first_id = lightcurves.iloc[0,0]
    final_id = lightcurves.iloc[-1,0]
    print(first_id,final_id)
    
    #metadata = metadata[metadata["true_target"].isin([42,52,62,67,90,95])]
    metadata = metadata[metadata["ddf_bool"].isin([1])]
    
    print(metadata)
    #assembles all remaining data points into id based dataframes
    for i in range(len(metadata["object_id"])-1):
        CSNR = 0
        j = metadata.iloc[i,0]

        temp_store = lightcurves[lightcurves["object_id"] == j]
            
        object_id = temp_store["object_id"]
        flux = temp_store["flux"]
        flux_err = temp_store["flux_err"]
        mjd = temp_store["mjd"] 
        try:
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
                    curve_dict["id_{0}".format(j)] = temp_store
        except:
                continue


    return curve_dict,metadata


def data_prep_and_fitting(curve_dict,metadata):
    all_ids = np.array([])
    all_trgts = np.array([])
    data_dict = {}
    passband2lam = {"0": 3670.69,"1" : 4826.85, "2" : 6223.24, "3" : 7545.98, "4" : 8590.90, "5": 9710.28}
    for key, lc_dat in curve_dict.items():
        #key = "id_34299"
        #print(key)
        
        #lc_dat = curve_dict["id_34299"]
        
        pbs = lc_dat["passband"].to_numpy()
        passbands = np.array([])
        
        for i in range(len(pbs)):
            a = str(pbs[i])
            passbands = np.append(passbands, a)
            
        
        flux = lc_dat["flux"]
        flux_err = lc_dat["flux_err"]
        mjd = lc_dat["mjd"]
        
        t_aug, flux_aug, flux_err_aug, passband_aug = curve_fitting(passband2lam, passbands, flux, flux_err, mjd)
        
        aug_data = augmented_data_format(t_aug, flux_aug, flux_err_aug)
        
        #metrics = Quaility_metrics(lc_dat,aug_data)
        
        print(key)
        #print(metrics)
        
        #if(filter_bad_plots(metrics) == False):
        
        #curve_plotting(passband2lam, passbands, flux, flux_err, mjd, t_aug, flux_aug, flux_err_aug, passband_aug)
        
        I,I_err = prep_1D(flux_aug, flux_err_aug, key)
            
            
            
        id_,target_id = save_to_csv(key, I, I_err, metadata)
        id_ = str(id_) +".csv"
        all_ids = np.append(all_ids,(id_))
        all_trgts = np.append(all_trgts,(target_id))
    np.savetxt("New cutouts/info.csv", np.transpose([all_ids,all_trgts]),fmt="%s",delimiter=",",header="file_name,target_id")
            
            
    
    
    return data_dict
        
        

def curve_fitting(passband2lam, passbands, flux, flux_err, t):
    
    #approximation
    aug = fulu.GaussianProcessesAugmentation(passband2lam)
    aug.fit(t, flux, flux_err, passbands)
    
    # augmentation
    t_aug, flux_aug, flux_err_aug, passband_aug = aug.augmentation(t.min(), t.max(), 128)
    return t_aug, flux_aug, flux_err_aug, passband_aug
    
def curve_plotting(passband2lam, passbands, flux, flux_err, t, t_aug, flux_aug, flux_err_aug, passband_aug):    
    
    # visualization
    plotic = fulu.LcPlotter(passband2lam)
    plotic.plot_one_graph_all(t=t, flux=flux, flux_err=flux_err, passbands=passbands,
                              t_approx=t_aug, flux_approx=flux_aug,
                              flux_err_approx=flux_err_aug, passband_approx=passband_aug)
    plt.show()

def filter_bad_plots(metrics):
    bad_plot = False
    count=0
    b_count=0
    
    max_vars = np.array([100,100,50,10,10,10,10**5,10**8])
    
    for i in range(6):
        try:
            a = pd.isna(metrics.iloc[0,i])

            if(a == True):
                count+=1
            
            for j in range(len(metrics)):
                b = metrics.iloc[j,i]
                c = max_vars[j]

                if(b >= c):
                    b_count+=1
          
        except:
            continue
            
    if(count >= 3 or
       b_count >=5):
        bad_plot = True
        print("bad plot")
    
    return bad_plot



def augmented_data_format(t_aug, flux_aug, flux_err_aug):
    aug_data = pd.DataFrame({
        "passband":
            list("0"*128 + "1"*128 + "2"*128 + "3"*128 + "4"*128 + "5"*128),
        "flux":
            flux_aug,
        "flux_err":
            flux_err_aug,
        "t_aug":
            t_aug
        })
    return aug_data



def prep_1D(flux_aug, flux_err_aug, object_id):
    
    data_dict = {}
    data_err_dict = {}
    
    for i in range(6):
        temp_store = np.array([])
        for j in range(128):      
            temp_store = np.append(temp_store, flux_aug[j+(128*i)])
            
        data_dict["data_{0}".format(i)] = temp_store
        
    
    
    for i in range(6):
        temp_err_store = np.array([])
        for j in range(128):      
            temp_err_store = np.append(temp_err_store, flux_err_aug[j+(128*i)])
            
        data_err_dict["data_{0}".format(i)] = temp_err_store
    
    
    
    flux_by_FLT = pd.DataFrame.from_dict(data_dict,orient="columns")
    flux_err_by_FLT = pd.DataFrame.from_dict(data_err_dict,orient="columns")
    print(flux_by_FLT)
    
    #flux_by_FLT, flux_err_by_FLT = convert_to_mag(flux_by_FLT, flux_err_by_FLT)
    
    
    
    #convert_to_1D(flux_by_FLT,flux_err_by_FLT,object_id)

    return flux_by_FLT,flux_err_by_FLT



def convert_to_1D(I,I_err,object_id):
    
    ticks = [0,1,2,3,4,5]
    
    xlabs = ["u","g","r","i","z","Y"]
    
    plt.subplot(1,2,1)
    
    plt.imshow(I,aspect="auto")
    
    plt.title(str(object_id))
    
    plt.xticks(ticks,labels=xlabs)
    
    plt.subplot(1,2,2)
    
    plt.imshow(I_err,aspect="auto")
    
    plt.title("error in "+str(object_id))
    
    plt.xticks(ticks,labels=xlabs)
    
    plt.show()
    

    return I


def binned_vals(obs_data, aug_data, bin_width=1):
    #collects values of obs and aug flux within the same time bins
    
    aug_store = np.array([])
    aug_err_store = np.array([])
    obs_store = np.array([])
    obs_err_store = np.array([])
    
    #print(obs_data)
    #print(aug_data)
    
    t_intial = int(aug_data.iloc[0,3])
    t_final = int(aug_data.iloc[-1,3])
    
    
    for i in range(t_intial,t_final+1):
        aug_temp_store = aug_data[(aug_data["t_aug"] >= i) & (aug_data["t_aug"] < i+1)]
        
        
        for j in range(len(obs_data)):
            t = obs_data.iloc[j,3]
            
            if(t >= i and
               t < i+1):
                obs_store = np.append(obs_store, obs_data.iloc[j,[1]])
                obs_err_store = np.append(obs_err_store, obs_data.iloc[j,[2]])

                flux_mean = np.sum(aug_temp_store["flux"])/len(aug_temp_store)
                flux_err_mean = np.sum(aug_temp_store["flux_err"])/len(aug_temp_store)
            
                aug_store = np.append(aug_store, flux_mean)
                aug_err_store = np.append(aug_err_store, flux_err_mean)
    
    data = pd.DataFrame({
           "obs_flux": obs_store,
           "obs_err": obs_err_store,
           "aug_flux": aug_store,
           "aug_err": aug_err_store
        })            

    return data


def Quaility_control(y, y_err, flux_mean, flux_err_mean):

    m = len(y)
    
    y_mean = np.sum(y)/len(y)
    
    RMSE = np.sqrt((1/m)*np.sum((y-flux_mean)**2))
    
    MAE = (1/m)*np.sum(abs(y-flux_mean))
    
    MAPE = (100/m)*np.sum(abs((y-flux_mean)/y))
    
    RSE = np.sqrt((np.sum(abs(y-flux_mean))**2)/(np.sum((abs(y-y_mean))**2)))
    
    RAE = np.sum(abs(y-flux_mean))/np.sum(abs(y-y_mean))
    
    NLPD = ((1/2)*np.log10(2*np.pi) + 
            (1/m)*np.sum(np.log10(flux_err_mean) + (((y-flux_mean)**2)/(2*flux_err_mean)**2)))
    
    nRMSEo = (1/m)*np.sum(((y-flux_mean)**2)/2*(y_err**2))
    
    nRMSEp = (1/m)*np.sum(((y-flux_mean)**2)/2*(flux_err_mean**2))

    return RMSE,MAE,MAPE,RSE,RAE,NLPD,nRMSEo,nRMSEp


def Quaility_metrics(obs_data,aug_data):
    FLTs = 6
    obs_data = obs_data.filter(["passband","flux","flux_err","mjd"])
    
    data_dict = {}
    
    for i in range(FLTs):
        print(i)
        
        FLT_obs_data = obs_data[obs_data["passband"].isin([i])]
        FLT_aug_data = aug_data[aug_data["passband"].isin([str(i)])]
        """
        print("obs_data")
        print(FLT_obs_data)
       
        print("aug_data")
        print(FLT_aug_data)
        print(FLT_aug_data.to_string())
        """
        if(len(FLT_obs_data.index) == 0 or 
           len(FLT_aug_data.index) == 0):
            continue
        
        flux_data = binned_vals(FLT_obs_data, FLT_aug_data)
        flux, flux_err, flux_mean, flux_mean_err = np.transpose(flux_data.to_numpy())
             
        data_dict["Filter_{0}".format(i)] = Quaility_control(flux, flux_err, flux_mean, flux_mean_err)
        
        
    
    report_metrics = pd.DataFrame.from_dict(data_dict,orient="columns")

    return report_metrics



def save_to_fits(key,image_data,metadata):
    print(metadata)
    image_data = np.transpose(image_data)
    hdu = fits.HDUList()    
    
    key = key.replace("id_","")
    print(type(key))
    for i in range(len(metadata)):

        object_id = str(metadata.iloc[i,0])
        #print(type(object_id))
        if(key == object_id):
            
            header = fits.Header()
            
            target_id = metadata.iloc[i,10]
            
            print(target_id)

            header["id"] = key
            header["targetid"] = target_id
            
            hdu.append(fits.ImageHDU(image_data,header=header))
            
            hdu.writeto("New cutouts/{0}.fits".format(key),overwrite= True)
            return hdu,key,target_id        
        
def save_to_csv(key,image_data,image_err,metadata):
    print(metadata)
    
    key = key.replace("id_","")
    print(type(key))
    
    
    
    image_data,image_err = normalise(image_data,image_err)
    
    
    
    #print(image_data)
    for i in range(len(metadata)):

        object_id = str(metadata.iloc[i,0])
        #print(type(object_id))
        if(key == object_id):
            
            
            
            target_id = metadata.iloc[i,10]
            
            print(target_id)

            
            
            image_data.to_csv("New cutouts/{0}.csv".format(key),index=False)
            image_err.to_csv("New cutouts/{0}_err.csv".format(key),index=False)
            return key,target_id    
    

def normalise(image_data,image_err):
    
    image_data2 = image_data.to_numpy().flatten()
    image_err2 = image_err.to_numpy().flatten()
    
    
    mean = image_data2.mean()

    stand_dev = image_data2.std()

    #print("metrics")
    #print(mean)
    #print(stand_dev)
    
    norm_data = (image_data - mean)/stand_dev
    #print(norm_data)
    
    data_max = image_data.max().max()
    
    norm_data_max = norm_data.max().max()
    #print(data_max)
    #print(norm_data_max)
    
    norm_const = data_max/norm_data_max
    #print(norm_const)
    
    norm_err = image_err/norm_const
    
    #print(norm_err)
    
    return norm_data,norm_err




def convert_to_mag(I, I_err):
    
    I = 2.5*np.log10(I+100)

    I_err = 2.5*(1/(I+100))*I_err
    
    print(I)
    print(I_err)
    
    return I,I_err




curves,metadata = format_data(lightcurves,metadata)

data_prep_and_fitting(curves,metadata)


































