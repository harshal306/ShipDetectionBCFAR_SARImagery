#!/usr/bin/env python
# coding: utf-8

# In[1]:


import import_ipynb
import cv2
import numpy as np
from tqdm import tqdm as tqdm
import matplotlib.pyplot as plt
from KDEpy import FFTKDE
from numpy import trapz
from sklearn.decomposition import PCA
import GeoProcess as gp
import pandas as pd
import concurrent.futures
from scipy.stats import pearsonr


# In[4]:


# CFAR version 2, in this slidin window is created 
#on the basis of value of the pixel

class BilateralCFAR_v2(object):

    #initializing the values
    
    def __init__(self,img,tw,gw,bw,pfa,channel,output_path,vpath,doPCA=True,visuals=False,masked=True,doSave=True):
        
        print("Configuring Kernel... ")
        self.masked = masked
        self.doSave = doSave
        self.channel = "None"
        self.vpath = vpath
        self.flag = 0
        self.kernel_width = 1
        self.doPCA = doPCA
        
        if self.masked:
            
            self.geoPro = gp.geoProcessing(img,output_path,False)
            
            self.tw = tw
            self.gw = gw
            self.bw = bw
            self.pfa = pfa
            self.output_path = output_path
            self.visuals = visuals
            self.channel = channel
            if self.channel == "VH":
                temp_img = self.geoPro.readGeoTiff()
                if len(temp_img.shape) == 2:
                    self.img = temp_img
                else:
                    self.img = temp_img[0,:,:]
                if self.doPCA:
                    print("Computing PCA Based Threshold...")
                    self.pixels = self.pca_threshold(self.img,int(min(self.img.shape[0],self.img.shape[1])*0.75))
                else:
                    self.pixels = 0
            elif self.channel == "VV":
                temp_img = self.geoPro.readGeoTiff()
                if len(temp_img.shape) == 2:
                    self.img = temp_img
                else:
                    self.img = temp_img[1,:,:]
                if self.doPCA:
                    print("Computing PCA Based Threshold...")
                    self.pixels = self.pca_threshold(self.img,int(min(self.img.shape[0],self.img.shape[1])*0.75))
                else:
                    self.pixels = 0
            elif self.channel == "fused":
                temp_img = self.geoPro.readGeoTiff()
                if len(temp_img.shape) == 2:
                    self.img = temp_img
                else:
                    self.img = temp_img[1,:,:]
                if self.doPCA:
                    print("Computing PCA Based Threshold...")
                    self.pixels = self.pca_threshold(self.img,int(min(self.img.shape[0],self.img.shape[1])*0.75))
                else:
                    self.pixels = 0
            else:
                raise(ValueError)
            gp.os.mkdir(output_path+"/BilateralCFAR_OutputforChannel_"+self.channel)
            self.geoPro.outputPath = output_path+"/BilateralCFAR_OutputforChannel_"+self.channel
            self.output_path = self.geoPro.outputPath
            
            print("Image Shape: row-"+str(self.img.shape[0])+" col-"+str(self.img.shape[1])+"\n")
            if self.visuals:
                gp.visualizeBinaryImg(self.img)
            print("Target Window size: ", self.tw)
            print("Guard Window Size: ", self.gw)
            print("Background Window Size: ",self.bw)
            print("Probability of false Alarm used: ",self.pfa)
            print("Computed Possiblities of ships is greater for pixel value greater than "+str(self.pixels))
            print("Channel used: ", self.channel)
            print("Generation of Output at location: ",self.output_path)
            
    
        else:
            
            
            print("Performing land water Segmentation...")
            
            self.geoPro = gp.geoProcessing(img,output_path,True)
            self.geoPro.shapefile = self.vpath
            print(self.geoPro.shapefile)
            
            if "VH" in channel:
                self.channel = "VH"
                gp.os.mkdir(output_path+"/BilateralCFAR_OutputforChannel_"+self.channel)
                self.geoPro.outputPath = output_path+"/BilateralCFAR_OutputforChannel_"+self.channel
                self.output_path = self.geoPro.outputPath
                
                
                
                data = []
                temp_img = self.geoPro.readGeoTiff()
                if len(temp_img.shape) == 2:
                    data = temp_img
                else:
                    data = temp_img[0,:,:]
                self.geoPro.save_img2Geotiff(data,"/Input_VH.tif")
                self.geoPro.reference_img = self.output_path+"/Input_VH.tif"
                
                self.geoPro.LandMasking("LandMasked_VH.tif")
                self.geoPro.reference_img = self.output_path+"/LandMasked_VH.tif"
                #print(self.geoPro.reference_img)
                self.img = self.geoPro.readGeoTiff()
                self.img[self.img<0] = 0
                print("Image Shape: row-"+str(self.img.shape[0])+" col-"+str(self.img.shape[1])+"\n")
                
                if self.doPCA:
                    print("Computing PCA Based Threshold...")
                    self.pixels = self.pca_threshold(self.img,int(min(self.img.shape[0],self.img.shape[1])*0.75))
                else:
                    self.pixels = 0
                
            
            elif "VV" in channel:
                self.channel = "VV"
                gp.os.mkdir(output_path+"/BilateralCFAR_OutputforChannel_"+self.channel)
                self.geoPro.outputPath = output_path+"/BilateralCFAR_OutputforChannel_"+self.channel
                self.output_path = self.geoPro.outputPath
                
                
                data = []
                temp_img = self.geoPro.readGeoTiff()
                if len(temp_img.shape) == 2:
                    data = temp_img
                else:
                    data = temp_img[1,:,:]
                self.geoPro.save_img2Geotiff(data,"/Input_VV.tif")
                self.geoPro.reference_img = self.output_path+"/Input_VV.tif"
                self.geoPro.LandMasking("LandMasked_VV.tif")
                
                self.geoPro.reference_img = self.output_path+"/LandMasked_VV.tif"
                self.img = self.geoPro.readGeoTiff()
                self.img[self.img<0] = 0
                if self.doPCA:
                    print("Computing PCA Based Threshold...")
                    self.pixels = self.pca_threshold(self.img,int(min(self.img.shape[0],self.img.shape[1])*0.75))
                else:
                    self.pixels = 0
                print("Image Shape: row-"+str(self.img.shape[0])+" col-"+str(self.img.shape[1])+"\n")
            elif "fused" in channel:

                self.channel = "Fused_(VH+VV)"
                gp.os.mkdir(output_path+"/BilateralCFAR_OutputforChannel_"+self.channel)
                self.geoPro.outputPath = output_path+"/BilateralCFAR_OutputforChannel_"+self.channel
                self.output_path = self.geoPro.outputPath
                
                
                data = self.geoPro.readGeoTiff()
                if len(data.shape) == 2:
                    self.geoPro.save_img2Geotiff(data,"/Input_Fused.tif")
                    self.geoPro.reference_img = self.output_path+"/Input_Fused.tif"
                    self.geoPro.LandMasking("LandMasked_Fused.tif")
                    
                    self.geoPro.reference_img = self.output_path+"/LandMasked_Fused.tif"
                    self.img = self.geoPro.readGeoTiff()
                    self.img[self.img<0] = 0
                    print("Image Shape: row-"+str(self.img.shape[0])+" col-"+str(self.img.shape[1])+"\n")

                else:
                    self.flag = 1
                    data1 = data[0,:,:]
                    self.geoPro.save_img2Geotiff(data1,"/Input_VH.tif")
                    self.geoPro.reference_img = self.output_path+"/Input_VH.tif"
                    self.geoPro.LandMasking("LandMasked_VH.tif")
                    
                    self.geoPro.reference_img = self.output_path+"/LandMasked_VH.tif"
                    self.img_vh = self.geoPro.readGeoTiff()
                    self.img_vh[self.img_vh<0] = 0

                    data2 = data[1,:,:]
                    self.geoPro.save_img2Geotiff(data2,"/Input_VV.tif")
                    self.geoPro.reference_img = self.output_path+"/Input_VV.tif"
                    self.geoPro.LandMasking("LandMasked_VV.tif")
                    
                    self.geoPro.reference_img = self.output_path+"/LandMasked_VV.tif"
                    self.img_vv = self.geoPro.readGeoTiff()
                    self.img_vv[self.img_vv<0] = 0
                    print("Image Shape: row-"+str(self.img_vv.shape[0])+" col-"+str(self.img_vv.shape[1])+"\n")
                if self.doPCA:
                    print("Computing PCA Based Threshold...")
                    self.pixels = self.pca_threshold(self.img_vh,int(min(self.img_vh.shape[0],self.img_vh.shape[1])*0.75))
                else:
                    self.pixels = 0
            else:
                raise(ValueError)
            
            
            print("Channel used: ", self.channel)
            print("Segmented the Image successfully...")
            
            self.tw = tw
            self.gw = gw
            self.bw = bw
            self.pfa = pfa
            #self.output_path = output_path
            self.visuals = visuals
            
            if self.visuals:
                gp.visualizeBinaryImg(self.img)
            print("Target Window size: ", self.tw)
            print("Guard Window Size: ", self.gw)
            print("Background Window Size: ",self.bw)
            print("Probability of false Alarm used: ",self.pfa)
            print("Computed Possiblities of ships is greater for pixel value greater than "+str(self.pixels))
            print("Generation of Output at location: ",self.output_path)
            
        print("\nKernel Ready.")     

    #class functions
    ## Computing PCA_threshold
    def pca_threshold(self,data,components):
        #s_pca = PCA(n_components=components)
        #for_s_pca = s_pca.fit_transform(data)
        #plt.imshow(for_s_pca,cmap='gray')
        #Image.fromarray(for_s_pca).show()

        #max_v = for_s_pca[:,0]
        #min_v = for_s_pca[:,(components-1)]
        #threshold = (max_v.std() + min_v.std())/2
        #print(threshold)
        if self.channel == 'VV':
            return 250
        else:
        #inv_s_pca = s_pca.inverse_transform(for_s_pca)
        
        #return (inv_s_pca,threshold)
            return 100

    #checking if the pixel exists
    def isPixelexists(self,size_img,a,b):
        r,c = size_img
        #print(r,c)
        if (a>=0 and a<r) and (b>=0 and b<c) :
            return True
        else:
            return False

    #Computing 4 buffer values.TOP,BOTTOM,LEFT and RIGHT
    def get_topBuffer(self,img,u,v,size_t,size_g):
        top_buffer = []

        #we have considered the target_window pixels too.
        for p in range(size_t,size_g+1):
                        
            x = u-p
            for m in range(-p,p+1):
                y = v+m
                #print(x,y)
                if self.isPixelexists(img.shape,x,y):
                    #print("Found")
                    top_buffer.append(img[x][y])
                else:
                    #print("Not found")
                    top_buffer.append(0)

        return top_buffer

    def get_bottomBuffer(self,img,u,v,size_t,size_g):
        bottom_buffer = []
        
        for p in range(size_t,size_g+1):
            
            x = u+p
            for m in range(-p,p+1):
                y = v+m
                #print(x,y)
                if self.isPixelexists(img.shape,x,y):
                    #print("Found")
                    bottom_buffer.append(img[x][y])
                else:
                    #print("Not found")
                    bottom_buffer.append(0)

        return bottom_buffer

    def get_leftBuffer(self,img,u,v,size_t, size_g):
        left_buffer = []
        for p in range(size_t,size_g+1):
            y = v-p
            for m in range(-p,p+1):
                x = u+m
                #print(x,y)
                if self.isPixelexists(img.shape,x,y):
                    #print("Found")
                    left_buffer.append(img[x][y])
                else:
                    #print("Not found")
                    left_buffer.append(0)

        return left_buffer

    def get_rightBuffer(self,img,u,v,size_t,size_g):
        right_buffer = []
        
        for p in range(size_t,size_g+1):
            y = v+p
            for m in range(-p,p+1):
                x = u+m
                #print(x,y)
                if self.isPixelexists(img.shape,x,y):
                    #print("Found")
                    right_buffer.append(img[x][y])
                else:
                    #print("Not found")
                    right_buffer.append(0)

        return right_buffer

    
    def computeFusedSpatialnCombined(self):
        dvi = []
        x_spatial = []
        print("Computing Fused Spatial and Intensity Component Image from Target Window")
        
        #radius_t = int(self.tw/2)
        radius_t = 0
        radius_g = int(self.tw/2)
        
        x_combined_vh = 0.0
        x_combined_vv = 0.0
        x_val = 0.0
        
        for i in tqdm(range(self.img_vh.shape[0])):
            for j in range(self.img_vh.shape[1]):
                
                if (self.img_vh[i,j]) > self.pixels:
                    #print("hello")
                    win_top_buffer_vh = self.get_topBuffer(self.img_vh,i,j,radius_t,radius_g)
                    win_bottom_buffer_vh = self.get_bottomBuffer(self.img_vh,i,j,radius_t,radius_g)
                    win_left_buffer_vh = self.get_leftBuffer(self.img_vh,i,j,radius_t,radius_g)
                    win_right_buffer_vh = self.get_rightBuffer(self.img_vh,i,j,radius_t,radius_g)

                    self.guard_buffer_vh = np.array(win_top_buffer_vh + win_bottom_buffer_vh + 
                                            win_left_buffer_vh + win_right_buffer_vh)
                    
                    win_top_buffer_vv = self.get_topBuffer(self.img_vv,i,j,radius_t,radius_g)
                    win_bottom_buffer_vv = self.get_bottomBuffer(self.img_vv,i,j,radius_t,radius_g)
                    win_left_buffer_vv = self.get_leftBuffer(self.img_vv,i,j,radius_t,radius_g)
                    win_right_buffer_vv = self.get_rightBuffer(self.img_vv,i,j,radius_t,radius_g)

                    self.guard_buffer_vv = np.array(win_top_buffer_vv + win_bottom_buffer_vv + 
                                            win_left_buffer_vv + win_right_buffer_vv)
                    
                    


                    # FOR VH
                    #x_intensity = (self.img[i,j] - guard_buffer.mean())/guard_buffer.std()
                    x_intensity = self.img_vh[i,j]
                    
                    n = len(self.guard_buffer_vh)
                    minimum = 1000
                    maximum = -1000
                    sum_spatial = 0.0
                    #print(n)
            
                    for v in self.guard_buffer_vh:
                        #print(v)
                        valspa = np.exp((-(self.img_vh[i,j] - v)**2)/(2*(self.kernel_width**2)))
                        #print("Valspa: ",valspa)
                        sum_spatial += valspa
                        if valspa < minimum:
                            minimum = valspa
                        elif valspa > maximum:
                            maximum = valspa
                          
                    #print("spatial_sum: ",sum_spatial)
                    x_spati = (sum_spatial - minimum)/(maximum - minimum)
                    #print("f_spatial: ",(f_spatial))
                    x_spatial_vh = x_spati
                    x_combined_vh = x_spati*x_intensity*(4/(self.bw**2))
                    #print(x_combined)


                    # FOR VV
                    #x_intensity = (self.img[i,j] - guard_buffer.mean())/guard_buffer.std()
                    x_intensity = self.img_vv[i,j]
                    
                    n = len(self.guard_buffer_vv)
                    minimum = 1000
                    maximum = -1000
                    sum_spatial = 0.0
                    #print(n)
            
                    for v in self.guard_buffer_vv:
                        #print(v)
                        valspa = np.exp((-(self.img_vv[i,j] - v)**2)/(2*(self.kernel_width**2)))
                        #print("Valspa: ",valspa)
                        sum_spatial += valspa
                        if valspa < minimum:
                            minimum = valspa
                        elif valspa > maximum:
                            maximum = valspa
                          
                    #print("spatial_sum: ",sum_spatial)
                    x_spati = (sum_spatial - minimum)/(maximum - minimum)
                    #print("f_spatial: ",(f_spatial))
                    x_spatial_vv = x_spati
                    x_combined_vv = x_spati*x_intensity*(4/(self.bw**2))
                    #print(x_combined)

                    corelation_coef,n = pearsonr(self.guard_buffer_vh,self.guard_buffer_vv)
                    x_spatial.append(1/(np.sqrt(2*(1+corelation_coef)))*(x_spatial_vh + x_spatial_vv))
                    x_combined = (1/(np.sqrt(2*(1+corelation_coef)))*(x_combined_vh + x_combined_vv))


                    dvi.append((x_combined))
                    #noise_data.append((guard_buffer.mean()))
                else:
                    x_spatial.append(0)
                    dvi.append(0)
                    #noise_data.append(0)
        
        self.spatial_component = np.array(x_spatial).reshape(self.img_vh.shape)
        #x_combined = (4/(self.tw**2))*sum(x_combined)
        self.combined_component = np.array(dvi).reshape(self.img_vh.shape)
        #noise_data = (np.array(noise_data))
        #P = np.array(self.compute_scaleFactor()*noise_data).reshape(self.img.shape)
        print("Process completed, Spatial and Intensity component Sucessfully Generated.\n")
        return self.combined_component,self.spatial_component


    def computeSpatialnCombined(self):
        dvi = []
        x_spatial = []
        noise_data = []
        print("Computing Spatial and Intensity Component Image from Target Window")
        
        #radius_t = int(self.tw/2)
        radius_t = 0
        radius_g = int(self.tw/2)
        x_combined = 0.0
        x_val = 0.0
        
        for i in tqdm(range(self.img.shape[0])):
            for j in range(self.img.shape[1]):
                
                if (self.img[i,j]) > self.pixels:
                    #print("hello")
                    win_top_buffer = self.get_topBuffer(self.img,i,j,radius_t,radius_g)
                    win_bottom_buffer = self.get_bottomBuffer(self.img,i,j,radius_t,radius_g)
                    win_left_buffer = self.get_leftBuffer(self.img,i,j,radius_t,radius_g)
                    win_right_buffer = self.get_rightBuffer(self.img,i,j,radius_t,radius_g)

                    self.guard_buffer = np.array(win_top_buffer + win_bottom_buffer + 
                                            win_left_buffer + win_right_buffer)
                    
                    
                    
                    #x_intensity = (self.img[i,j] - guard_buffer.mean())/guard_buffer.std()
                    x_intensity = self.img[i,j]
                    
                    n = len(self.guard_buffer)
                    minimum = 1000
                    maximum = -1000
                    sum_spatial = 0.0
                    #print(n)
            
                    for v in self.guard_buffer:
                        #print(v)
                        valspa = np.exp((-(self.img[i,j] - v)**2)/(2*(self.kernel_width**2)))
                        #print("Valspa: ",valspa)
                        sum_spatial += valspa
                        #print(valspa)
                        if valspa < minimum:
                            minimum = valspa
                        elif valspa > maximum:
                            maximum = valspa
                          
                    #print("spatial_sum: ",sum_spatial)
                    #print(sum_spatial,minimum,maximum)
                    x_spati = (sum_spatial - minimum)/(maximum - minimum)
                    #print("f_spatial: ",(f_spatial))
                    x_spatial.append(x_spati)
                    x_combined = x_spati*x_intensity*(4/(self.bw**2))
                    #print(x_combined)
                    dvi.append((x_combined))
                    #noise_data.append((guard_buffer.mean()))
                else:
                    x_spatial.append(0)
                    dvi.append(0)
                    #noise_data.append(0)
        
        self.spatial_component = np.array(x_spatial).reshape(self.img.shape)
        #x_combined = (4/(self.tw**2))*sum(x_combined)
        self.combined_component = np.array(dvi).reshape(self.img.shape)
        #noise_data = (np.array(noise_data))
        #P = np.array(self.compute_scaleFactor()*noise_data).reshape(self.img.shape)
        print("Process completed, Spatial and Intensity component Sucessfully Generated.\n")
        return self.combined_component,self.spatial_component

    def binary_search(self,arr, val, start,end, data,x_offset): 
  
        mid = int((start+end)/2)
        #print(mid)
        area = trapz(data[mid:],dx=x_offset)
        #print(area)
        #print(abs(area-val))
        if abs(area - val) < 0.001: 
            return mid 
        elif area > val: 
            return self.binary_search(arr, val, mid, end, data,x_offset) 
        elif area < val:
            return self.binary_search(arr, val, start, mid, data,x_offset) 
    
    
    def computeFusedThreshold(self):
        
        print("Computing Fused Threshold from Background Window...")
        threshold = []
        radius_t = int(self.gw/2)
        radius_g = int(self.bw/2)
        x_combined = 0.0
        x_val = 0.0
        
        for i in tqdm(range(self.img_vh.shape[0])):
            for j in range(self.img_vh.shape[1]):
                
                if (self.img_vh[i,j]) > self.pixels:
                    #print("hello")
                    win_top_buffer_vh = self.get_topBuffer(self.img_vh,i,j,radius_t,radius_g)
                    win_bottom_buffer_vh = self.get_bottomBuffer(self.img_vh,i,j,radius_t,radius_g)
                    win_left_buffer_vh = self.get_leftBuffer(self.img_vh,i,j,radius_t,radius_g)
                    win_right_buffer_vh = self.get_rightBuffer(self.img_vh,i,j,radius_t,radius_g)

                    self.noise_buffer_vh = np.array(win_top_buffer_vh + win_bottom_buffer_vh + 
                                            win_left_buffer_vh + win_right_buffer_vh)


                    win_top_buffer_vv = self.get_topBuffer(self.img_vv,i,j,radius_t,radius_g)
                    win_bottom_buffer_vv = self.get_bottomBuffer(self.img_vv,i,j,radius_t,radius_g)
                    win_left_buffer_vv = self.get_leftBuffer(self.img_vv,i,j,radius_t,radius_g)
                    win_right_buffer_vv = self.get_rightBuffer(self.img_vv,i,j,radius_t,radius_g)

                    self.noise_buffer_vv = np.array(win_top_buffer_vv + win_bottom_buffer_vv + 
                                            win_left_buffer_vv + win_right_buffer_vv)
                    
                    
                    ### FOR VH
                    # x,y = FFTKDE(kernel="gaussian", bw="silverman").fit(self.noise_buffer_vh).evaluate()
                    # data = np.array(y)
                    # arr = np.array(np.arange(len(data)))
                    # threshold_index = self.binary_search(arr,self.pfa,0,len(arr)-1,data,x[2]-x[1])
                    # threshold_vh = x[threshold_index]

                    threshold_vh.append(self.noise_buffer_vh.mean())


                    ### FOR VV
                    # x,y = FFTKDE(kernel="gaussian", bw="silverman").fit(self.noise_buffer_vv).evaluate()
                    # data = np.array(y)
                    # arr = np.array(np.arange(len(data)))
                    # threshold_index = self.binary_search(arr,self.pfa,0,len(arr)-1,data,x[2]-x[1])
                    # threshold_vv = x[threshold_index]

                    threshold_vv.append(self.noise_buffer_vv.mean())

                    corelation_coef,n = pearsonr(self.noise_buffer_vh,self.noise_buffer_vv)

                    threshold_combined = (1/(np.sqrt(2*(1+corelation_coef)))*(threshold_vh + threshold_vv))



                    threshold.append(threshold_combined)

                else:
                    threshold.append(0)
        
        threshold = self.scaleFactor()*np.array(threshold)
        threshold = threshold.reshape(self.img_vh.shape)
        print("Threshold Image Successfully generated.\n")
        return threshold
    
    def computeThreshold(self):
        
        print("Computing Threshold from Background Window...")
        threshold = []
        radius_t = int(self.gw/2)
        radius_g = int(self.bw/2)
        x_combined = 0.0
        x_val = 0.0
        
        for i in tqdm(range(self.img.shape[0])):
            for j in range(self.img.shape[1]):
                
                if (self.img[i,j]) > self.pixels:
                    #print("hello")
                    win_top_buffer = self.get_topBuffer(self.img,i,j,radius_t,radius_g)
                    win_bottom_buffer = self.get_bottomBuffer(self.img,i,j,radius_t,radius_g)
                    win_left_buffer = self.get_leftBuffer(self.img,i,j,radius_t,radius_g)
                    win_right_buffer = self.get_rightBuffer(self.img,i,j,radius_t,radius_g)

                    self.noise_buffer = np.array(win_top_buffer + win_bottom_buffer + 
                                            win_left_buffer + win_right_buffer)
                    
                    
                    # x,y = FFTKDE(kernel="gaussian", bw="silverman").fit(self.noise_buffer).evaluate()
                    # data = np.array(y)
                    # arr = np.array(np.arange(len(data)))
                    # threshold_index = self.binary_search(arr,self.pfa,0,len(arr)-1,data,x[2]-x[1])
                    
                    # threshold.append(x[threshold_index])

                    threshold.append(self.noise_buffer.mean())

                else:
                    threshold.append(0)
        
        threshold = self.scaleFactor()*np.array(threshold)
        threshold = threshold.reshape(self.img.shape)
        print("Threshold Image Successfully generated.\n")
        return threshold
        

    def morphological_operation(self,img,kernel,iteration):

        k = np.ones((kernel,kernel), np.uint8)
        dil = cv2.dilate(img,k,iterations=iteration)
        return dil


    def shipDetection(self):
        final_image = []
        x_combined = np.array([])
        x_spatial = np.array([])
        threshold = np.array([])
        
        if self.flag:

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_thread1 = executor.submit(self.computeFusedSpatialnCombined)
                #future_thread2 = executor.submit(self.computeFusedThreshold)
                x_combined,x_spatial = future_thread1.result()
                #threshold = future_thread2.result()

            if self.doPCA:
                self.subset = self.img_vh[self.img_vh<self.pixels]
            else:
                self.pixels = self.pca_threshold(self.img_vh,int(min(self.img_vh.shape[0],self.img_vh.shape[1])*0.97))
                self.subset = self.img_vh[self.img_vh<self.pixels]

            threshold = self.scaleFactor()*(sum(self.subset)/(len(self.subset)))

            print("Generating Final Binary Image...")
            for i in tqdm(range(self.img_vh.shape[0])):
                for j in range(self.img_vh.shape[1]):
                    
                    if x_combined[i][j] > threshold:
                        
                        final_image.append(1)
                    else:
                        final_image.append(0) #valid Ships
            
            final_image = np.array(final_image).reshape(self.img_vh.shape)
            print("Binary Image of Ships is Succesfully Generated.\n")


            final_image = final_image.astype('uint8')
            final_image = self.morphological_operation(final_image,3,1)
        else:      
    
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_thread1 = executor.submit(self.computeSpatialnCombined)
                #future_thread2 = executor.submit(self.computeThreshold)
                x_combined,x_spatial = future_thread1.result()
                #threshold = future_thread2.result()

            if self.doPCA:
                self.subset = self.img[self.img<self.pixels]
            else:
                self.pixels = self.pca_threshold(self.img,int(min(self.img.sself.hape[0],self.img.shape[1])*0.97))
                self.subset = self.img[self.img<self.pixels]
            
            threshold = self.scaleFactor()*(sum(self.subset)/(len(self.subset)))
            print(threshold)

            print("Generating Final Binary Image...")
            for i in tqdm(range(self.img.shape[0])):
                for j in range(self.img.shape[1]):
                    
                    if x_combined[i][j] > threshold:
                        
                        final_image.append(1)
                    else:
                        final_image.append(0) #valid Ships
        
            final_image = np.array(final_image).reshape(self.img.shape)
            print("Binary Image of Ships is Succesfully Generated.\n")
            #print(return_value_thread1, return_value_thread2)
            final_image = final_image.astype('uint8')
            final_image = self.morphological_operation(final_image,3,1)

        if self.doSave:
            print("Saving the Images...")
            
            self.geoPro.save_img2Geotiff(final_image,'/BilateralCFAR_BinaryImage_'+str(self.tw)+
                               str(self.gw)+str(self.bw)+'.tif')
            print("Final Image Saved.")
            self.geoPro.save_img2Geotiff(x_combined,'/BilateralCFAR_XCOMBINEDImage_'+str(self.tw)+
                               str(self.gw)+str(self.bw)+'.tif')
            
            print("X_COMBINED Image Saved.")
            self.geoPro.save_img2Geotiff(x_spatial,'/BilateralCFAR_XSPATIALImage_'+str(self.tw)+
                               str(self.gw)+str(self.bw)+'.tif')
            print("X_SPATIAL Image Saved.")
            
            # self.geoPro.save_img2Geotiff(threshold,'/BilateralCFAR_ThresholdImage_'+str(self.tw)+
            #                    str(self.gw)+str(self.bw)+'.tif')
            # print("Threshold Image Saved.")
            
            self.geoPro.convert2Shapefile('/BilateralCFAR_BinaryImage_'+str(self.tw)+
                               str(self.gw)+str(self.bw)+'.tif', '/BilateralCFAR_OutputShapefile_'+str(self.tw)+
                               str(self.gw)+str(self.bw)+'.shp')
            
            print("Shapefile Image Generated.")

            df = self.geoPro.createReport('/BilateralCFAR_OutputShapefile_'+str(self.tw)+
                               str(self.gw)+str(self.bw)+'.shp')
            
            self.updatingDataframe(self.output_path+'/ShipDetection_Report.csv',x_combined)
            print("Generated the Ship Detection Report Sucessfully.")
        
        return final_image, x_combined, x_spatial,threshold
    
    def updatingDataframe(self,df,dv):

        print("Updating the Report... ")
        df = pd.read_csv(df)
        data = list(df['ScanPixel'])
        for i in tqdm(range(len(data))):
            arr = []
            x,y = data[i].split(',')
            x = int(x[1:])
            y = int(y[:len(y)-1])
            arr.append(float(dv[y,x]))
            arr.append(float(dv[y-1,x-1]))
            arr.append(float(dv[y+1,x+1]))

            arr.append(float(dv[y,x+1]))
            arr.append(float(dv[y,x-1]))
            arr.append(float(dv[y+1,x]))
            arr.append(float(dv[y-1,x]))

            arr.append(float(dv[y-1,x+1]))
            arr.append(float(dv[y+1,x-1]))

            arr = np.array(arr)
            val = sum(arr)/9.0
            df.loc[i,'Pixel_Val'] = val

        final_df = df[['ID','ScanPixel','lat-Long','Area','Perimeter','Pixel_Val']]
        final_df.to_csv(self.output_path+"/ShipDetection_Report.csv")

    def scaleFactor(self):
        if self.flag:
            l = len(self.subset)
        else:
            l = len(self.subset)
        
        alpha = 0.00560*l*(self.pfa**(-1/l) - 1)
        return alpha

# In[ ]:




