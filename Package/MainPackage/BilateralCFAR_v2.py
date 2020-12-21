#!/usr/bin/env python
# coding: utf-8

# In[1]:


import import_ipynb
import numpy as np
from tqdm import tqdm as tqdm
import matplotlib.pyplot as plt
from KDEpy import FFTKDE
from numpy import trapz
import GeoProcess as gp
import concurrent.futures


# In[4]:


# CFAR version 2, in this slidin window is created 
#on the basis of value of the pixel

class BilateralCFAR_v2(object):

    #initializing the values
    
    def __init__(self,img,tw,gw,bw,pfa,channel,output_path,visuals=False,masked=True,doSave=True):
        
        print("Configuring Kernel... ")
        self.masked = masked
        self.doSave = doSave
        self.channel = "ABCD"
        
        if self.masked:
            
            self.geoPro = gp.geoProcessing(img,output_path,True)
            self.img = self.geoPro.readGeoTiff()
        
            self.kernel_width = 1
            self.tw = tw
            self.gw = gw
            self.bw = bw
            self.pfa = pfa
            self.output_path = output_path
            self.visuals = visuals
            self.channel = channel
            self.spatial_component = []
            self.intensity_component = []
            self.combined_component = []
            self.threshold = []
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
            
            print("Channel used: ", self.channel)
            print("Generation of Output at location: ",self.output_path)
            
    
        else:
            
            
            print("Performing land water Segmentation...")
            
            self.geoPro = gp.geoProcessing(img,output_path,True)
            
            if "VH" in channel:
                self.channel = "VH"
                gp.os.mkdir(output_path+"/BilateralCFAR_OutputforChannel_"+self.channel)
                self.geoPro.outputPath = output_path+"/BilateralCFAR_OutputforChannel_"+self.channel
                self.output_path = self.geoPro.outputPath
                
                
                
                data = self.geoPro.readGeoTiff()[0,:,:]
                self.geoPro.save_img2Geotiff(data,"/Input_VH.tif")
                self.geoPro.reference_img = self.output_path+"/Input_VH.tif"
                
                self.geoPro.LandMasking("LandMasked_VH.tif")
                self.geoPro.reference_img = self.output_path+"/LandMasked_VH.tif"
                #print(self.geoPro.reference_img)
                self.img = self.geoPro.readGeoTiff()
                self.img[self.img<0] = 0
                self.pixels = 238
                
            
            elif "VV" in channel:
                self.channel = "VV"
                gp.os.mkdir(output_path+"/BilateralCFAR_OutputforChannel_"+self.channel)
                self.geoPro.outputPath = output_path+"/BilateralCFAR_OutputforChannel_"+self.channel
                self.output_path = self.geoPro.outputPath
                
                
                data = self.geoPro.readGeoTiff()[1,:,:]
                self.geoPro.save_img2Geotiff(data,"/Input_VV.tif")
                self.geoPro.reference_img = self.output_path+"/Input_VV.tif"
                self.geoPro.LandMasking("LandMasked_VV.tif")
                
                self.geoPro.reference_img = self.output_path+"/LandMasked_VV.tif"
                self.img = self.geoPro.readGeoTiff()
                self.img[self.img<0] = 0
                self.pixels = 400
            else:
                raise(ValueError)
            
            
            print("Channel used: ", self.channel)
            print("Segmented the Image successfully...")
            
            print("Image Shape: row-"+str(self.img.shape[0])+" col-"+str(self.img.shape[1])+"\n")
            
            self.tw = tw
            self.gw = gw
            self.bw = bw
            self.pfa = pfa
            self.output_path = output_path
            self.visuals = visuals
            self.spatial_component = []
            self.intensity_component = []
            self.combined_component = []
            self.threshold = []
            self.kernel_width = 1
            
            if self.visuals:
                gp.visualizeBinaryImg(self.img)
            print("Target Window size: ", self.tw)
            print("Guard Window Size: ", self.gw)
            print("Background Window Size: ",self.bw)
            print("Probability of false Alarm used: ",self.pfa)
            
            print("Generation of Output at location: ",self.output_path)
            
        print("\nKernel Ready.")     
           
    #checking if the pixel exists
    def isPixelexists(self,size_img,a,b):
        r,c = size_img
        #print(r,c)
        if (a>=0 and a<r) and (b>=0 and b<c) :
            return True
        else:
            return False

    #Computing 4 buffer values.TOP,BOTTOM,LEFT and RIGHT
    def get_topBuffer(self,u,v,size_t,size_g):
        top_buffer = []

        #we have considered the target_window pixels too.
        for p in range(size_t,size_g+1):
                        
            x = u-p
            for m in range(-p,p+1):
                y = v+m
                #print(x,y)
                if self.isPixelexists(self.img.shape,x,y):
                    #print("Found")
                    top_buffer.append(self.img[x][y])
                else:
                    #print("Not found")
                    top_buffer.append(0)

        return top_buffer

    def get_bottomBuffer(self,u,v,size_t,size_g):
        bottom_buffer = []
        
        for p in range(size_t,size_g+1):
            
            x = u+p
            for m in range(-p,p+1):
                y = v+m
                #print(x,y)
                if self.isPixelexists(self.img.shape,x,y):
                    #print("Found")
                    bottom_buffer.append(self.img[x][y])
                else:
                    #print("Not found")
                    bottom_buffer.append(0)

        return bottom_buffer

    def get_leftBuffer(self,u,v,size_t, size_g):
        left_buffer = []
        for p in range(size_t,size_g+1):
            y = v-p
            for m in range(-p,p+1):
                x = u+m
                #print(x,y)
                if self.isPixelexists(self.img.shape,x,y):
                    #print("Found")
                    left_buffer.append(self.img[x][y])
                else:
                    #print("Not found")
                    left_buffer.append(0)

        return left_buffer

    def get_rightBuffer(self,u,v,size_t,size_g):
        right_buffer = []
        
        for p in range(size_t,size_g+1):
            y = v+p
            for m in range(-p,p+1):
                x = u+m
                #print(x,y)
                if self.isPixelexists(self.img.shape,x,y):
                    #print("Found")
                    right_buffer.append(self.img[x][y])
                else:
                    #print("Not found")
                    right_buffer.append(0)

        return right_buffer

    
    def computeSpatialnCombined(self):
        dvi = []
        x_spatial = []
        noise_data = []
        print("Computing Spatial and Intensity Component Image from Target Window")
        
        #radius_t = int(self.tw/2)
        radius_t = 0
        radius_g = int(self.tw/2)
        rows = self.img.shape[0]
        cols = self.img.shape[1]
        x_combined = 0.0
        x_val = 0.0
        
        for i in tqdm(range(self.img.shape[0])):
            for j in range(self.img.shape[1]):
                
                if (self.img[i,j]) > self.pixels:
                    #print("hello")
                    win_top_buffer = self.get_topBuffer(i,j,radius_t,radius_g)
                    win_bottom_buffer = self.get_bottomBuffer(i,j,radius_t,radius_g)
                    win_left_buffer = self.get_leftBuffer(i,j,radius_t,radius_g)
                    win_right_buffer = self.get_rightBuffer(i,j,radius_t,radius_g)

                    guard_buffer = np.array(win_top_buffer + win_bottom_buffer + 
                                            win_left_buffer + win_right_buffer)
                    
                    
                    
                    #x_intensity = (self.img[i,j] - guard_buffer.mean())/guard_buffer.std()
                    x_intensity = self.img[i,j]
                    
                    n = len(guard_buffer)
                    minimum = 1000
                    maximum = -1000
                    sum_spatial = 0.0
                    #print(n)
            
                    for v in guard_buffer:
                        #print(v)
                        valspa = np.exp((-(self.img[i,j] - v)**2)/(2*(self.kernel_width**2)))
                        #print("Valspa: ",valspa)
                        sum_spatial += valspa
                        if valspa < minimum:
                            minimum = valspa
                        elif valspa > maximum:
                            maximum = valspa
                          
                    #print("spatial_sum: ",sum_spatial)
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
    
    def computeThreshold(self):
        
        print("Computing Threshold from Background Window...")
        threshold = []
        radius_t = int(self.tw/2)
        radius_g = int(self.bw/2)
        rows = self.img.shape[0]
        cols = self.img.shape[1]
        x_combined = 0.0
        x_val = 0.0
        
        for i in tqdm(range(self.img.shape[0])):
            for j in range(self.img.shape[1]):
                
                if (self.img[i,j]) > self.pixels:
                    #print("hello")
                    win_top_buffer = self.get_topBuffer(i,j,radius_t,radius_g)
                    win_bottom_buffer = self.get_bottomBuffer(i,j,radius_t,radius_g)
                    win_left_buffer = self.get_leftBuffer(i,j,radius_t,radius_g)
                    win_right_buffer = self.get_rightBuffer(i,j,radius_t,radius_g)

                    guard_buffer = np.array(win_top_buffer + win_bottom_buffer + 
                                            win_left_buffer + win_right_buffer)
                    
                    
                    x,y = FFTKDE(kernel="gaussian", bw="silverman").fit(guard_buffer).evaluate()
                    data = np.array(y)
                    arr = np.array(np.arange(len(data)))
                    threshold_index = self.binary_search(arr,self.pfa,0,len(arr)-1,data,x[2]-x[1])
                    
                    threshold.append(x[threshold_index])

                else:
                    threshold.append(0)
        
        threshold = np.array(threshold).reshape(self.img.shape)
        print("Threshold Image Successfully generated.\n")
        return threshold
        
        
    def shipDetection(self):
        final_image = []
        x_combined = np.array([])
        x_spatial = np.array([])
        threshold = np.array([])

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_thread1 = executor.submit(self.computeSpatialnCombined)
            future_thread2 = executor.submit(self.computeThreshold)
            x_combined,x_spatial = future_thread1.result()
            threshold = future_thread2.result()


        # x_combined, x_spatial = self.computeSpatialnCombined()
        # threshold = self.computeThreshold()
        
        print("Generating Final Binary Image...")
        
        for i in tqdm(range(self.img.shape[0])):
            for j in range(self.img.shape[1]):
                
                if x_combined[i][j] < threshold[i][j]:
                    
                    final_image.append(1)
                else:
                    final_image.append(0) #valid Ships
        
        final_image = np.array(final_image).reshape(self.img.shape)
        print("Binary Image of Ships is Succesfully Generated.\n")
        
        
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
            
            self.geoPro.save_img2Geotiff(threshold,'/BilateralCFAR_ThresholdImage_'+str(self.tw)+
                               str(self.gw)+str(self.bw)+'.tif')
            print("Threshold Image Saved.")
            
            self.geoPro.convert2Shapefile('/BilateralCFAR_BinaryImage_'+str(self.tw)+
                               str(self.gw)+str(self.bw)+'.tif', '/BilateralCFAR_OutputShapefile_'+str(self.tw)+
                               str(self.gw)+str(self.bw)+'.shp')
            
            print("Shapefile Image Generated.")
        
        return final_image, x_combined, x_spatial,threshold
    


# In[ ]:




