# author@ Harshal Mittal
#!/usr/bin/env python
# coding: utf-8

# In[7]:


import import_ipynb
import numpy as np
from tqdm import tqdm
import GeoProcess as gp
from KDEpy import FFTKDE
from numpy import trapz
from sklearn.decomposition import PCA
import concurrent.futures
from scipy.stats import pearsonr

# In[9]:


# CFAR version 2, in this slidin window is created 
#on the basis of value of the pixel


class CFAR_v2(object):

    #initializing the values
    
    def __init__(self,img,tw,gw,bw,pfa,channel,output_path,vpath,doPCA=True,visuals=False,masked=True,doSave=True):
        
        print("Configuring Kernel... ")
        self.flag = 0
        self.masked = masked
        self.doSave = doSave
        self.channel = "None"
        self.vpath = vpath
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
                    #print("yes")
                else:
                    self.img = temp_img[0,:,:]
                
                if self.doPCA:
                    print("Computing PCA Based Threshold...")
                    self.pixels = self.pca_threshold(self.img,int(min(self.img.shape[0],self.img.shape[1])*0.97))
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
                    self.pixels = self.pca_threshold(self.img,int(min(self.img.shape[0],self.img.shape[1])*0.97))
                else:
                    self.pixels = 0
            else:
                raise(ValueError)
            gp.os.mkdir(output_path+"/StandardCFAR_OutputforChannel_"+self.channel)
            self.geoPro.outputPath = output_path+"/StandardCFAR_OutputforChannel_"+self.channel
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
                gp.os.mkdir(output_path+"/StandardCFAR_OutputforChannel_"+self.channel)
                self.geoPro.outputPath = output_path+"/StandardCFAR_OutputforChannel_"+self.channel
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
                    self.pixels = self.pca_threshold(self.img,int(min(self.img.shape[0],self.img.shape[1])*0.97))
                else:
                    self.pixels = 0
                
            
            elif "VV" in channel:
                self.channel = "VV"
                gp.os.mkdir(output_path+"/StandardCFAR_OutputforChannel_"+self.channel)
                self.geoPro.outputPath = output_path+"/StandardCFAR_OutputforChannel_"+self.channel
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
                    self.pixels = self.pca_threshold(self.img,int(min(self.img.shape[0],self.img.shape[1])*0.97))
                else:
                    self.pixels = 0
                print("Image Shape: row-"+str(self.img.shape[0])+" col-"+str(self.img.shape[1])+"\n")
            elif "fused" in channel:

                self.channel = "Fused_(VH+VV)"
                gp.os.mkdir(output_path+"/StandardCFAR_OutputforChannel_"+self.channel)
                self.geoPro.outputPath = output_path+"/StandardCFAR_OutputforChannel_"+self.channel
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
                    self.pixels = self.pca_threshold(self.img_vh,int(min(self.img_vh.shape[0],self.img_vh.shape[1])*0.97))
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
            self.output_path = output_path
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
        s_pca = PCA(n_components=components)
        for_s_pca = s_pca.fit_transform(data)
        #plt.imshow(for_s_pca,cmap='gray')
        #Image.fromarray(for_s_pca).show()

        max_v = for_s_pca[:,0]
        min_v = for_s_pca[:,(components-1)]
        threshold = (max_v.std() + min_v.std())/2
        #print(threshold)

        #inv_s_pca = s_pca.inverse_transform(for_s_pca)
        
        #return (inv_s_pca,threshold)
        return threshold

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

    def computeFusedDV(self):

        dvi = np.zeros(self.img_vh.shape[0]*self.img_vh.shape[1]).reshape(self.img_vh.shape)
        noise_data = []
        print("Computing Fused DVi Image from target window...")
        
        radius_t = int(self.gw/2)
        radius_g = int(self.bw/2)
        
        for i in tqdm(range(radius_g,self.img_vh.shape[0]-radius_g+1)):
            for j in range(radius_g,self.img_vh.shape[1]-radius_g+1):
                
                #print((self.img[i,j]))
     
                # Find row and column index
    #             i = d // cols
    #             j = d % cols

                if (self.img_vh[i,j]) > self.pixels:
                    #print("hello")
                    ## for VH channels
                    win_top_buffer_vh = self.get_topBuffer(self.img_vh,i,j,radius_t,radius_g)
                    win_bottom_buffer_vh = self.get_bottomBuffer(self.img_vh,i,j,radius_t,radius_g)
                    win_left_buffer_vh = self.get_leftBuffer(self.img_vh,i,j,radius_t,radius_g)
                    win_right_buffer_vh = self.get_rightBuffer(self.img_vh,i,j,radius_t,radius_g)

                     ## for vv channels
                    win_top_buffer_vv = self.get_topBuffer(self.img_vv,i,j,radius_t,radius_g)
                    win_bottom_buffer_vv = self.get_bottomBuffer(self.img_vv,i,j,radius_t,radius_g)
                    win_left_buffer_vv = self.get_leftBuffer(self.img_vv,i,j,radius_t,radius_g)
                    win_right_buffer_vv = self.get_rightBuffer(self.img_vv,i,j,radius_t,radius_g)

                   
                    self.guard_buffer_vh = np.array(win_top_buffer_vh+win_bottom_buffer_vh+
                                            win_left_buffer_vh+win_right_buffer_vh)

                    self.guard_buffer_vv = np.array(win_top_buffer_vv+win_bottom_buffer_vv+
                                            win_left_buffer_vv+win_right_buffer_vv)

                    corelation_coef,n = pearsonr(self.guard_buffer_vh,self.guard_buffer_vv)

                    val_vh = (self.img_vh[i,j] - self.guard_buffer_vh.mean())/self.guard_buffer_vh.std()
                    val_vv = (self.img_vv[i,j] - self.guard_buffer_vv.mean())/self.guard_buffer_vv.std()

                    dv = (1/(np.sqrt(2*(1+corelation_coef)))*(val_vh+val_vv))

                    dvi[i,j]= (dv)
                    #noise_data.append((guard_buffer.mean()))
                else:
                    dvi[i,j] = (0)
                    #noise_data.append(0)

        #dvi = np.array(dvi).reshape(self.img_vh.shape)
        #noise_data = (np.array(noise_data))
        #P = np.array(self.compute_scaleFactor()*noise_data).reshape(self.img.shape)
        print("Process completed, DV image succesfully Computed.\n")
        return dvi#,P
    
    def computeDV(self):


        dvi = np.zeros(self.img.shape[0]*self.img.shape[1]).reshape(self.img.shape)
        noise_data = []
        print("Computing DVi Image from target window...")
        
        radius_t = int(self.gw/2)
        radius_g = int(self.bw/2)
        rows = self.img.shape[0]
        cols = self.img.shape[1]
        
        for i in tqdm(range(radius_g,self.img.shape[0]-radius_g+1)):
            for j in range(radius_g,self.img.shape[1]-radius_g+1):
                
                #print((self.img[i,j]))
     
                # Find row and column index
    #             i = d // cols
    #             j = d % cols

                if (self.img[i,j]) > self.pixels:
                    #print("hello")
                    win_top_buffer = self.get_topBuffer(self.img,i,j,radius_t,radius_g)
                    win_bottom_buffer = self.get_bottomBuffer(self.img,i,j,radius_t,radius_g)
                    win_left_buffer = self.get_leftBuffer(self.img,i,j,radius_t,radius_g)
                    win_right_buffer = self.get_rightBuffer(self.img,i,j,radius_t,radius_g)

                    self.guard_buffer = np.array(win_top_buffer+win_bottom_buffer+
                                            win_left_buffer+win_right_buffer)

                    dvi[i,j] = ((self.img[i][j] - self.guard_buffer.mean())/self.guard_buffer.std())
                    #noise_data.append((guard_buffer.mean()))
                else:
                    dvi[i,j] = (0)
                    #noise_data.append(0)

        #dvi = np.array(dvi).reshape(self.img.shape)
        #noise_data = (np.array(noise_data))
        #P = np.array(self.compute_scaleFactor()*noise_data).reshape(self.img.shape)
        print("Process completed, DV image succesfully Computed.\n")
        return dvi#,P
    
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

        
        print("Computing Threshold from background Window...")
        threshold = np.zeros(self.img.shape[0]*self.img.shape[1]).reshape(self.img.shape)
        radius_t = int(self.gw/2)
        radius_g = int(self.bw/2)
        rows = self.img.shape[0]
        cols = self.img.shape[1]
        x_combined = 0.0
        x_val = 0.0
        
        for i in tqdm(range(radius_g,self.img.shape[0]-radius_g+1)):
            for j in range(radius_g,self.img.shape[1]-radius_g+1):
                
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
                    
                    threshold[i,j] = (self.noise_buffer.mean())
                    
                else:
                    threshold[i,j] = (0)
        
        threshold = self.scaleFactor()*np.array(threshold)
        #threshold = threshold.reshape(self.img.shape)
        print("Threshold Image Successfully computed.")
        return threshold


    def computeFusedThreshold(self):

        
        print("Computing Fused Threshold from background Window...")
        threshold = np.zeros(self.img_vh.shape[0]*self.img_vh.shape[1]).reshape(self.img_vh.shape)
        radius_t = int(self.gw/2)
        radius_g = int(self.bw/2)
        
        for i in tqdm(range(radius_g,self.img_vh.shape[0]-radius_g+1)):
            for j in range(radius_g,self.img_vh.shape[1]-radius_g+1):
                
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

                    

                    corelation_coef,n = pearsonr(self.noise_buffer_vh,self.noise_buffer_vv)

                    val_vh = self.noise_buffer_vh.mean()
                    val_vv = self.noise_buffer_vv.mean()

                    thr = (1/(np.sqrt(2*(1+corelation_coef)))*(val_vh+val_vv))

                    #x,y = FFTKDE(kernel="gaussian", bw="silverman").fit(guard_buffer).evaluate()
                    #data = np.array(y)
                    #arr = np.array(np.arange(len(data)))
                    #threshold_index = self.binary_search(arr,self.pfa,0,len(arr)-1,data,x[2]-x[1])
                    
                    #threshold.append(x[threshold_index])
                    
                    threshold[i,j] = (thr)
                    
                else:
                    threshold[i,j] = (0)
        
        threshold = self.scaleFactor()*np.array(threshold)
        #threshold = threshold.reshape(self.img_vh.shape)
        print("Threshold Image Successfully computed.")
        return threshold
    
    def shipDetection(self):
        final_image = []
        DV = np.array([])
        T = np.array([])
        if self.flag:

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_thread1 = executor.submit(self.computeFusedDV)
                future_thread2 = executor.submit(self.computeFusedThreshold)
                DV = future_thread1.result()
                T = future_thread2.result()


            print("Generating Final Binary Image...")
            for i in tqdm(range(self.img_vh.shape[0])):
                for j in range(self.img_vh.shape[1]):
                    
                    if DV[i][j] > T[i][j]:
                        
                        final_image.append(1)
                    else:
                        final_image.append(0) #valid Ships
            
            final_image = np.array(final_image).reshape(self.img_vh.shape)
            print("Binary Image of Ships is Succesfully Generated.\n")

        else:      
    
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_thread1 = executor.submit(self.computeDV)
                future_thread2 = executor.submit(self.computeThreshold)
                DV = future_thread1.result()
                T = future_thread2.result()


            print("Generating Final Binary Image...")
            for i in tqdm(range(self.img.shape[0])):
                for j in range(self.img.shape[1]):
                    
                    if DV[i][j] > T[i][j]:
                        
                        final_image.append(1)
                    else:
                        final_image.append(0) #valid Ships
        
            final_image = np.array(final_image).reshape(self.img.shape)
            print("Binary Image of Ships is Succesfully Generated.\n")
            #print(return_value_thread1, return_value_thread2)
            
        # DV = self.computeDV()
        # T = self.computeThreshold()
        
        
        
        if self.doSave:
            print("Saving the Images...")
            self.geoPro.save_img2Geotiff(final_image,'/StandardCFAR_BinaryImage_'+str(self.tw)+
                               str(self.gw)+str(self.bw)+'.tif')
            print("Final Image Saved.")
            self.geoPro.save_img2Geotiff(DV,'/StandardCFAR_DVImage_'+str(self.tw)+
                               str(self.gw)+str(self.bw)+'.tif')
            
            print("DV Image Saved.")
            self.geoPro.save_img2Geotiff(T,'/StandardCFAR_ThresholdImage_'+str(self.tw)+
                               str(self.gw)+str(self.bw)+'.tif')
            print("Threshold Image Saved.")
            
            self.geoPro.convert2Shapefile('/StandardCFAR_BinaryImage_'+str(self.tw)+
                               str(self.gw)+str(self.bw)+'.tif', '/StandardCFAR_OutputShapefile_'+str(self.tw)+
                               str(self.gw)+str(self.bw)+'.shp')
            
            print("Shapefile Image Generated.")
        return final_image,DV,T


    def scaleFactor(self):
        if self.flag:
            l = len(self.noise_buffer_vh)
        else:
            l = len(self.noise_buffer)
        
        alpha = (self.pfa**(-1/l) - 1)
        return alpha
# In[ ]:




