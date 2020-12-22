#!/usr/bin/env python
# coding: utf-8

# In[7]:

from __future__ import absolute_import
from . import GeoProcess
import import_ipynb
import numpy as np
from tqdm import tqdm
from KDEpy import FFTKDE
from numpy import trapz
import concurrent.futures

# In[9]:


# CFAR version 2, in this slidin window is created
#on the basis of value of the pixel


class CFAR_v2(object):

    #initializing the values

    def __init__(self,img,tw,gw,bw,pfa,channel,output_path,vpath,visuals=False,masked=True,doSave=True):

        print("Configuring Kernel... ")
        self.masked = masked
        self.doSave = doSave
        self.channel = "ABCD"
        self.vpath = vpath

        if self.masked:

            self.geoPro = GeoProcess.geoProcessing(img,output_path,False)
            self.img = self.geoPro.readGeoTiff()
            self.tw = tw
            self.gw = gw
            self.bw = bw
            self.pfa = pfa
            self.output_path = output_path
            self.visuals = visuals
            self.channel = channel
            if self.channel == "VH":
                self.pixels = 238
            else:
                self.pixels = 400
            GeoProcess.os.mkdir(output_path+"/StandardCFAR_OutputforChannel_"+self.channel)
            self.geoPro.outputPath = output_path+"/StandardCFAR_OutputforChannel_"+self.channel
            self.output_path = self.geoPro.outputPath

            print("Image Shape: row-"+str(self.img.shape[0])+" col-"+str(self.img.shape[1])+"\n")
            if self.visuals:
                GeoProcess.visualizeBinaryImg(self.img)
            print("Target Window size: ", self.tw)
            print("Guard Window Size: ", self.gw)
            print("Background Window Size: ",self.bw)
            print("Probability of false Alarm used: ",self.pfa)

            print("Channel used: ", self.channel)
            print("Generation of Output at location: ",self.output_path)


        else:

            print("Performing land water Segmentation...")

            self.geoPro = GeoProcess.geoProcessing(img,output_path,True)
            self.geoPro.shapefile = self.vpath
            print(self.geoPro.shapefile)

            if "VH" in channel:
                self.channel = "VH"
                GeoProcess.os.mkdir(output_path+"/StandardCFAR_OutputforChannel_"+self.channel)
                self.geoPro.outputPath = output_path+"/StandardCFAR_OutputforChannel_"+self.channel
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
                GeoProcess.os.mkdir(output_path+"/StandardCFAR_OutputforChannel_"+self.channel)
                self.geoPro.outputPath = output_path+"/StandardCFAR_OutputforChannel_"+self.channel
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

            if self.visuals:
                GeoProcess.visualizeBinaryImg(self.img)
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


    def computeDV(self):


        dvi = []
        noise_data = []
        print("Computing DVi Image from target window...")

        radius_t = 0
        radius_g = int(self.tw/2)
        rows = self.img.shape[0]
        cols = self.img.shape[1]

        for i in tqdm(range(self.img.shape[0])):
            for j in range(self.img.shape[1]):

                #print((self.img[i,j]))

                # Find row and column index
    #             i = d // cols
    #             j = d % cols

                if (self.img[i,j]) > self.pixels:
                    #print("hello")
                    win_top_buffer = self.get_topBuffer(i,j,radius_t,radius_g)
                    win_bottom_buffer = self.get_bottomBuffer(i,j,radius_t,radius_g)
                    win_left_buffer = self.get_leftBuffer(i,j,radius_t,radius_g)
                    win_right_buffer = self.get_rightBuffer(i,j,radius_t,radius_g)

                    guard_buffer = np.array(win_top_buffer+win_bottom_buffer+
                                            win_left_buffer+win_right_buffer)

                    dvi.append((self.img[i][j] - guard_buffer.mean())/guard_buffer.std())
                    #noise_data.append((guard_buffer.mean()))
                else:
                    dvi.append(0)
                    #noise_data.append(0)

        dvi = np.array(dvi).reshape(self.img.shape)
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


                    #x,y = FFTKDE(kernel="gaussian", bw="silverman").fit(guard_buffer).evaluate()
                    #data = np.array(y)
                    #arr = np.array(np.arange(len(data)))
                    #threshold_index = self.binary_search(arr,self.pfa,0,len(arr)-1,data,x[2]-x[1])

                    #threshold.append(x[threshold_index])

                    threshold.append(guard_buffer.mean())

                else:
                    threshold.append(0)

        threshold = np.array(threshold).reshape(self.img.shape)
        print("Threshold Image Successfully computed.")
        return threshold

    def shipDetection(self):
        final_image = []
        DV = np.array([])
        T = np.array([])
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_thread1 = executor.submit(self.computeDV)
            future_thread2 = executor.submit(self.computeThreshold)
            DV = future_thread1.result()
            T = future_thread2.result()
            #print(return_value_thread1, return_value_thread2)

        # DV = self.computeDV()
        # T = self.computeThreshold()
        print("Generating Final Binary Image...")
        for i in tqdm(range(self.img.shape[0])):
            for j in range(self.img.shape[1]):

                if DV[i][j] < T[i][j]:

                    final_image.append(1)
                else:
                    final_image.append(0) #valid Ships

        final_image = np.array(final_image).reshape(self.img.shape)
        print("Binary Image of Ships is Succesfully Generated.\n")


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



# In[ ]:
