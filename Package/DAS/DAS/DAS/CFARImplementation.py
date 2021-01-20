#!/usr/bin/env python
# coding: utf-8

# # Importing Modules

# In[1]:

import CFAR_v2 as cfarv2
import BilateralCFAR_v2 as bcfar
import easygui


# In[2]:


DATA_PATH = '/media/prsd/New Volume/Dissertation/Dataset_Wrkin/Input.tif'#easygui.fileopenbox()
OUT_PATH = '/media/prsd/New Volume/Dissertation/Results/test'#easygui.diropenbox()
VPATH = '/media/prsd/New Volume/Dissertation/Dataset_Wrkin/VectorLayer/GSHHGcoastline.shp'#easygui.fileopenbox()


# 1. subset_img = (gp.subsetImg(band_data_arr,2000,4500)) 1200x1200
# 2. subset_img = band_data_arr[2518:2875,3882:4282] 375x400

# In[3]:


backgroundWindow_size = 50
guardWindow_size = 44
targetWindow_size = 38
pfa = 1e-4


# In[4]:


cfar_version2 = cfarv2.CFAR_v2(DATA_PATH,
                               targetWindow_size,
                               guardWindow_size,
                               backgroundWindow_size,
                               pfa,
                               channel='VH',
                               output_path=OUT_PATH,
                               vpath=VPATH,
                               doPCA = True,
                               visuals=False,
                               masked=True,
                               doSave=True)


# In[5]:

result = cfar_version2.shipDetection()

# In[ ]:




