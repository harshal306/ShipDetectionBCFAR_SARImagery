import GeoProcess as gp
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


# INPUT = '/media/prsd/New Volume1/Dissertation/Dataset_963A'
# OUTPUT = '/media/prsd/New Volume1/Dissertation/Results/fusion'

# geodata = gp.geoProcessing(INPUT+'/terrain_corr_subset_of_S1A_IW_GRDH_1SDV_20200825T010320_20200825T010345_034056_03F420_963A_TC.tif',OUTPUT,False)

# data = geodata.readGeoTiff()


# channel_a = data[0,:,:]
# channel_b = data[1,:,:]
a = [196.57,287.27,471.33,199.27,263.00,401.42,172.20,175.70,259.48]
b = [376.48,608.90,981.78,400.88,492.03,783.33,406.43,376.99,710.54]
arr1 = np.array(a)
arr2 = np.array(b)

#print(arr1)
#print(arr2)

correlation_coef = pearsonr(arr1, arr2)

# cov_mat = np.cov(arr1,arr2)
# print(cov_mat)
# correlation_coef = (cov_mat/(arr1.std()*arr2.std()))

print(correlation_coef[0])

# merged_channel = channel_a * channel_b
# mean_channel = (channel_a + channel_b)/2
# fused_channel = (merged_channel/mean_channel)
# geodata.save_img2Geotiff(fused_channel,'/fused_usingMean.tif')

# plt.plot(arr1,arr2,'bo',markersize=3)
# plt.show()
#print(len(arr1.shape))