# Ship Detection Algorithms for SAR Satellite Imagery

Adaptive Threshold Kernel is designed by the integration of Bilateral filters and Genetic Algorithm. Dual Polarized Channels of SAR Imagery are fused to enhance the features of detected ships.

CONTENTS OF THIS FILE
---------------------

 * Introduction
 * Requirements
 * Dependecies
 * Installation
 * Contribute
 * FAQ
 * Maintainers
 
 ## Introduction
 
Space-borne Synthetic Aperture radar is the prominent data source for the surveillance of various world-wide activities and its applications. Interpretation of the SAR imagery involves the principles of wave interaction with the targets. Moreover, polarization state in SAR also helps us to understand the orientation of the targets. SAR can generate the imagery 24x7, without any restrictions on weather conditions, thus ship detection using SAR data is more powerful than optical data as optical sensors cannot work at night and also affected by cloud cover. Moreover, metallic objects are better detected in SAR images as compare to optical images. Many techniques have been evolved for the detection of ships, conventional constant false alarm rate (CFAR) is one of them. In this study, an advanced adaptive threshold CFAR kernel has been developed with the usage of bilateral filters to reduce the false alarms. The innovation of this study is based on the fusion of dual polarized channels of SAR imagery to
detect ships using bilateral CFAR. Moreover, due to the good geometrical accuracy of the SAR imagery, we have developed a methodology to generate the land mask using a vector based approach to get the precise region of interest for the processing. Finally the results are analysed and validated using the Receiver Operating Characteristic (ROC) curves.
 
 
 ## Requirements
 
 1. System Requirements:
     
     * Prefably Linux (ubuntu), Windows will also work. 

 
 ## Dependencies
 
 * GDAL (>==3.0.4) (pip3 install gdal-bin)
 * Numpy (compatible with python version)
 * tqdm
 * Tkinter (For GUI)
 * Qgis (>==3.10)
 * dbf4
 * pandas
 
 ## Installation
 
 <code>
    
    $ git clone https://github.com/harshal306/ShipDetectionBCFAR_SARImagery
    $ cd ShipDetectionBCFAR_SARImagery
    $ cd setup/DAS/dist
    $ pip3 install Detection_Algorithms_for_Ships_IIRS-0.0.1-py3-none-any.whl
    
 </code>
 
 ## Contribute
 
 <code>
    
    $ git branch 'new-branch-name'
    $ git checkout -b 'new-branch-name'
    
 </code>
 
 ## Author and Co- Author
 
 1. Harshal Mittal
     M.Tech in Remote Sensing and GIS 
     Specialization in Satellite Image Analysis and Photogrammetry
     Indian Institute of Remote Sensing, ISRO
     Contact: hrshlmittal306@gmail.com
     
 2. Ashish Joshi
     Scientist/Engineer 'SE'
     Photogrametry and Remote Sensin Department, 
     Indian Institute of Remote Sensing, ISRO
     Contact: ashish@iirs.gov.in