# Ship Detection Algorithms for SAR Satellite Imagery

Adaptive Threshold Kernel is designed by the integration of Bilateral filters and Genetic Algorithm. Dual Polarized Channels of SAR Imagery are fused to enhance the features of detected ships.

CONTENTS OF THIS FILE
---------------------

 * Introduction
 * Requirements
 * Recommended modules
 * Installation
 * Configuration
 * Troubleshooting
 * FAQ
 * Maintainers
 
 ## Introduction
 
Space-borne Synthetic Aperture radar is the prominent data source for the surveillance of various world-wide activities and its applications. Interpretation of the SAR imagery involves the principles of wave interaction with the targets. Moreover, polarization state in SAR also helps us to understand the orientation of the targets. SAR can generate the imagery 24x7, without any restrictions on weather conditions, thus ship detection using SAR data is more powerful than optical data as optical sensors cannot work at night and also affected by cloud cover. Moreover, metallic objects are better detected in SAR images as compare to optical images. Many techniques have been evolved for the detection of ships, conventional constant false alarm rate (CFAR) is one of them. In this study, an advanced adaptive threshold CFAR kernel has been developed with the usage of bilateral filters to reduce the false alarms. The innovation of this study is based on the fusion of dual polarized channels of SAR imagery to
detect ships using bilateral CFAR. Moreover, due to the good geometrical accuracy of the SAR imagery, we have developed a methodology to generate the land mask using a vector based approach to get the precise region of interest for the processing. Finally the results are analysed and validated using the Receiver Operating Characteristic (ROC) curves.
 
 
 ## Requirements
 
 1. Software Requirements:
     
     * Python any IDLE ()
 

 
 ## Dependencies
 
 * Python (>==3.6)
 * GDAL (>==3.0.4)
 * Numpy
 * tqdm
 * Tkinter (For GUI)
 * Qgis (>==3.10)
 
 ## Installation
 
 ## configuration
 
 ## Contribute
