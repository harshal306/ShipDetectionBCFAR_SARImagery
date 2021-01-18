#!/usr/bin/env python
# coding: utf-8

# # Importing Modules

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal,gdal_array,ogr,osr
from shapely.geometry import box
from qgis.core import QgsRasterLayer,QgsVectorLayer,QgsField,QgsDistanceArea,QgsPoint
from PyQt5.QtCore import QVariant
from qgis.analysis import QgsRasterCalculatorEntry, QgsRasterCalculator
import os, shutil
from simpledbf import Dbf5
import easygui


# # Defining Methods

# In[18]:


class geoProcessing(object):
    
    def __init__(self,reference_img,output_path,vectorlayer=False):
        
        
        ### reference img = rasterfile main image path
        ### output_path = it is the folder name for output files.
        ### shapfile = absoulute path for shapfile
        
        
        self.reference_img = reference_img
        self.outputPath = output_path
        self.currImage = np.array([])
        self.shapefile = ""


    def save_img2Geotiff(self,img,filename):
        r,c = img.shape
        pr_d = gdal.Open(self.reference_img)
        driver = gdal.GetDriverByName("GTiff")
        outdata = driver.Create(self.outputPath+str(filename), c, r, 1, gdal.GDT_Float32)
        outdata.SetGeoTransform(pr_d.GetGeoTransform())##sets same geotransform as input
        outdata.SetProjection(pr_d.GetProjection())##sets same projection as input
        outdata.GetRasterBand(1).WriteArray(img)
        print("Image Saved Succesfully.")

    def readGeoTiff(self):
        # Importing Product and getting data from the band
        product = gdal_array.LoadFile(self.reference_img)
        product = np.array(product)
        self.currImage = product
        return product

    def subsetImg(self,row,col):
        #Comuting subset of the image for demo processing
        subset_min_size = row
        subset_max_size = col
        size = subset_max_size - subset_min_size
        subset_data = self.currImage[subset_min_size:subset_max_size,subset_min_size:subset_max_size]
        subset_data = np.array(subset_data)
        #plt.imsave('Input_Image.tiff',subset_data,cmap='gray')
        #print(subset_data.shape)

        return subset_data


    def visualizeImg(self):
        plt.imshow(self.currImage,cmap='gray',vmax=255,vmin=0)

    def visualizeBinaryImg(self):
        plt.imshow(self.currImage,cmap='gray')

    def getLatLong(self,i,j):
        pr_d = gdal.Open(self.reference_img)
        xoff, a, b, yoff, d, e = pr_d.GetGeoTransform()
        #print(xoff,yoff,a,e)
        # px = (a * i) + (b * j) + a*0.5 + b*0.5 + xoff
        # py = (d * i) + (e * j) + d*0.5 + e*0.5 + yoff 

        px = (a * i) + xoff
        py = (e * j) + yoff 

        # r,c = self.currImage.shape
        # UL_lat,UL_long = yoff,xoff
        # BR_lat,BR_long = 
        # UR_lat,UR_long =
        # BL_lat,BL_long =        
        # #print(BL_lat,UL_lat,UR_long,UL_long,c,r)
        # d_long = (BL_long - UL_long)/c
        # d_lat = (UR_lat - UL_lat)/r

        # px = UL_lat + d_lat*i
        # py = UL_long +d_long*j

        return (px,py)

    def getPixelFromLatLong(self,x,y):
        
        pr_d = gdal.Open(self.reference_img)
        xoff, a, b, yoff, d, e = pr_d.GetGeoTransform()
        #print(xoff,yoff,a,e)
        # r,c = self.currImage.shape
        # UL_lat,UL_long = self.getLatLong(0,0)
        # BR_lat,BR_long = self.getLatLong(c-1,r-1)
        # UR_lat,UR_long = self.getLatLong(c-1,0)
        # BL_lat,BL_long = self.getLatLong(0,r-1)
        # #print(BL_lat,UL_lat,UR_long,UL_long,c,r)
        # d_long = (BL_long - UL_long)/c
        # d_lat = (UR_lat - UL_lat)/r

        i = int((x - xoff)/(a))
        j = int((y-yoff)/(e))

        return (i,j)

    def createBuffer(self,outputBufferfn, bufferDist):
        inputds = ogr.Open(self.shapefile)
        inputlyr = inputds.GetLayer()

        shpdriver = ogr.GetDriverByName('ESRI Shapefile')
        if os.path.exists(self.outputPath+"/"+outputBufferfn):
            shpdriver.DeleteDataSource(self.outputPath+"/"+outputBufferfn)
        outputBufferds = shpdriver.CreateDataSource(self.outputPath+"/"+outputBufferfn)
        bufferlyr = outputBufferds.CreateLayer(self.outputPath+"/"+outputBufferfn, geom_type=ogr.wkbPolygon)
        featureDefn = bufferlyr.GetLayerDefn()

        for feature in inputlyr:
            ingeom = feature.GetGeometryRef()
            geomBuffer = ingeom.Buffer(bufferDist)

            outFeature = ogr.Feature(featureDefn)
            outFeature.SetGeometry(geomBuffer)
            bufferlyr.CreateFeature(outFeature)
        self.shapefile = self.outputPath+"/"+outputBufferfn
    
    
    def LandMasking(self,outputfileName):
        
        print("Starting LandMasking Algorithm...\n")
        os.mkdir(self.outputPath+"/temp")
        
        self.buildBox()
        self.Difference()
        self.Rasterization()
        self.BandCalc(outputfileName)
        
        shutil.rmtree(self.outputPath+"/temp")
        print("Land Masking Process compeleted.")  
        
        
    def buildBox(self):   
        
        raster = gdal.Open(self.reference_img) 
        ulx, xres, xskew, uly, yskew, yres  = raster.GetGeoTransform()
        print("Using Raster File Geo Extent: ",ulx, xres, xskew, uly, yskew, yres)
        lrx = ulx + (raster.RasterXSize * xres) 
        lry = uly + (raster.RasterYSize * yres)

        data = box(lrx,lry,ulx,uly)

        print("\nCreating Raster Mask from Raster data...")
        driver = ogr.GetDriverByName("ESRI Shapefile")
        data_source = driver.CreateDataSource(self.outputPath+"/temp/TileBox.shp")
        layer = data_source.CreateLayer("result",None, ogr.wkbPolygon)
        layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))
        defn = layer.GetLayerDefn()

        ## If there are multiple geometries, put the "for" loop here

        # Create a new feature (attribute and geometry)
        feat = ogr.Feature(defn)
        feat.SetField('id', 0)
        feat.SetField('id',1)

        # Make a geometry, from Shapely object
        geom = ogr.CreateGeometryFromWkb(data.wkb)
        feat.SetGeometry(geom)

        layer.CreateFeature(feat)
        feat = geom = None  # destroy these

        # Save and close everything
        ds = layer = feat = geom = None
        
        #########VectorBOX created.....####
        
    def Difference(self):
        poly1 = ogr.Open(self.outputPath+"/temp/TileBox.shp")
        poly2 = ogr.Open(self.shapefile)
        layer1 = poly1.GetLayer()
        layer1.GetFeatureCount()
        #1
        # first feature
        feature1 = layer1.GetFeature(0)
        #print(feature1)
        # geometry
        geom1 = feature1.GetGeometryRef()
        layer2 = poly2.GetLayer()
        layer2.GetFeatureCount()
        # first feature
        feature2 = layer2.GetFeature(0)
        # geometry
        geom2 = feature2.GetGeometryRef()
        # symmetric difference
        simdiff = geom1.Difference(geom2)
        # create a new shapefile
        driver = ogr.GetDriverByName("ESRI Shapefile")
        data_source = driver.CreateDataSource(self.outputPath+"/temp/Difference.shp")
        layer = data_source.CreateLayer("result",None, ogr.wkbPolygon)
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetGeometry(simdiff)
        layer.CreateFeature(feature)
        feature.Destroy()
        data_source.Destroy()
    
    def Rasterization(self):
        ####### Difference Calculated....##############

        _in = self.outputPath+"/temp/Difference.shp"
        _out = self.outputPath+"/temp/Rastered.tif"

        rasterLyr = QgsRasterLayer(self.reference_img)

        # 1. Define pixel_size and NoData value of new raster
        NoData_value = 0
        x_res = rasterLyr.rasterUnitsPerPixelX() # assuming these are the cell sizes
        y_res = rasterLyr.rasterUnitsPerPixelY() # change as appropriate

        print(x_res,y_res)
        pixel_size = 1

        # 2. Filenames for in- and output
        

        # 3. Open Shapefile
        source_ds = ogr.Open(_in)
        source_layer = source_ds.GetLayer()

        source_ds_extent = ogr.Open(self.outputPath+"/temp/TileBox.shp")
        source_layer_extent = source_ds_extent.GetLayer()

        x_min, x_max, y_min, y_max = source_layer_extent.GetExtent()

        #print(x_min, x_max, y_min, y_max)

        # 4. Create Target - TIFF
        cols = int( (x_max - x_min)/x_res )
        rows = int( (y_max - y_min)/y_res )
        #print(cols,rows)

        _raster = gdal.GetDriverByName('GTiff').Create(_out, cols, rows, 1, gdal.GDT_Float32)
        _raster.SetGeoTransform((x_min, x_res, 0, y_max, 0, -y_res))
        _band = _raster.GetRasterBand(1)
        _band.SetNoDataValue(NoData_value)

        # 5. Rasterize why is the burn value 0... isn't that the same as the background?
        gdal.RasterizeLayer(_raster, [1], source_layer,burn_values=[1],options=['ALL_TOUCHED=TRUE'])
        
        del _raster
        del _in
        del _out
        del source_ds
        del source_ds_extent
        
        print("Raster Mask successfully generated.")
        ##############Rasterization.... done...#####################
    
    def BandCalc(self,name):
        print("Creating Land masked image...")
                         
        data1 = gdal_array.LoadFile(self.reference_img)
        data1 = np.array(data1)
        data2 = gdal_array.LoadFile(self.outputPath+"/temp/Rastered.tif")
        data2 = np.array(data2)

        result = data1*data2
        #print(data1.shape)

        self.save_img2Geotiff(result,"/"+name)

        # input_raster1 = QgsRasterLayer(self.reference_img)      
        # input_raster2 = QgsRasterLayer(self.outputPath+"/temp/Rastered.tif")      
        # output_raster = self.outputPath+"/"+name

        # enteries = []

        # ras1 = QgsRasterCalculatorEntry()
        # ras1.ref = 'ras1@1'
        # ras1.raster= input_raster1
        # ras1.bandNumber = 1
        # enteries.append(ras1)

        # ras2 = QgsRasterCalculatorEntry()
        # ras2.ref = 'ras2@1'
        # ras2.raster= input_raster2
        # ras2.bandNumber = 1
        # enteries.append(ras2)
        # #print(enteries)
        # calc = QgsRasterCalculator('((ras2@1 * ras1@1))', output_raster,'GTiff',
        #                           input_raster1.extent(), input_raster1.width(), input_raster1.height(), enteries)
        # calc.processCalculation()       
        
        ##### Land Masked Image Computed...#####
    
    def dbf2DF(self,dbfile, upper=True):
        "Read dbf file and return pandas DataFrame"
        with ps.open(dbfile) as db:  # I suspect just using open will work too
            df = pd.DataFrame({col: db.by_col(col) for col in db.header})
            if upper == True: 
                df.columns = map(str.upper, db.header) 
            return df

    def createReport(self,outputfile):

        d = QgsDistanceArea()
        d.setEllipsoid('WGS84')
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)   
        print("Creating final Report")
        layer=QgsVectorLayer(self.outputPath+outputfile)
        #print(layer.fields().names())

        layer_provider=layer.dataProvider()
        layer_provider.addAttributes([QgsField("Area",QVariant.Double)])
        layer.updateFields()
        #print (layer.fields().names())

        features=layer.getFeatures()
        layer.startEditing()
        for f in features:
            _id=f.id()
            area= d.measureArea(f.geometry())
            attr_value={1:area}
            layer_provider.changeAttributeValues({_id:attr_value})
        layer.commitChanges()


        layer_provider=layer.dataProvider()
        layer_provider.addAttributes([QgsField("Perimeter",QVariant.Double)])
        layer.updateFields()
        #print (layer.fields().names())

        features=layer.getFeatures()
        layer.startEditing()
        for f in features:
            _id=f.id()
            area=d.measurePerimeter(f.geometry())
            attr_value={2:area}
            layer_provider.changeAttributeValues({_id:attr_value})
        layer.commitChanges()

        ##xcorrdinate and y coordinate

        layer_provider=layer.dataProvider()
        layer_provider.addAttributes([QgsField("lat-Long",QVariant.String)])
        layer.updateFields()

        layer_provider=layer.dataProvider()
        layer_provider.addAttributes([QgsField("ScanPixel",QVariant.String)])
        layer.updateFields()
        #print (layer.fields().names())

        features=layer.getFeatures()
        layer.startEditing()
        for f in features:
            _id=f.id()
            #x= f.extent().xMinimum()
            x = f.geometry().asMultiPolygon()[0][0][2][0]
            y = f.geometry().asMultiPolygon()[0][0][2][1]

            pixel = str(self.getPixelFromLatLong(x,y))
            ll = (x,y)
            attr_value={3:str(ll)}
            layer_provider.changeAttributeValues({_id:attr_value})
            attr_value={4:pixel}
            layer_provider.changeAttributeValues({_id:attr_value})
        layer.commitChanges()

        ##adding ID
        layer_provider=layer.dataProvider()
        layer_provider.addAttributes([QgsField("ID",QVariant.String)])
        layer.updateFields()
        #print (layer.fields().names())
        count = 0
        features=layer.getFeatures()
        layer.startEditing()
        for f in features:
            _id=f.id()
            #x= f.extent().xMinimum()
            y = count
            attr_value={5:y}
            count = count +1
            layer_provider.changeAttributeValues({_id:attr_value})
        layer.commitChanges()

        ## Deletion of value attribute
        layer_provider.deleteAttributes([0])
        layer.updateFields()
# In[ ]:
        ### Saving to csv
        new_file = outputfile.split('.')
        dbf = Dbf5(self.outputPath+str(new_file[0])+'.dbf')
        dbf.to_csv(self.outputPath+"/ShipDetection_Report.csv")
        return dbf


    def convert2Shapefile(self,rasterfile,outputfile):
        
        d = QgsDistanceArea()
        d.setEllipsoid('WGS84')
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)    
        sourceRaster = gdal.Open(self.outputPath+rasterfile)
        band = sourceRaster.GetRasterBand(1)
        bandArray = band.ReadAsArray()
        outShapefile = self.outputPath+outputfile
        driver = ogr.GetDriverByName("ESRI Shapefile")
        outDatasource = driver.CreateDataSource(outShapefile)
        outLayer = outDatasource.CreateLayer("polygonized", srs)
        newField = ogr.FieldDefn("Value", ogr.OFTInteger)
        outLayer.CreateField(newField)
        
        # for feat in outLayer:
        #     f = feat.getField("Value")
        #     print(f)
        #outLayer.deteleFeature(f.id())
        gdal.Polygonize(band, None, outLayer, 0, [], callback=None )
        outDatasource.Destroy()
        sourceRaster = None

        poly1 = ogr.Open(self.outputPath+outputfile,1)
        layer = poly1.GetLayer()
        layer.SetAttributeFilter("Value<1")

        for feat in layer:
            layer.DeleteFeature(feat.GetFID())
            poly1.ExecuteSQL('REPACK ' + layer.GetName())

        # layer1 = poly1.GetLayer()
        # layer1.GetFeatureCount()
        # feature1 = layer1.GetFeatures()

        



