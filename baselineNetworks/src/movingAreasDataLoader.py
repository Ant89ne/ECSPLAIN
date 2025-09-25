#########################################################
#                      LIBRARIES                        #
#########################################################

import torch
from torch.utils.data import Dataset

import os

import numpy as np
import rioxarray as rio
import cv2

import random
from tqdm import trange

from PIL import Image

#########################################################
#                       DATASET                         #
#########################################################


class MA_Truth_Dataset(Dataset):
    def __init__(self, path1, path0, percentage = 0.8, typedataset = 0, dataType = "both", kernel = 5, stock = False):
        """
        Initialization of the dataset class

        @input path1:           Path to moving areas images (class 1)
        @input path0:           Path to empty areas (class 0)
        @input percentage:      Percentage of training images within the whole dataset
        @input typedataset:     Whether to compute a training (0) or an evaluation (1) dataset
        @input dataType:        Type of output data (cosine, sine or both)
        """
        super(MA_Truth_Dataset,self).__init__()

        #Save parameters 
        self.path1=path1
        self.path0 = path0
        self.percentage = percentage
        self.typedataset = typedataset
        self.dataType = dataType
        self.blurSize= kernel
        self.stock = stock
        self.allIms = []

        #Get the list of all moving areas images
        self.getImages1()

        #Keep images containing actual moves
        self.keepOnes()

        #Get the list of empty areas
        self.getImages0()
        
        #Test image ready and extract image shape
        imgTestName = self.imgs[0]
        imgTest = rio.open_rasterio(imgTestName).data
        self.sizeZone = imgTest.shape

        if self.stock:
            self.readAndStock()

    def readAndStock(self):
        for k in range(len(self.imgs)):
            self.allIms.append(self.getIdx(k))
        for k in range(len(self.imgs0)):
            self.allIms.append(self.getIdx(k))


    def keepOnes(self):
        """
        Function to extract only images with actual moves
        """
        print("Select only images with actual moves")
        #Read segmentation files in reverse order
        for s in trange(len(self.segs)-1, -1, -1) :
            segIm = np.array(Image.open(self.segs[s]))

            #Remove filename if no actual move in the segmentation
            if np.sum(segIm) < 10 :
                self.segs.pop(s)
                self.imgs.pop(s)

    def getImages1(self):
        """
        Look for the images and segmentations files available
        """

        #Initialization
        self.imgs = []
        self.segs = []

        #Extract the delays available
        delays = [f for f in os.listdir(self.path1)]

        for d in delays : 
            delayPath = self.path1 + d + '/'

            #Extract zones availables
            zones = sorted([f for f in os.listdir(delayPath)])

            #Avoid Queyras zone for the training and reserve it for testing
            if self.typedataset : 
                zones = [z for z in zones if "Queyras" in z] 
            else : 
                zones = [z for z in zones if "Queyras" not in z]

            for z in zones :
                zoneDelayPath = delayPath + z + "/"

                #Extract the names of the interferograms available
                interferos = sorted([f for f in os.listdir(zoneDelayPath) if (f != "Segmentations" and "30Oct" not in f)])

                #Keep only interferograms from the given dataset type
                if self.typedataset : 
                    interferos = interferos[int(self.percentage*len(interferos)):]
                else : 
                    interferos = interferos[:int(self.percentage*len(interferos))]

                #Path to the segmentation folder
                pathSeg = zoneDelayPath + "Segmentations/"

                #List moves observed in each interferogram as well as its annotations
                for i in interferos:
                    interfZoneDelayPath = zoneDelayPath + i + "/phase/"
                    imgs = [interfZoneDelayPath + f for f in os.listdir(interfZoneDelayPath) if f.endswith(".tif")]
                    imgs.sort()

                    interfSegPath = pathSeg + i + '/'
                    segs = [ interfSegPath + f for f in os.listdir(interfSegPath) if f.endswith('.tif')]
                    segs.sort()
                
                    #Add the moves to all the moves available for the dataset
                    self.imgs += imgs
                    self.segs += segs

                    
    def getImages0(self):
        """
        Look for images without moving areas
        """
        #Initialization
        self.imgs0 = []

        #Delays avaiable (corresponding to subfolder names)
        delays = [f for f in os.listdir(self.path0)]

        for d in delays :
            #List available interferograms
            delayPath = self.path0 + d + '/'
            interferos = sorted([f for f in os.listdir(delayPath) if "30Oct" not in f])

            #Select a subset of images (to distinguish training and evaluation sets)
            if self.typedataset : 
                interferos = interferos[int(self.percentage*len(interferos)):]
            else : 
                interferos = interferos[:int(self.percentage*len(interferos))]

            #List available images for the kept interferograms
            for i in interferos :
                interfZoneDelayPath = delayPath + i + "/phase/"
                self.imgs0 += [interfZoneDelayPath +  f for f in os.listdir(interfZoneDelayPath) if f.endswith(".tif")]


    def __len__(self):
        """
        Function giving the total size of the dataset
        """
        return len(self.imgs) + len(self.imgs0) 
    
    def getIdx(self, idx):
        """
        Function used to get a sample of the dataset

        @input idx:     Index of the sample to extract

        @return:        input image, corresponding label, corresponding segmentation
        """

        #Read the image depending on the index parameter
        if idx >= len(self.imgs) :                          
            path_img = self.imgs0[idx - len(self.imgs)]     #Get no moving image path
            label=torch.tensor(0, dtype = torch.float32)    #Create label
            imgSeg=torch.zeros(self.sizeZone)               #Create segmentation
            imgSeg[:,self.sizeZone[1]-1, self.sizeZone[2]-1] = 1
        else :
            path_img = self.imgs[idx]                       #Get moving area path  
            label=torch.tensor(1, dtype=torch.float32)      #Create label

            #Read segmentation
            path_seg = self.segs[idx]                       
            imgSeg = np.array(Image.open(path_seg))*255
            kernel = np.ones((self.blurSize,self.blurSize)) / self.blurSize**2

            imgSegBlur = cv2.filter2D(imgSeg, -1, kernel)
            imgSeg = torch.Tensor(imgSegBlur).to(torch.float32) /255.0
            imgSeg = imgSeg.unsqueeze(0)

            if torch.sum(imgSeg) == 0:
                label = torch.tensor(0, dtype=torch.float32)


        #Choose a random phase offset for augmentation purposes
        alphaPI = random.random()*2*torch.pi

        #Read the phase difference
        img = np.array(Image.open(path_img)) 
        img = torch.Tensor(img).unsqueeze(0)
        img = img  + alphaPI        #Add the offset

        path_coh = path_img.replace("phase", "coherence")
        coh = np.array(Image.open(path_coh)) 
        coh = torch.Tensor(coh).unsqueeze(0)

        #Format the input based on user wills
        if self.dataType == "both" :        #Both cosine and sine
            img = torch.cat((coh*torch.cos(img), coh*torch.sin(img)), 0)
        elif self.dataType == "sine" :      #Sine only
            img = torch.sin(img)
        elif self.dataType == "cosine" :    #Cosine only
            img = torch.cos(img)

        return img,label,imgSeg
    
    def __getitem__(self, index):

        if not self.stock :
            return self.getIdx(index)
        else:
            return self.allIms[index]
    
