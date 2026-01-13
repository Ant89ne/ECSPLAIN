import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
import cv2

import torch

def checkDir(path):
    """
    Check if directory already exists and create it
    
    @input pat :        Path to be checked
    """

    #Check if the path exists
    if os.path.exists(path):
        #Remove the path if it exists
        shutil.rmtree(path)
    #Recreate it empty
    os.makedirs(path)


def visIms(model, dataloader, saveDir, epoch, nbIms = 20, device = "cpu"):
    """
    Function used for image results visualization

    @input model:           Model to be evaluated
    @input dataloader:      Dataset to be visually evaluated
    @input saveDir:         Folder to save images
    @input epoch:           Number of training epochs
    @input nbIms:           Number of images to visualize (default = 20)
    @input device:          Device on which to compute the evaluation (default = cpu)
    """

    #Create the saving folder
    checkDir(saveDir + f'/Ims{epoch}/')

    #Extract the dataset to be evaluated
    dataset = dataloader.dataset

    for k in range(nbIms):
        #Extract an image to be computed
        d,l = dataset[k]
        d = d.unsqueeze(0)

        #Get a prediction
        pred = model(d.to(device))
        pred = pred["out"].cpu()

        #Visualization
        plt.figure(figsize = (20,10))

        #Input coherence map
        plt.subplot(231)
        plt.imshow(d[0,0,:,:], vmin = 0, vmax = 1)
        plt.title("Coherence map")
        #Input Cosine
        plt.subplot(232)
        plt.imshow(d[0,1,:,:], vmin = -1, vmax = 1)
        plt.title("Cosine")
        #Input Sine
        plt.subplot(233)
        plt.imshow(d[0,2,:,:], vmin = -1, vmax = 1)
        plt.title("Sine")
        #Phase
        plt.subplot(234)
        plt.imshow(np.arctan2(d[0,2,:,:], d[0,1,:,:]), vmin = -np.pi, vmax = np.pi)
        plt.title("Phase")
        #Ground Truth
        plt.subplot(235)
        plt.imshow(l, vmin = 0, vmax = 1)
        plt.title("Ground Truth")
        #Prediction
        plt.subplot(236)
        plt.imshow(pred[0,0,:,:], vmin = 0, vmax = 1)
        plt.title("Prediction map")
        
        #Save image
        plt.savefig(saveDir + f'/Ims{epoch}/Im{k}.png')
        plt.close()


def visImInference(model, dataset, epoch, dir, nbIm = 4, saveSep = False, device = "cpu", thresh = 0.5):
    """
    Function used to visualize images after training
    
    @input model :          Network to be used
    @input dataset :        Dataset to be used
    @input epoch :          Number of training epochs already done
    @input dir :            Directory where to save the images
    @input nbIm :           Number of images to visualize (defualt : 4)
    @input saveSep :        Separate the saving of GT and prediction (default : False)
    """
    
    #Create directory if not yet existing
    checkDir(dir)


    if nbIm == -1:
        nbIm = len(dataset)

    print("Save images")
    for k in trange(nbIm):

        #Get an image
        d,_,l = dataset[k]
        d1,_,_ = dataset[k]

        #Copy network on cpu
        model.to(device)

        #Reshape image
        s = d.shape
        d = torch.unsqueeze(d, 0).to(device)
        d1 = torch.unsqueeze(d1, 0).to(device)
        
        d = torch.cat((d,d1),0)

        #Prediction
        cams = model(d)["out"]
        cams = cams.cpu().detach()

        #Transpose for visualization purposes
        d = torch.transpose(d, 1,2).cpu()
        d = torch.transpose(d, 2,3).cpu()
        d = np.array(d)

        if len(d[0,:,:,:]) > 1:
            newIm = np.arctan2(d[0,:,:,0], d[0,:,:,1])
            newIm = (newIm + np.pi) / (2*np.pi)
        else:
            newIm = d
        
        #Save images separately
        if saveSep :
            cv2.imwrite(dir + "solo" + str(k) + "_" + str(epoch) + "pred_seg.png", np.array(cams[0,0,:,:]*255).astype(int))
            cv2.imwrite(dir + "solo" + str(k) + "_" + str(epoch) + "real_seg.png", np.array(l[0,:,:]*255).astype(int))
            cv2.imwrite(dir + "soloInp" + str(k) + "_" + str(epoch) + ".png", np.array(newIm*255).astype(int))

        #Input
        plt.figure(figsize=(20,10))
        plt.clf()
        plt.subplot(221)
        plt.imshow(newIm, cmap = "hsv")
        plt.title("Input image")

        # Ground truth
        plt.subplot(222)
        plt.imshow(l[0,:,:])
        plt.title("Ground Truth")

        #Predicted Cam
        plt.subplot(223)
        plt.imshow(cams[0,0,:,:])
        plt.colorbar()
        

        grad = np.gradient((cams[0,0,:,:]> thresh)*1)
        grad = grad[0]**2 + grad[1]**2

        grad = np.tile(np.expand_dims(1*(grad>0), 2),(1,1,4))
        grad[:,:,1:3]*=0


        #Predicted segm (0.5 thresh)
        plt.subplot(224)
        plt.imshow(cams[0,0,:,:])
        plt.imshow(grad*255)

        plt.title("Predicted Segmentation (" +str(thresh)+")" )

        #Save images
        plt.savefig(dir + "im" + str(k) + "_" + str(epoch))
        plt.close()