import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import trange
import cv2

#############################################################
#                    Architectural                          #
#############################################################

class Identity(nn.Module):
    def __init__(self, *kwargs):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    
#############################################################
#                   Folder Management                       #
#############################################################

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

#############################################################
#                    VISUALIZATION                          #
#############################################################

def visGraph(losses, classifLoss, segmLoss, metrics, filename = "Loss.png"):
    """
    Function used to visualize the final loss evolution

    @input losses:           
    @input classifLoss:      
    @input segmLoss:         
    @input metrics:
    """

    nb = [k for k in range(len(losses[:,0]))]

    plt.figure(figsize=(20,10))
    
    #Translation Loss visualization
    plt.subplot(331)
    plt.plot(nb, losses[:,0], 'r-')
    plt.plot(nb, losses[:,1], 'b-')
    plt.title("Loss over epochs")
    plt.legend(["Training", "Evaluation"])
    
    #Classification Loss visualization
    plt.subplot(332)
    plt.plot(nb, classifLoss[:,0], 'r-')
    plt.plot(nb, classifLoss[:,1], 'b-')
    plt.title("Classification Loss over epochs")
    plt.legend(["Training", "Evaluation"])

    #Segmentation Loss visualization
    plt.subplot(333)
    plt.plot(nb, segmLoss[:,0], 'r-')
    plt.plot(nb, segmLoss[:,1], 'b-')
    plt.title("Segmentation Loss over epochs")
    plt.legend(["Training", "Evaluation"])
    
    #Metrics visualization
    for k, met in enumerate(metrics.keys()):
        plt.subplot(3,4, 4+k+1)
        plt.plot(nb, np.array(metrics[met])[:,0], 'r-')
        plt.plot(nb, np.array(metrics[met])[:,1], 'b-')
        plt.title(f"{met} over epochs")
        plt.legend(["Training", "Evaluation"])
    
    plt.savefig(filename)
    plt.close()


def visIm(model, dataset, epoch, dir, nbIm = 4, saveSep = False, device = "cpu"):
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

    with torch.no_grad():        
        print("Save images")
        for k in trange(nbIm):

            #Get an image
            d,lab,l = dataset[k]
            d1,l1,lab = dataset[k]

            #Copy network on cpu
            model.to(device)

            #Reshape image
            d = torch.unsqueeze(d, 0).to(device)
            d1 = torch.unsqueeze(d1, 0).to(device)
            
            d = torch.cat((d,d1),0)

            #Prediction
            _, cams = model(d)

            #Transpose for visualization purposes
            d = torch.transpose(d, 1,2).cpu()
            d = torch.transpose(d, 2,3).cpu()
            d = np.array(d)

            if len(d[0,:,:,:]) > 1:
                newIm = np.arctan2(d[0,:,:,0], d[0,:,:,1])
            else:
                newIm = d
            
            #Save images separately
            if saveSep :
                cv2.imwrite(dir + "solo" + str(k) + "_" + str(epoch) + "pred_seg.png", np.array(cams[0,0,:,:]*255).astype(int))
                cv2.imwrite(dir + "solo" + str(k) + "_" + str(epoch) + "real_seg.png", np.array(l[0,:,:]*255).astype(int))
                cv2.imwrite(dir + "soloInp" + str(k) + "_" + str(epoch) + ".png", np.array(newIm[:,:,0:3]*255).astype(int))

            #Input
            plt.figure(figsize=(20,10))
            plt.clf()
            plt.subplot(221)
            plt.imshow(newIm)
            plt.title("Input image")

            # Ground truth
            plt.subplot(222)
            plt.imshow(l[0,:,:])
            plt.title("Ground Truth")

            #Predicted Cam
            plt.subplot(223)
            plt.imshow(cams[0,0,:,:], cmap = "gray")
            plt.title("Predicted CAM")

            #Predicted segm (0.5 thresh)
            plt.subplot(224)
            plt.imshow(cams[0,0,:,:]>0.5, cmap="gray")
            plt.title("Predicted Segmentation (0.5)")

            #Save images
            plt.savefig(dir + "im" + str(k) + "_" + str(epoch))
            plt.close()
