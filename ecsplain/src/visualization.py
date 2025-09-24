import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import trange
import cv2

from src.utils import checkDir


def visGraph(losses, classifLoss, segmLoss, metrics, filename = "Loss.png"):

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


def visIm(model, cam_fn, dataset, epoch, dir, nbIm = 4, saveSep = False, device = "cpu", thresh = 0.5):
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

        #Copy network and data on device
        model.to(device)
        d = torch.unsqueeze(d, 0).to(device)
        d1 = torch.unsqueeze(d1, 0).to(device)
        d = torch.cat((d,d1),0)

        #Prediction
        cams, preds = cam_fn(d, model, device = device)
        cams = cams.cpu().detach()
        preds = preds.cpu().detach()

        #Transpose for visualization purposes
        d = torch.transpose(d, 1,2).cpu()
        d = torch.transpose(d, 2,3).cpu()
        d = np.array(d)

        #Compute the phase value
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
        plt.title(f"Predicted CAM (detection : {preds[0,0].item()})")
        plt.colorbar()
        
        # Compute the boundaries of the detection based on gradient
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
