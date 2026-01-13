from tqdm import tqdm
from src.metricsEvals import getMetrics, compute_classif_mets, compute_seg_mets
import numpy as np
import torch
from copy import deepcopy
from torch import nn

from src.cams import scoreCAM

#########################################################
#                   TRAINING ROUTINE                    #
#########################################################

def training_routine(dataloader, model, cam_fn, losses, optimizer, device):
    # Set model to training configuration
    model.train()

    #Initialization
    meanClassif, meanSegm, meanTot = 0,0,0
    allmets = np.zeros(8)
    s = len(dataloader)

    #Routine
    for k in tqdm(dataloader):

        #Send data to the selected device
        inputImgs = k[0].to(device)
        labels = k[1].to(device)
        segms = k[2].to(device)

        all_cams, all_scores = cam_fn(inputImgs, model, device = device, training = True)

        # Classification loss calculation
        classifLoss = losses["Classif"](all_scores.squeeze(), labels)

        # Segmentation loss calculation
        if type(losses["Segm"]) == nn.BCELoss:
            trueSegm = ((segms>0)*1).to(torch.float32)
        else:
            trueSegm = segms

        segmLoss = losses["Segm"](all_cams, trueSegm)

        # Total Loss calculation
        totalLoss = losses["WClassif"] * classifLoss + losses["WSegm"] * segmLoss
        

        # Set gradients to 0
        optimizer.zero_grad()

        # Backward pass
        totalLoss.backward()
       
        # Optimization
        optimizer.step()

        #Save metrics and losses
        meanClassif += classifLoss.item()
        meanSegm += segmLoss.item()
        meanTot += totalLoss.item()
        allmets += getMetrics(all_scores, labels, all_cams, segms)

    allmets[0:4] = compute_classif_mets(allmets[0], allmets[1], allmets[2], allmets[3])
    
    #Mean of Dice and Hausdorff distance over the true moves (not calculated when no move appears)
    allmets[-3:-1] /= allmets[-1]
    allmets[4] /= s

    return deepcopy(meanClassif/s), deepcopy(meanSegm/s), deepcopy(meanTot/s), allmets

#########################################################
#                  EVALUATION ROUTINE                   #
#########################################################

def evaluation_routine(dataloader, model, cam_fn, losses, device):

    # Set model to training configuration
    model.eval()

    #Initialization
    meanClassif, meanSegm, meanTot = 0,0,0
    allmets = np.zeros(8)
    s = len(dataloader)

    #Routine    
    for k in tqdm(dataloader):

        # Send data to device
        inputImgs = k[0].to(device)
        labels = k[1].to(device)
        segms = k[2].to(device)

        if cam_fn == scoreCAM:
            all_cams, all_scores = cam_fn(inputImgs, model, device = device)
        else: 
            all_cams, all_scores = cam_fn(inputImgs, model, device = device, training = False)

        # Classification loss calculation
        classifLoss = losses["Classif"](all_scores.squeeze(), labels)
        
        # Segmentation loss calculation
        if type(losses["Segm"]) == nn.BCEWithLogitsLoss:
            trueSegm = ((segms>0)*1).to(torch.float32)
        else:
            trueSegm = segms

        segmLoss = losses["Segm"](all_cams, trueSegm)

        # Total Loss calculation
        totalLoss = losses["WClassif"] * classifLoss + losses["WSegm"] * segmLoss

        #Save losses and metrics
        meanClassif += deepcopy(classifLoss.item())
        meanSegm += deepcopy(segmLoss.item())
        meanTot += deepcopy(totalLoss.item())
        allmets += deepcopy(getMetrics(all_scores, labels, all_cams, segms))

        
    #Mean of Dice and Hausdorff distance over the true moves (not calculated when no move appears)
    allmets[-3:-1] /= allmets[-1]
    allmets[0:4] = compute_classif_mets(allmets[0], allmets[1], allmets[2], allmets[3])
    allmets[4] /= s
    
    return deepcopy(meanClassif/s), deepcopy(meanSegm/s), deepcopy(meanTot/s), allmets



#########################################################
#                    TEST ROUTINE                       #
#########################################################

def test_routine(dataloader, model, cam_fn, device, threshold = 0.5):

    # Set model to training configuration
    model.eval()

    #Initialization
    allmets = np.zeros(8)
    moremets = np.zeros(2)
    s = len(dataloader)

    #Routine
    for k in dataloader:
        # Send data to device
        inputImgs = k[0].to(device)
        labels = k[1].to(device)
        segms = k[2].to(device)

        if cam_fn == scoreCAM:
            with torch.no_grad():
                all_cams, all_scores = cam_fn(inputImgs, model, device = device, select = False)
        else: 
            all_cams, all_scores = cam_fn(inputImgs, model, device = device)
        model.eval()

        allmets += deepcopy(getMetrics(all_scores, labels, all_cams, segms, threshold=threshold))
        moremets += deepcopy(compute_seg_mets(all_cams, segms, thresh= threshold, device = device))


    #Mean of Dice and Hausdorff distance over the true moves (not calculated when no move appears)
    allmets[-3:-1] /= allmets[-1]
    allmets[0:4] = compute_classif_mets(allmets[0], allmets[1], allmets[2], allmets[3])
    allmets[4] /= s
    moremets /= allmets[-1]


    return deepcopy(allmets[:-1]), deepcopy(moremets)



