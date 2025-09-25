######################################################
#                    LIBRAIRIES                      #
######################################################

from src.metricsEval import getMetrics, compute_classif_mets, compute_seg_mets
from src.utils import visImInference
from src.movingAreasDataLoader import MA_Truth_Dataset

import torch
from torch.utils.data import DataLoader

import numpy as np
import random

from tqdm import tqdm
from copy import deepcopy

######################################################
#                 REPRODUCTIBILITY                   #
######################################################

seed = 1
torch.random.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
device = "cuda"

#########################################################
#                   PARAMS TO BE CHOSEN                 #
#########################################################

#Path to moving areas images
path1 = ""
#Path to no move areas images
path0 = ""
#Path to the model to be loaded
modelPath = ""
#Path to save results
saveDir = modelPath[:modelPath.rfind("/")] + "/ImsFinal/" 

#########################################################
#                   DATASETS                            #
#########################################################

#Create evaluation dataset and dataloader
dataset_e = MA_Truth_Dataset(path1=path1, path0=path0, typedataset=1, stock = False)
dataloader_e = DataLoader(dataset_e, batch_size=1, shuffle=True)

#########################################################
#                   DEEP MODEL                          #
#########################################################

#Load the pre-trained model
model = torch.load(modelPath, map_location=device)

#########################################################
#                   MAIN ROUTINE                        #
#########################################################

maxiDice = 0
finalMets = None
finalMoreMets = None
bestT = 0
nbT = 100

for k in range(nbT+1):
    threshold = k/nbT
    #Metrics initialization
    s = len(dataloader_e)
    nbs = 100
    allmets = np.zeros(8)
    moremets = np.zeros(2)

    for d,l,seg in tqdm(dataloader_e) :
        d = d.to(device)
        l = l.to(device)
        seg = seg.to(device)

        #Network prediction
        pred = model(d)

        #Metrics calculation
        allmets += deepcopy(getMetrics(pred["out"][0], l, seg, device, threshold=threshold))
        moremets += deepcopy(compute_seg_mets(pred["out"][0], seg, l, thresh= threshold, device = device))

    allmets[-3:-1] /= allmets[-1]
    allmets[0:4] = compute_classif_mets(allmets[0], allmets[1], allmets[2], allmets[3])
    allmets[4] /= s
    moremets /= allmets[-1]

    if allmets[-3] > maxiDice:
        maxiDice = allmets[-3]
        bestT = k/nbT
        finalMets = allmets
        finalMoreMets = moremets


print(bestT, finalMets[:-1], finalMoreMets)

#########################################################
#                   VISUALIZATION                       #
#########################################################

visImInference(model, dataset_e, -1, saveDir, 200, True, device, 0.09)

