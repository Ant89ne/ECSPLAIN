#########################################################
#                      LIBRARIES                        #
#########################################################

#Home made libraries
from src.movingAreasDataLoader import MA_Truth_Dataset
from src.routine import test_routine
from src.visualization import visIm
from src.cams import scoreCAM, gradCAM, layerCAM, gradCAMplusplus, normCAM


#Pytorch libraries
import torch
from torch.utils.data import DataLoader
from torchvision.models.feature_extraction import create_feature_extractor

#Standard libraries
import numpy as np
import random
import argparse
from tqdm import trange

#########################################################
#                   REPRODUCTIBILITY                    #
#########################################################

#Get user parameters (optionnal)
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--cam_type", type = str, default="grad_cam")
parser.add_argument("--model", type = str)
parser.add_argument("--cuda", action="store_true", default=False)
parser.add_argument("--save_ims", action="store_true", default=False)
parser.add_argument("--thresh", type = float, default=0)
parser.add_argument("--threshNb", type = int, default=0)


args = parser.parse_args()

#Set the seed for the current experiment
seed = args.seed
torch.random.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


#########################################################
#                   PARAMS TO BE CHOSEN                 #
#########################################################

#Path to moving areas images
path1 = ""
#Path to no move areas images
path0 = ""

#Path to save results
saveDir = ""

#Number of workers for the training
nw = 15
   

#########################################################
#                   AUTOMATIC PARAMS                    #
#########################################################

#Select the device on which the network is trained
device = "cuda" if (torch.cuda.is_available() and args.cuda) else "cpu"      #Whether to use gpu or cpu
print(device)


#########################################################
#                   DATASETS                            #
#########################################################

#Evaluation dataset and dataloader
dataset_e = MA_Truth_Dataset(path1=path1, path0=path0, typedataset=1)
dataloader_e = DataLoader(dataset_e, batch_size=1, shuffle=True, num_workers=nw, drop_last=True)


#########################################################
#                     DEEP MODEL                        #
#########################################################

#Create the deep model based on user choices
initModel = torch.load(args.model, map_location= device)
xaiModel = create_feature_extractor(initModel, return_nodes = {"maxpool":"c1", "layer2.0.conv1":"conv2","layer3.0.conv1":"conv3","layer4.0.conv1":"conv4","layer4.1.conv2":"conv5","fc":"classif"})

#Send the model to the training device
xaiModel = xaiModel.to(device)
xaiModel.eval()

#########################################################
#                     CAM TO USE                        #
#########################################################

if args.cam_type == "grad_cam":
    cam_fn = gradCAM
elif args.cam_type == "grad_cam_plus_plus":
    cam_fn = gradCAMplusplus
elif args.cam_type == "layer_cam":
    cam_fn = layerCAM
elif args.cam_type == "score_cam":
    cam_fn = scoreCAM
elif args.cam_type == "norm_cam":
    cam_fn = normCAM

#########################################################
#                   MAIN ROUTINE                        #
#########################################################

# If a given threshold is provided, compute the metrics with the given threshold
if args.thresh:
    met_e, moremets = test_routine(dataloader_e, xaiModel, cam_fn, device, threshold=args.thresh)
    print(met_e, moremets)

# Else, compute the metrics with different thresholds and give the best threshold for Dice and Hausdorff Distance
elif args.threshNb:
    thr_mets = []
    for k in trange(args.threshNb):
        # print("Threshold value: ", k/args.threshNb)
        thr_mets.append(test_routine(dataloader_e, xaiModel, cam_fn, device, threshold=k/args.threshNb)[0])
    
    thr_mets = np.array(thr_mets)
    
    maxiDice = np.argmax(thr_mets[:,5])
    miniHaus = np.argmin(thr_mets[:,6])
    
    print(f"Dice max: {maxiDice/args.threshNb} --> {thr_mets[maxiDice]}")
    print(f"Hausdorff min: {miniHaus/args.threshNb} --> {thr_mets[miniHaus]}")

    
if args.save_ims:
    visIm(xaiModel, cam_fn, dataset_e, -1, saveDir + "/Eval/", nbIm = 200, saveSep = True, device = device, thresh=args.thresh)



