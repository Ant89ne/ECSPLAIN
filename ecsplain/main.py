#########################################################
#                      LIBRARIES                        #
#########################################################

#Home made libraries
from src.movingAreasDataLoader import MA_Truth_Dataset
from src.routine import training_routine, evaluation_routine
from src.utils import checkDir, Identity
from src.visualization import visGraph, visIm
from src.cams import scoreCAM, gradCAM, layerCAM, normCAM

#Pytorch libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models import resnet50, resnet18

#Standard libraries
import numpy as np
import random
import argparse
from datetime import datetime
import time

#########################################################
#                   REPRODUCTIBILITY                    #
#########################################################

#Get user parameters (optionnal)
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--cam_type", type = str, default="grad_cam")
parser.add_argument("--wSegm", type = float, default=0)
parser.add_argument("--wClass", type = float, default=1)
parser.add_argument("--nb_epochs", type = int, default = 50)
parser.add_argument("--lr", type = float, default = 0.001)
parser.add_argument("--batch_size", type = int, default = 8)
parser.add_argument("--model", type = str, default = '')
parser.add_argument("--segLoss", type = str, default = "BCE")
parser.add_argument("--model_type", type = str, default="resnet18")

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
#                    HYPERPARAMETERS                    #
#########################################################

#Training hyperparameters
epochsNb=args.nb_epochs                 #Number of training epochs
batchsize = args.batch_size             #Batch size
lr = args.lr                            #Learning rate

lambdaSegm = args.wSegm                 #Weight for the segmentation loss               
lambdaClass = args.wClass               #Weight for the classification loss
   

#########################################################
#                   AUTOMATIC PARAMS                    #
#########################################################

#Select the device on which the network is trained
device = "cuda" if (torch.cuda.is_available()) else "cpu"      #Whether to use gpu or cpu
print(device)

#Create the saving directory for the running experiment
currDate = datetime.now()
saveDir += str(currDate).replace(' ', '_') + args.cam_type + '_' + args.segLoss + "/"
checkDir(saveDir)


#########################################################
#                   DATASETS                            #
#########################################################

#Training dataset and dataloader
dataset_t = MA_Truth_Dataset(path1=path1, path0=path0, stock = False)
dataloader_t = DataLoader(dataset_t, batch_size=batchsize, shuffle=True, num_workers=nw, drop_last=True)

#Evaluation dataset and dataloader
dataset_e = MA_Truth_Dataset(path1=path1, path0=path0, typedataset=1, stock = False)
dataloader_e = DataLoader(dataset_e, batch_size=batchsize, shuffle=True, num_workers=nw, drop_last=True)

#Get the size of the input images
inputSize = dataset_t.sizeZone


#########################################################
#                     DEEP MODEL                        #
#########################################################

#Create the deep model based on user choices
if not len(args.model):
    if args.model_type == "resnet18":
        initModel = resnet18(norm_layer = Identity)
        initModel.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        initModel.fc = nn.Linear(512, 1)
    elif args.model_type == "resnet50":
        initModel = resnet50(norm_layer = Identity)
        initModel.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        initModel.fc = nn.Linear(2048, 1)  
else:
    initModel = torch.load(args.model, map_location=device)


xaiModel = create_feature_extractor(initModel, return_nodes = {"maxpool":"c1", "layer2.0.conv1":"conv2","layer3.0.conv1":"conv3","layer4.0.conv1":"conv4","layer4.1.conv2":"conv5","fc":"classif"})


for k in xaiModel.parameters():
    k.requires_grad = True

#Send the model to the training device
xaiModel = xaiModel.to(device)

#########################################################
#                     CAM TO USE                        #
#########################################################

if args.cam_type == "grad_cam":
    cam_fn = gradCAM
elif args.cam_type == "layer_cam":
    cam_fn = layerCAM
elif args.cam_type == "score_cam":
    cam_fn = scoreCAM
elif args.cam_type == "norm_cam":
    cam_fn = normCAM

#########################################################
#                    OPTIMIZATION                       #
#########################################################

#Optimizer
optimizer = Adam(xaiModel.parameters(), lr=lr, betas=(0.5,0.999))

#Losses for the training
if args.segLoss == "BCE":
    lossSegm = nn.BCELoss()       
elif args.segLoss == "MSE":
    lossSegm = nn.MSELoss()
lossClassif = nn.BCEWithLogitsLoss()           
lossDict = {"Classif": lossClassif, "Segm": lossSegm, "WClassif": lambdaClass, "WSegm": lambdaSegm}

#########################################################
#                   MAIN ROUTINE                        #
#########################################################

# Initializations
loss = []
classif_loss = []
segm_loss = []
allMets = {"Accuracy": [], "Precision": [], "Recall": [], "F1Score": [], "MAE": [], "Dice Index": [], "Hausdorff Distance": [], }

# Initial network performances
classif_t, segm_t, total_t, met_t = evaluation_routine(dataloader_t, xaiModel, cam_fn, lossDict, device)
classif_e, segm_e, total_e, met_e = evaluation_routine(dataloader_e, xaiModel, cam_fn, lossDict, device)

#Save metrics and losses
loss.append([total_t, total_e])
classif_loss.append([classif_t, classif_e])
segm_loss.append([segm_t, segm_e])
for i,k in enumerate(allMets.keys()):
    allMets[k].append([met_t[i], met_e[i]])


# Training routine
for epoch in range(epochsNb):
    t0 = time.time()    
    print(f"\n***********\n* Epoch {epoch} *\n***********\n")
    # Training step
    classif_t, segm_t, total_t, met_t = training_routine(dataloader_t, xaiModel, cam_fn, lossDict, optimizer, device)
    print(f"\tTotal Loss : {total_t}\t\t|\tClassif Loss : {classif_t}\t|\tSegm Loss : {segm_t}")
    print(f"\tAccuracy : {met_t[0]}\t\t|\tPrecision : {met_t[1]}\t|\tRecall : {met_t[2]}\t|\tF1 Score : {met_t[3]}")
    print(f"\tMAE : {met_t[4]}\t|\tDice : {met_t[5]}\t\t|\tHausdorff Dist : {met_t[6]}")
    

    # Evaluation step
    print("\nEvaluation")
    classif_e, segm_e, total_e, met_e = evaluation_routine(dataloader_e, xaiModel, cam_fn, lossDict, device)
    print(f"\tTotal Loss : {total_e}\t\t|\tClassif Loss : {classif_e}\t|\tSegm Loss : {segm_e}")
    print(f"\tAccuracy : {met_e[0]}\t\t|\tPrecision : {met_e[1]}\t|\tRecall : {met_e[2]}\t|\tF1 Score : {met_e[3]}")
    print(f"\tMAE : {met_e[4]}\t|\tDice : {met_e[5]}\t\t|\tHausdorff Dist : {met_e[6]}")

    # Save losses and metrics
    loss.append([total_t, total_e])
    classif_loss.append([classif_t, classif_e])
    segm_loss.append([segm_t, segm_e])
    for i,k in enumerate(allMets.keys()):
        allMets[k].append([met_t[i], met_e[i]])

    visGraph(np.array(loss), np.array(classif_loss), np.array(segm_loss), allMets, filename = saveDir + 'Loss.png')

    #Checkpoint every 10 epochs
    if (epoch+1) % 2 == 0 :
        torch.save(xaiModel, f'{saveDir}/model{epoch+1}.pth')
        visIm(xaiModel, cam_fn, dataset_e, epoch, saveDir + "/Eval/", nbIm = 20, saveSep = False, device = device)
        visIm(xaiModel, cam_fn, dataset_t, epoch, saveDir + "/Train/", nbIm = 20, saveSep = False, device = device)

    t1 = time.time()
    print("Elapsed time: ", t1-t0)
