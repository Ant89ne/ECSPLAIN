import torch
from torch import nn

import numpy as np
import math

import monai.metrics as mmet
import torchmetrics as tmet

def getMetrics(preds, truth, camPred, camTruth, threshold = 0.5):
    """
    Function which evaluates the predictions of the network in terms of classification and segmentation

    @input preds:               Classification prediction
    @input truth:               Classification labels
    @input camPred:             Segmentation generated (CAM)
    @input camTruth:            Segmentaion labels

    @return:                    Array of Accuracy, Precision, Recall, F1-Score, MAE, Dice and Hausdorff distance
    """

    with torch.no_grad():

        #Prepare data
        sout = preds.shape
        predictions = preds.squeeze()
        if sout[0] == 1 :
            predictions = predictions.unsqueeze(0)

        sout2 = camTruth.shape
        trueSeg = camTruth.detach().cpu().squeeze()
        if sout2[0] == 1 :
            trueSeg = trueSeg.unsqueeze(0)

        sout3 = camPred.shape
        predSeg = camPred.detach().cpu().squeeze()
        if sout3[0] == 1 :
            predSeg = predSeg.unsqueeze(0)

        predictions = nn.Sigmoid()(predictions) > 0.5


        #Classification metrics
        corrects = (predictions == truth)

        true_pos = torch.sum(corrects[truth == 1] == 1)
        true_neg = torch.sum(corrects[truth == 0] == 1)
        false_pos = torch.sum(corrects[truth == 0] == 0)
        false_neg = torch.sum(corrects[truth == 1] == 0)

        

        #Segmentation metrics

        #MAE
        MeanAbs = nn.L1Loss()
        mae = MeanAbs(camPred, camTruth)

        nbones = torch.sum(truth)

        if not nbones:
            diceScore = torch.zeros(1)
            hausdorffDist = torch.zeros(1)
        else : 
            camTruth = trueSeg.unsqueeze(1) > 0
            camPred = predSeg.unsqueeze(1) > threshold

            s = camPred.shape

            test = torch.where(torch.sum(camPred, (1,2,3)) == 0)

            for k in test:
                camPred[k,0,0,0] = 1

            #Dice Score
            diceScore = mmet.DiceMetric(reduction="none")(camPred,camTruth)
            
            #Hausdorff Distance
            hausdorffDist = mmet.compute_hausdorff_distance(camPred, camTruth)

            infs = torch.isinf(hausdorffDist)
            if torch.sum(infs) > 0 :
                hausdorffDist[infs] = math.sqrt(s[-2]**2 + s[-1]**2)

            diceScore = torch.sum(diceScore[truth.cpu()==1])

            hausdorffDist = torch.sum(hausdorffDist[truth.cpu()==1])


    return np.array([true_pos.cpu().item(), true_neg.cpu().item(), false_pos.cpu().item(), false_neg.cpu().item(), mae.cpu().item(), diceScore.cpu().item(), hausdorffDist.cpu().item(), nbones.cpu().item()])

def compute_classif_mets(tp, tn, fp, fn):

    accuracy = (tp + tn)/(tp+tn+fp+fn)

    precision = tp / (tp+fp)

    recall = tp / (tp+fn)

    f1score = (2*precision*recall)/(precision+recall)

    return accuracy, precision, recall, f1score

def compute_seg_mets(pred,truth, thresh = 0.5, device = "cpu"):

    with torch.no_grad():
        nbones = torch.sum(truth)
        if not nbones:
            iou = torch.zeros(1)
            auroc = torch.zeros(1)
        else:
            iouOp = tmet.JaccardIndex(task = "binary").to(device)
            iou = iouOp(pred > thresh, truth >0)

            aurocOp = tmet.AUROC(task = "binary", thresholds = 100).to(device)
            auroc = aurocOp(pred, truth)

        return iou.cpu().item(), auroc.cpu().item()