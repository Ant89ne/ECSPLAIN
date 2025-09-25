import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np

def scoreCAM(inputs, model, batch_size = 4, device = "cpu", training = True, select = True):

    dropFeats = 0.25

    all_outputs = model(inputs)
    score = all_outputs["classif"]

    all_cams = torch.zeros((inputs.shape[0],1,inputs.shape[-2], inputs.shape[-1])).to(device)

    for key in all_outputs.keys():
        if key == "classif":
            continue

        midFeats = all_outputs[key]

        # Feature dropout to compute the ScoreCAM during training
        if select:
            choice = np.random.choice(midFeats.shape[1], int(dropFeats*midFeats.shape[1])).astype(np.int16)

            midFeats = midFeats[:,choice,:,:]

        upFeats = nn.Upsample((inputs.shape[-2], inputs.shape[-1]), mode = "bilinear")(midFeats)

        maxis = torch.max(torch.max(upFeats, dim = (-1)).values, dim = -1).values
        minis = torch.min(torch.min(upFeats, dim = (-1)).values, dim = -1).values

        upFeatsNorm = (upFeats - minis[:,:,None,None]) / (maxis[:,:,None,None] - minis[:,:,None,None] + 1e-6)

        masked = inputs[:,None,:,:,:] * upFeatsNorm[:,:,None,:,:]

        for batch in range(len(masked)):
            weights = torch.zeros((len(masked[batch]),1)).to(device)
            for m in range(0,len(masked[batch]),batch_size):
                weights[m:min(m+batch_size, len(weights)),:] = model(masked[batch,m:min(m+batch_size,len(masked[batch])),:,:,:])["classif"]
            
            weights -= score[batch]
            weights = nn.Softmax(0)(weights)
            
            cam = midFeats[batch:batch+1,:,:,:]*weights[:,:,None]
            cam = torch.sum(cam,1, keepdim=True)
            cam = nn.ReLU()(cam)
            cam = nn.Upsample((inputs.shape[-2], inputs.shape[-1]), mode = "bilinear")(cam)
            cam = (cam - torch.min(cam))/(torch.max(cam) - torch.min(cam)+ 1e-7)
            all_cams[batch:batch+1,:,:,:] += cam

    return all_cams/(len(all_outputs.keys())-1), score
        
def gradCAM(inputs, model, device, training = True):
    score = model(inputs)

    
    all_cams = torch.zeros((inputs.shape[0],1,inputs.shape[-2], inputs.shape[-1])).to(device)
    
    grads = torch.autograd.grad([score["classif"][k] for k in range(len(score["classif"]))], score.values(), retain_graph=training, create_graph=training)

    for m, map in enumerate(grads[:-1]):

        alpha_map = torch.mean(torch.mean(map, dim = -1), dim = -1)

        cam = alpha_map[:,:,None,None]*score[list(score.keys())[m]]
        cam = torch.sum(cam, dim = 1, keepdim = True)
        cam = nn.ReLU()(cam)
        cam =  nn.Upsample((inputs.shape[-2], inputs.shape[-1]), mode = "bilinear")(cam)
        
        minis = torch.min(torch.min(cam, dim = -1).values, dim = -1).values
        maxis = torch.max(torch.max(cam, dim = -1).values, dim = -1).values
        cam = (cam - minis[:,:,None,None])/( maxis[:,:,None, None] - minis[:,:,None,None]+ 1e-7)
        
        all_cams += cam

    return all_cams / (len(score.keys())- 1), score["classif"]



def layerCAM(inputs, model, device, training = True):
    
    score = model(inputs)
    
    all_cams = torch.zeros((inputs.shape[0],1,inputs.shape[-2], inputs.shape[-1])).to(device)
    
    grads = torch.autograd.grad([score["classif"][k] for k in range(len(score["classif"]))], score.values(), retain_graph=training, create_graph=training)
    
    for m, map in enumerate(grads[:-1]):

        cam = nn.ReLU()(map) * score[list(score.keys())[m]]
        cam = torch.sum(cam, dim = 1, keepdim = True)
        cam = nn.ReLU()(cam)
        cam =  nn.Upsample((inputs.shape[-2], inputs.shape[-1]), mode = "bilinear")(cam)
        
        minis = torch.min(torch.min(cam, dim = -1).values, dim = -1).values
        maxis = torch.max(torch.max(cam, dim = -1).values, dim = -1).values
        cam = (cam - minis[:,:,None,None])/( maxis[:,:,None, None] - minis[:,:,None,None]+ 1e-7)

        all_cams += cam

    return all_cams / (len(score.keys())- 1), score["classif"]


def normCAM(inputs, model, device, training = True):
    
    score = model(inputs)
    
    all_cams = torch.zeros((inputs.shape[0],1,inputs.shape[-2], inputs.shape[-1])).to(device)
    
    grads = torch.autograd.grad([score["classif"][k] for k in range(len(score["classif"]))], score.values(), retain_graph=training, create_graph=training)
      
    for m, map in enumerate(grads[:-1]):
        
        cam = nn.ReLU()(map * score[list(score.keys())[m]])
        cam = torch.norm(cam, p=2,dim = 1, keepdim = True)
        cam = nn.ReLU()(cam)
        cam = nn.Upsample((inputs.shape[-2], inputs.shape[-1]), mode = "bilinear")(cam)

        minis = torch.min(torch.min(cam, dim = -1).values, dim = -1).values
        maxis = torch.max(torch.max(cam, dim = -1).values, dim = -1).values
        cam = (cam - minis[:,:,None,None])/( maxis[:,:,None, None] - minis[:,:,None,None]+ 1e-7)

        all_cams += cam

    return all_cams / (len(score.keys())- 1), score["classif"]
