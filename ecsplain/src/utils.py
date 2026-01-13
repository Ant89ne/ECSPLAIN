import os
import shutil
from torch import nn

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

