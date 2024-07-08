""" MIT License """
'''
    Project: PulmonaryMAE
    Authors: Axel Masquelin
    Description:
'''
# Libraries and Dependencies
# --------------------------------------------
from sklearn.metrics import roc_curve, confusion_matrix
from RadiomicConcept.utils.metrics import plot_confusion_matrix, mean_delta, concept_accs
from RadiomicConcept.utils.utils import select_model

import torch.optim as optim
import torch.nn as nn
import torchvision
import torch

import numpy as np
import sys, os
import time
# --------------------------------------------

class Trials():
    """
    Description: Pytorch Trail class that handles training and validation of provided models
    """
    def __init__(self, fold:int, model:nn.Module, config:dict):
        """
        Initialization function for Training and Validation Environment
        -----------
        Parameters:
            fold - int:
                integer defining the current fold of the K-fold cross-validation if utilized.
                If no K-fold cross validation is being use, will just return 1
            model - nn.Module:
                Variable storing the Model class
            config - dict:
                dictionary containing all network parameters
        """
        
        # General Variablels

        # Loss Functions

        # Optimizers

    def _progress_(self):
        """
        Description: Progress bar providing information on current epoch, and network training
        and validation performance
        -----------
        Parameters:
        --------
        Returns:
        """
        pass
    
    def _update_(self, loss, component:str):
        """
        Function to update weights
        -----------
        Parameters:
            loss - :
                loss performance of the model
            component - str:
                string defining which tasks will be updated
        """

    def _batches_(self, loader:dict, training:bool=False):
        """
        Batch function for training and validation
        -----------
        Parameters:
            loader - dict:
                loader containing images
            training - bool:
                bool value that checks whether weights should be updated
        --------
        Returns:
        """
        for i, data in enumerate(loader):
            input = data['input'].to(self.device, dtype=torch.float)
            labels = data['labels'].to(self.device)
            
            output = self.model(input)
            loss = self.taskloss(output, labels)
            if training: self._update_(loss, component='classifier')
            
    def training(self, loader:dict):
        """
        Training Function for the model
        -----------
        Parameters:
            loader - dict:
                dictionary containing data
        """
        pass

    def validation(self, loader:dict):
        """
        Training Function for the model
        -----------
        Parameters:
            loader - dict:
                dictionary containing data
        """
        pass