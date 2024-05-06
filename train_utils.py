import torch 
from torch import nn 

def criterion(input, target):
    pass 


def evaluate(model, data_loader, device, num_class):
    pass 


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, lr_scheduler, print_freq=10, scaler=None):
    pass


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    pass
