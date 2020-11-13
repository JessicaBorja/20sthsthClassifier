
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
import torchvision.transforms.functional as TF
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.transforms import Grayscale
from classifier import ResNet18LSTM
from omegaconf import DictConfig, OmegaConf
import os, yaml
import numpy as np 
from datasets import SthSthDataset
import datetime, time
import sys

def save_only_best(epoch, model, optimizer, models_folder, logger,\
                    train_loss, best_train_loss, val_loss, best_val_loss,\
                    train_acc, best_train_acc, val_acc, best_val_acc):
    if(train_loss<best_train_loss):
        logger.info('[Epoch %d] saving best train loss model: %.3f' % (epoch, train_loss))
        save(epoch, model, optimizer, models_folder ,"best_train_loss")
        best_train_loss = train_loss
    if(val_loss<best_val_loss):
        logger.info('[Epoch %d] saving best validation loss model: %.3f' % (epoch, val_loss))
        save(epoch, model, optimizer, models_folder ,"best_val_loss")
        best_val_loss = val_loss
    #Best accuracy
    if(train_acc>best_train_acc):
        logger.info('[Epoch %d] saving best train accuracy model: %.3f' % (epoch, train_acc))
        save(epoch, model, optimizer, models_folder ,"best_train_acc")
        best_train_acc = train_acc
    if(val_acc>best_val_acc):
        logger.info('[Epoch %d] saving best validation accuracy model: %.3f' % (epoch, val_acc))
        save(epoch, model, optimizer, models_folder ,"best_val_acc")
        best_val_acc = val_acc
    return best_train_loss, best_val_loss, best_train_acc, best_val_acc

def save(epoch, model, optim, folder, name):
    save_dict = {"model_state_dict": model.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
                "epoch": epoch}
    filename = os.path.join(folder, name+".pth")
    torch.save(save_dict, filename)
    print("file saved to: %s"%filename)

def load(path, model, optimizer, train=True):
    epoch = 0
    try:
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        if(train):
            model.train()
        else:
            model.eval()
        print("load succesful!: %s"%path)
    except:
        print("Error in loading: ", sys.exc_info()[0])
    
    return epoch

def resume_training(hydra_folder, model_name):
    cfg  = yaml.load(open( "%s/.hydra/config.yaml"%hydra_folder, 'r'))
    cfg["model"]["save_dir"] = cfg["models_folder"]
    model = ResNet18LSTM(**cfg["model"]).cuda()
    optimizer = torch.optim.SGD(model.parameters(), **cfg["optim"]) #cfg.lr
    models_path = "%s/trained_models/%s"%(hydra_folder, model_name)
    epoch = load(models_path, model, optimizer, train=True)
    return model, optimizer, epoch