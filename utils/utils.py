
import torch
import torch.nn as nn
from torch.utils import data
from torchvision import transforms
import torchvision.models as models
import torchvision.transforms.functional as TF
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.transforms import Grayscale
from models.resnet_lstm import ResNetLSTM
from models.frame_lstm import FrameLSTM
from omegaconf import DictConfig, OmegaConf
import os, yaml
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from datasets import SthSthDataset
import datetime, time
import sys
from hydra.core.hydra_config import HydraConfig
import hydra

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
    if(cfg["model_name"] == "FrameLSTM"):
        model = FrameLSTM(**cfg["model_cfg"]).cuda()
    else:
        model = ResNetLSTM(**cfg["model_cfg"]).cuda()
    optimizer = torch.optim.SGD(model.parameters(), **cfg["optim"]) #cfg.lr
    models_path = "%s/trained_models/%s"%(hydra_folder, model_name)
    epoch = load(models_path, model, optimizer, train=True)
    return model, optimizer, epoch

#map labels of 78 to 174 original classes
def map_orig2new():
    keep_ids = [5,6] + list(range(8,25)) + [27,29] +\
        list(range(40,48)) + [49] +  list(range(53,59)) +\
        [60,62,69,83] + list(range(85,90)) + list(range(93,97))+\
        list(range(98,102)) + list(range(104,111))+\
        [118,121,122,123,129,130,148,151,152] +\
        list(range(155,161)) + [164,170,171,172,173]
    keep_ids.remove(159)
    keep_ids.remove(108)
    new_ids_lst = list(range(len(keep_ids)))
    old2new = dict(zip(keep_ids,new_ids_lst))
    return old2new

#map labels of 78 to 174 original classes
def map_new2orig():
    keep_ids = [5,6] + list(range(8,25)) + [27,29] +\
        list(range(40,48)) + [49] +  list(range(53,59)) +\
        [60,62,69,83] + list(range(85,90)) + list(range(93,97))+\
        list(range(98,102)) + list(range(104,111))+\
        [118,121,122,123,129,130,148,151,152] +\
        list(range(155,161)) + [164,170,171,172,173]
    keep_ids.remove(159)
    keep_ids.remove(108)
    new_ids_lst = list(range(len(keep_ids)))
    new2old = dict(zip(new_ids_lst, keep_ids))
    return new2old

def id2str(id, new2oldDict, id2strDict):
    full_classes_id = new2oldDict[id]
    str_class = id2strDict[full_classes_id]
    return str_class

def get_class_dist(labels_file = "78-classes_train.json", str2id_file = "78-classes_labels.json"):
    labels_dir = "./datasets/20bn-sth-sth-v2/labels/"
    if HydraConfig.initialized():
        labels_dir = os.path.join(hydra.utils.get_original_cwd(), labels_dir)
        data_info_path = os.path.join(labels_dir, labels_file)
    else:
        labels_dir = os.path.join(os.getcwd(), labels_dir)
        data_info_path = os.path.abspath(os.path.join(labels_dir, labels_file))
    data_info_frame = pd.read_json(data_info_path)
    ids_frame = pd.read_json(os.path.join(labels_dir, str2id_file),\
                             typ='series')
    data_info_frame["template"] = data_info_frame["template"].str.replace("[", "")
    data_info_frame["template"] = data_info_frame["template"].str.replace("]", "")
    #classes present in labels_file (78)
    classes = ids_frame[data_info_frame["template"]]
    unique, counts = np.unique(classes, return_counts=True)
    old2new = map_orig2new()
    #str: new id
    classes = ids_frame[unique].replace(old2new)
    #returns
    unique_str = ids_frame[unique]
    unique_new_ids = [ old2new[i] for i in unique]
    #new_ids: count
    new_ids_counts = dict(zip(unique_new_ids, counts))
    #str_count:
    str_counts = dict(zip(unique_str, counts))
    #new_ids in order of the dataframe
    class_labels = list(classes[data_info_frame["template"]])
    return new_ids_counts, str_counts, class_labels