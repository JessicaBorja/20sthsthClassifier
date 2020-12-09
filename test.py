import torch
import torch.nn as nn
from torch.utils import data
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision import transforms
from models.resnet_lstm import ResNetLSTM
from models.frame_lstm import FrameLSTM
from omegaconf import DictConfig, OmegaConf
import os, yaml
import numpy as np 
import pandas as pd
from datasets import SthSthTestset,SthSthDataset
from utils.utils import load, map_new2orig, id2str
import json, logging
import hydra
logger = logging.getLogger(__name__)

#no labels
def make_predictions(loader, model, base_dir):
    small2largeDict = map_new2orig()
    str2idDict = pd.read_json(\
                      os.path.join("%s/labels/"%base_dir, "something-something-v2-labels.json"),\
                      typ='series')
    id2strDict = {v: k for k, v in str2idDict.items()} #map id to string
    filenames, classes = [], []
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(loader, 0):
            images, filepaths = data
            #cuda
            images = images.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            str_class = id2str(predicted.item(), small2largeDict, id2strDict)
            classes.append(str_class)
            filenames+= [i.split('/')[-1] for i in filepaths]
    predictions = dict(zip(filenames, classes))
    return predictions

def test_model(cfg, models_path):
    #Setup
    model = ResNetLSTM(**cfg.model).cuda()
    optimizer = torch.optim.SGD(model.parameters(), **cfg.optim) #cfg.lr

    epoch = load(models_path, model, optimizer, train=False)
    criterion = nn.CrossEntropyLoss()
    reshape_transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize((64, 64)),
                                    #transforms.Grayscale(),
                                    transforms.ToTensor()])
    base_dir = "../../../datasets/20bn-sth-sth-v2"
    data = SthSthTestset(ids_file = "something-something-v2-test.json",
                         base_dir = base_dir,
                         transform = reshape_transform,
                         n_frames = 8, #frames to pick from each video
                        )
    data_loader = torch.utils.data.DataLoader(data, num_workers = 2, batch_size=1 ,shuffle=False)
    predictions = make_predictions(data_loader, model, base_dir)
    #write file
    output_file = "./predictions.json"
    with open(output_file, 'w') as file:
        file.write(json.dumps(predictions, indent=2))# use `json.loads` to do the reverse

def validation(loader, model, criterion):
    n_minibatches = len(loader)
    top1_acc = 0
    top5_acc = 0
    total = 0
    mean_loss = 0
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(loader, 0):
            images, labels = data
            #cuda
            images = images.cuda()
            labels = labels.cuda()

            outputs = model(images)
            #_, predicted = torch.max(outputs.data, dim=-1)
            _, top_1 = torch.topk(outputs.data, 1, dim=-1) #values, indices 
            _, top_5 = torch.topk(outputs.data, 5, dim=-1)
            total += labels.size(0)
            top1_acc += (top_1.squeeze(-1) == labels).sum().item()
            top5_acc += top_5.eq(labels.view(-1,1).expand_as(top_5)).sum().item()
            #mean loss
            loss = criterion(outputs, labels)
            mean_loss += loss.item()#(1/(i+1))*(loss.item() - mean_val_loss)
            if i % 200 == 0:    # print every 100 mini-batches
                logger.info(
                  "[mb %5d/%d] mean val loss: %.3f, top1 accuracy: %.3f, top5 accuracy: %.3f"%
                  (i + 1, n_minibatches, mean_loss/(i+1) , top1_acc / total, top5_acc/total))
    mean_loss = mean_loss / n_minibatches
    top1_acc = top1_acc / total
    top5_acc = top5_acc / total
    return mean_loss, top1_acc, top5_acc

def eval_model(cfg, models_path):
    #Setup
    model = ResNetLSTM(**cfg.model).cuda()
    optimizer = torch.optim.SGD(model.parameters(), **cfg.optim)
    epoch = load(models_path, model, optimizer, train=False)

    criterion = nn.CrossEntropyLoss()
    reshape_transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize((cfg.img_size, cfg.img_size)),
                                    #transforms.Grayscale(),
                                    transforms.ToTensor()])
    base_dir = "../../../datasets/20bn-sth-sth-v2"
    val_data = SthSthDataset(labels_file = "78-classes_validation.json",
                             transform = reshape_transform,
                             base_dir = base_dir,
                             n_frames = cfg.dataset.n_frames, #frames to pick from each video
                             str2id_file = cfg.dataset.str2id_file)

    data_loader = torch.utils.data.DataLoader(val_data, num_workers = 2, batch_size=8 , shuffle=False)
    mean_loss, top1acc, top5acc = validation(data_loader, model, criterion)

    #write file

@hydra.main(config_path="./config", config_name="config")
def main(cfg):
    test_model_dir = "../../../%s/"%cfg.test_dir
    test_cfg = OmegaConf.load(test_model_dir + ".hydra/config.yaml")
    model_name = "epoch_25.pth"
    models_path = os.path.abspath(os.path.join(test_model_dir + cfg.models_folder, model_name))
    #test_model(test_cfg, models_path=models_path)
    eval_model(test_cfg, models_path=models_path)
    
if __name__ == "__main__":
    main()