import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision import transforms
from classifier import ResNet18LSTM
from omegaconf import DictConfig, OmegaConf
import os, yaml
import numpy as np 
import pandas as pd
from datasets import SthSthTestset
from utils.utils import load
import json

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

def id2str(id, small2largeDict, id2strDict):
    full_classes_id = small2largeDict[id]
    str_class = id2strDict[full_classes_id]
    return str_class
    
#no labels
def make_predictions(loader, model):
    base_dir = "./datasets/20bn-sth-sth-v2"
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

def test_model(hydra_folder, model_name, output_dir = "./"):
    #Setup
    cfg  = yaml.load(open( "%s/.hydra/config.yaml"%hydra_folder, 'r'))
    cfg["model"]["save_dir"] = cfg["models_folder"]
    model = ResNet18LSTM(**cfg["model"]).cuda()
    optimizer = torch.optim.SGD(model.parameters(), **cfg["optim"]) #cfg.lr
    models_path = "%s/trained_models/%s"%(hydra_folder, model_name)
    epoch = load(models_path, model, optimizer, train=False)
    criterion = nn.CrossEntropyLoss()
    reshape_transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize((64, 64)),
                                    #transforms.Grayscale(),
                                    transforms.ToTensor()])
    data = SthSthTestset(ids_file = "something-something-v2-test.json",
                         base_dir = "./datasets/20bn-sth-sth-v2",
                         transform = reshape_transform,
                         n_frames = 8, #frames to pick from each video
                        )
    data_loader = torch.utils.data.DataLoader(data, num_workers = 2, batch_size=1 ,shuffle=False)
    predictions = make_predictions(data_loader, model)
    #write file
    output_file = "%s/predictions.json"%output_dir
    with open(output_file, 'w') as file:
        file.write(json.dumps(predictions, indent=2))# use `json.loads` to do the reverse

if __name__ == "__main__":
    hydra_folder = "./outputs/2020-11-13/01-17-46"
    model_name = "epoch_18.pth"
    test_model(hydra_folder, model_name=model_name, output_dir = "./")
