import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
import torchvision.transforms.functional as TF
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.transforms import Grayscale
from classifier import ResNet18LSTM
import pandas as pd
import hydra
import logging
from omegaconf import DictConfig, OmegaConf
import os
import cv2
import numpy as np 
from datasets import SthSthDataset
import datetime
import time
# @hydra.main(config_name="config")
# def main_hydra(cfg : DictConfig) -> None:
#     print(OmegaConf.to_yaml(cfg))
#     log = logging.getLogger(__name__)
#     log.info("Info level message")
#     log.debug("Debug level message")
#     model = models.resnet18()
#     model1 = ResNet18Backbone(pretrained = False)

def train(loader, model, criterion, optimizer):
    n_minibatches = len(loader)
    model.train()
    mean_train_loss = 0.0
    mean_train_accuracy = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        #cuda
        inputs = inputs.cuda()
        labels = labels.cuda()
        output = model(inputs)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        #accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # print statistics
        mean_train_loss += (1/(i+1))*(loss.item() - mean_train_loss)
        if i % 200 == 0:    # print every 100 mini-batches
            print('[mb %5d/%d] mean loss: %.3f, mean accuracy: %.3f' %
                  (i + 1, n_minibatches, mean_train_loss, correct / total))
    mean_train_accuracy = correct / total
    print('mean train acc: %.3f' % (mean_train_accuracy))
    return mean_train_loss, mean_train_accuracy

def validate(val_loader, model, criterion):
    print("validation...")
    correct = 0
    total = 0
    mean_val_loss = 0
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            images, labels = data
            #cuda
            images = images.cuda()
            labels = labels.cuda()

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            #mean loss
            loss = criterion(outputs, labels)
            mean_val_loss += (1/(i+1))*(loss.item() - mean_val_loss)

    mean_val_accuracy = correct / total
    print('Accuracy of the network on the val images: %.3f' % (
        mean_val_accuracy))
    return mean_val_loss, mean_val_accuracy

@hydra.main(config_name="config")
def main(cfg : DictConfig) -> None:
#def main():
    logger = logging.getLogger(__name__)
    models_folder = cfg.models_folder #"./trained_models/"
    base_dir = cfg.base_dir
    reshape_transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize((64, 64)),
                                    #transforms.Grayscale(),
                                    transforms.ToTensor()])
    train_data = SthSthDataset(labels_dir = "%s/labels/"%base_dir,
                               data_dir ="%s/data/"%base_dir,
                               labels_file = cfg.train_filename, #"something-something-v2-train_new.json",
                               str2id_file = cfg.str2id_file,
                               n_frames = cfg.n_frames,
                               transform = reshape_transform)
    val_data = SthSthDataset(labels_dir = "%s/labels/"%base_dir,
                               data_dir ="%s/data/"%base_dir,
                               labels_file = cfg.validation_filename,
                               str2id_file = cfg.str2id_file,
                               n_frames = cfg.n_frames,
                               transform = reshape_transform)
    train_loader = torch.utils.data.DataLoader(train_data,
                                                batch_size = cfg.batch_size, shuffle=True,
                                                num_workers = cfg.num_workers)
    val_loader = torch.utils.data.DataLoader(val_data,
                                                batch_size = cfg.batch_size, shuffle=True,
                                                num_workers = cfg.num_workers)
    #n_classes = train_data.calc_n_classes()
    #Original number of classes: 174, new:78
    model = ResNet18LSTM(pretrained=cfg.pretrained, n_classes = 78, save_dir="./trained_models").cuda()
    #print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay) #cfg.lr

    logger.info('train_data {}'.format(train_data.__len__()))
    logger.info('val_data {}'.format(val_data.__len__()))
    #print('train_data {}'.format(train_data.__len__()))
    #print('val_data {}'.format(val_data.__len__()))

    # best_val_loss, best_train_loss = np.inf, np.inf
    # best_val_acc, best_train_acc = -np.inf, -np.inf
    model_name = cfg.exp_name#
    model_name = "{}_{}".format(model_name, datetime.datetime.now().strftime('%d-%m_%I-%M'))
    
    #Tensorboard log
    writer_name = "./results/{}".format(model_name)
    writer = SummaryWriter(writer_name)
    
    #Training loop
    for epoch in range(cfg.n_epochs):
        start_time = time.time()
        print("Epoch {}".format(epoch))
        train_loss, train_accuracy = train(train_loader, model, criterion, optimizer)
        val_loss, val_acc = validate(val_loader, model, criterion)
        results_dict = {"Loss/train" : train_loss, "Loss/validation": val_loss,
                    "Accuracy/train" : train_accuracy, "Accuracy/validation": val_acc}
        for key,value in results_dict.items():
            writer.add_scalar(key, value, epoch)

        #save all models
        model.save("epoch_"+str(epoch)+".pth")
        end_time = time.time()
        seconds = end_time - start_time
        print("Elapsed seconds:%0.3f, Time: %s"%(seconds, str(datetime.timedelta(seconds=seconds))))

def save(epoch, model, optim, folder, name):
    save_dict = {"model_state_dict": model.state_dict(),
                "optimizer_state_dict": optim.sate_dict(),
                "epoch": epoch}
    filename = os.path.join(folder, name+".pth")
    torch.save(save_dict, filename)

def load(path, model, optimizer, train=True):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']

    if(train):
        model.train()
    else:
        model.eval()
    
    return epoch

if __name__ == "__main__":
    #test_load()
    main()
