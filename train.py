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
        if i % 1000 == 0:    # print every 100 mini-batches
            print('[i %5d] mean loss: %.3f' %
                  (i + 1, mean_train_loss))
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

# @hydra.main(config_name="config")
# def main_hydra(cfg : DictConfig) -> None:
def main():
    logger = logging.getLogger(__name__)
    models_folder = "./trained_models/"
    base_name = "./datasets/20bn-something-something-v2"
    reshape_transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize((64, 64)),
                                    #transforms.Grayscale(),
                                    transforms.ToTensor()])
    train_data = SthSthDataset(labels_dir = "%s_labels/"%base_name,
                               data_dir ="%s_data/"%base_name,
                               labels_file = "something-something-v2-train_new.json",
                               n_frames = 8,
                               transform = reshape_transform)
    val_data = SthSthDataset(labels_dir = "%s_labels/"%base_name,
                               data_dir ="%s_data/"%base_name,
                               labels_file = "something-something-v2-validation_new.json",
                               n_frames = 8,
                               transform = reshape_transform)
    train_loader = torch.utils.data.DataLoader(train_data,
                                                batch_size = 4, shuffle=True,
                                                num_workers = 2)
    val_loader = torch.utils.data.DataLoader(val_data,
                                                batch_size = 4, shuffle=True,
                                                num_workers = 2)
    #n_classes = train_data.calc_n_classes()
    #val_n_classes = val_data.calc_n_classes()
    #Original number of classes: 174, new:78
    model = ResNet18LSTM(pretrained=False, n_classes = 78).cuda()
    #print(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9, weight_decay=1e-4) #cfg.lr

    logger.info('train_data {}'.format(train_data.__len__()))
    logger.info('val_data {}'.format(val_data.__len__()))
    print('train_data {}'.format(train_data.__len__()))
    print('val_data {}'.format(val_data.__len__()))

    best_val_loss = np.inf
    model_name = "20sth_sth_v2"
    model_name = "{}_{}".format(model_name, datetime.datetime.now().strftime('%d-%m_%I-%M'))
    writer_name = "./results/{}".format(model_name)
    writer = SummaryWriter(writer_name)
    for epoch in range(30):
        start_time = time.time()
        print("Epoch {}".format(epoch))
        train_loss, train_accuracy = train(train_loader, model, criterion, optimizer)
        val_loss, val_acc = validate(val_loader, model, criterion)
        results_dict = {"Loss/train" : train_loss, "Loss/validation": val_loss,
                    "Accuracy/train" : train_accuracy, "Accuracy/validation": val_acc}
        # results_dict = {
        #     "train/Loss" : train_loss, "train/accuracy": val_loss,
        #     "validation/Loss": val_loss, "validation/accuracy":val_acc
        # }
        for key,value in results_dict.items():
            writer.add_scalar(key, value, epoch)

        #save all models
        torch.save(model.state_dict(), os.path.join(models_folder, "epoch_"+str(epoch)+".pth" ))
        # if val_loss < best_val_loss:
        #     torch.save(model.state_dict(), os.path.join(model_folder, "epoch_"+str(epoch)+".pth" ))
        end_time = time.time()
        seconds = end_time - start_time
        print("Elapsed seconds:%0.3f, Time: %s"%(seconds, str(datetime.timedelta(seconds=seconds))))

if __name__ == "__main__":
    #test_load()
    main()
