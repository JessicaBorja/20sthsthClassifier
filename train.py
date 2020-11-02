import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from pretraining_backbone import ResNet18Backbone
import pandas as pd
import hydra
import logging
from omegaconf import DictConfig, OmegaConf
import os
import cv2
import numpy as np 
from datasets import SthSthDataset
from datetime import datetime

# @hydra.main(config_name="config")
# def main_hydra(cfg : DictConfig) -> None:
#     print(OmegaConf.to_yaml(cfg))
#     log = logging.getLogger(__name__)
#     log.info("Info level message")
#     log.debug("Debug level message")
#     model = models.resnet18()
#     model1 = ResNet18Backbone(pretrained = False)

def train(loader, model, criterion, optimizer):
    # model.train()
    mean_train_loss = 0.0
    mean_train_accuracy = 0.0
    # correct = 0
    # total = 0
    for i, data in enumerate(loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        #cuda
        inputs = inputs.cuda()
        labels = labels.cuda()

    #     # zero the parameter gradients
    #     optimizer.zero_grad()

    #     # forward + backward + optimize
    #     outputs = model(inputs)
    #     loss = criterion(outputs, labels)
    #     loss.backward()
    #     optimizer.step()

    #     #accuracy
    #     _, predicted = torch.max(outputs.data, 1)
    #     total += labels.size(0)
    #     correct += (predicted == labels).sum().item()
        
    #     # print statistics
    #     mean_train_loss += (1/(i+1))*(loss.item() - mean_train_loss)
    #     if i % 100 == 99:    # print every 100 mini-batches
    #         print('[i %5d] mean loss: %.3f' %
    #               (i + 1, mean_train_loss))
    # mean_train_accuracy = correct / total
    # print('mean train acc: %.3f' % (mean_train_accuracy))
    return mean_train_loss, mean_train_accuracy

def validate(val_loader, model, criterion):
    return val_loss, val_acc

def main():
    logger = logging.getLogger(__name__)
    base_name = "./datasets/20bn-something-something-v2"
    train_data = SthSthDataset(labels_dir = "%s_labels/"%base_name,
                               data_dir ="%s_data/"%base_name,
                               labels_file = "something-something-v2-train.json",
                               skip_frames = 4)
    val_data = SthSthDataset(labels_dir = "%s_labels/"%base_name,
                               data_dir ="%s_data/"%base_name,
                               labels_file = "something-something-v2-validation.json",
                               skip_frames = 4)
    train_loader = torch.utils.data.DataLoader(train_data,
                                                batch_size=4, shuffle=True,
                                                num_workers=4)
    val_loader = torch.utils.data.DataLoader(train_data,
                                                batch_size=4, shuffle=True,
                                                num_workers=4)

    # TODO: loss function
    model = models.resnet18()
    # model1 = ResNet18Backbone(pretrained = False)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9, weight_decay=1e-4) #cfg.lr

    logger.info('train_data {}'.format(train_data.__len__()))
    logger.info('val_data {}'.format(val_data.__len__()))

    best_val_loss = np.inf
    model_name = "20sth_sth_v2"
    model_name = "{}_{}".format(model_name, datetime.now().strftime('%d-%m_%I-%M'))
    writer_name = "./results/{}".format(model_name)
    #writer = SummaryWriter("")
    for epoch in range(100):
        print("Epoch {}".format(epoch))
        train_loss, train_accuracy = train(train_loader, model, criterion, optimizer)
        # val_loss, val_acc = validate(val_loader, model, criterion)
        
        # eval_dict = {"Loss/train" : train_loss, "Loss/validation": val_loss,
        #             "Accuracy/train" : train_accuracy, "Accuracy/validation": val_acc}

        # for key,value in eval_dict.items():
        #     writer.add_scalar(key, value, epoch)

        #save model
        #torch.save(model.state_dict(), os.path.join(args.model_folder, "epoch_"+str(epoch)+".pth" ))
        # if val_loss < best_val_loss:
        #     torch.save(model.state_dict(), os.path.join(args.model_folder, "epoch_"+str(epoch)+".pth" ))

if __name__ == "__main__":
    main()