import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
import torchvision.transforms.functional as TF
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.transforms import Grayscale
from classifier import ResNet18LSTM
import hydra
import logging
from omegaconf import DictConfig, OmegaConf
import os, yaml
import numpy as np 
from datasets import SthSthDataset
import datetime, time
from utils.utils import load, save, resume_training

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
    return mean_train_loss, mean_train_accuracy

def validate(loader, model, criterion):
    n_minibatches = len(loader)
    correct = 0
    total = 0
    mean_val_loss = 0
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(loader, 0):
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
            if i % 200 == 0:    # print every 100 mini-batches
                print('[mb %5d/%d] mean loss: %.3f, mean accuracy: %.3f' %
                  (i + 1, n_minibatches, mean_val_loss, correct / total))
    mean_val_accuracy = correct / total
    return mean_val_loss, mean_val_accuracy

@hydra.main(config_name="config")
def main(cfg : DictConfig) -> None:
    print("Running configuration: ", cfg)
    logger = logging.getLogger(__name__)
    logger.info("Running configuration: %s", cfg)
    reshape_transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize((cfg.img_size, cfg.img_size)),
                                    #transforms.Grayscale(),
                                    transforms.ToTensor()])
    train_data = SthSthDataset(labels_file = cfg.train_filename, #"something-something-v2-train_new.json",
                               transform = reshape_transform,
                               **cfg.dataset)
    val_data = SthSthDataset(labels_file = cfg.validation_filename,
                             transform = reshape_transform,
                             **cfg.dataset)
    train_loader = torch.utils.data.DataLoader(train_data, **cfg.dataloader)
    val_loader = torch.utils.data.DataLoader(val_data, **cfg.dataloader)
    #n_classes = train_data.calc_n_classes()
    #Original number of classes: 174, new:78
    model = ResNet18LSTM(**cfg.model).cuda()
    #print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), **cfg.optim) #cfg.lr

    logger.info('train_data {}'.format(train_data.__len__()))
    logger.info('val_data {}'.format(val_data.__len__()))

    best_val_loss, best_train_loss = np.inf, np.inf
    best_val_acc, best_train_acc = -np.inf, -np.inf
    model_name = cfg.exp_name#
    model_name = "{}_{}".format(model_name, datetime.datetime.now().strftime('%d-%m_%I-%M'))
    
    #Tensorboard log
    writer_name = "./results/{}".format(model_name)
    writer = SummaryWriter(writer_name)
    
    #Training loop
    for epoch in range(cfg.n_epochs):
        start_time = time.time()
        print("Epoch {}".format(epoch))
        train_loss, train_acc = train(train_loader, model, criterion, optimizer)
        logger.info('[Epoch %d] mean train acc: %.3f' % (epoch, train_acc))

        val_loss, val_acc = validate(val_loader, model, criterion)
        logger.info('[Epoch %d] mean validation acc: %.3f' % (epoch, val_acc))

        results_dict = {"Loss/train" : train_loss, "Loss/validation": val_loss,
                    "Accuracy/train" : train_acc, "Accuracy/validation": val_acc}
        for key,value in results_dict.items():
            writer.add_scalar(key, value, epoch)

        #save(epoch, model, optimizer, cfg.models_folder ,"epoch_"+str(epoch)) #save all models
        best_train_loss, best_val_loss, best_train_acc, best_val_acc = \
            save_only_best(epoch, model, optimizer, cfg.models_folder, logger, \
                        train_loss, best_train_loss, val_loss, best_val_loss,\
                        train_acc, best_train_acc, val_acc, best_val_acc)

        end_time = time.time()
        seconds = end_time - start_time
        
        logger.info("Elapsed seconds:%0.3f, Time: %s"%(seconds, str(datetime.timedelta(seconds=seconds))))

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

if __name__ == "__main__":
    # hydra_folder = "./outputs/2020-11-13/01-17-46"
    # eval_model(hydra_folder, model_name="epoch_18.pth")
    main()
