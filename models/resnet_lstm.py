import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import rnn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
import os

class ResNetLSTM(nn.Module):
    def __init__(self, pretrained, n_classes, rnn_hidden = 256, num_layers = 4, \
                 fc1_hidden=512, fc2_hidden=512, fc3_hidden=256, dropout_rate = 0.2,
                 backbone= "resnet18", save_dir = "./trained_models/"):
        super().__init__()
        self.save_dir = save_dir
        self.pretrained = pretrained
        self.backbone_fixed = torch.nn.Sequential(*(list(self.get_backbone(backbone,pretrained).children())[:-5])).cuda()
        self.backbone_train = torch.nn.Sequential(*(list(self.get_backbone(backbone,pretrained).children())[-5:-1])).cuda()
        _output_len =  512 if backbone =="resnet18" or backbone =="resnet30" else 2048
        self.fc1 = nn.Linear(_output_len,512)
        self.lstm =  nn.LSTM(
                        input_size = 512,
                        hidden_size = rnn_hidden,
                        num_layers = num_layers,
                        batch_first =  True 
                    )
        # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        #self.fc2 = nn.Linear(rnn_hidden, 128)
        self.fc3 = nn.Linear(rnn_hidden, n_classes, bias = True)
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        #self.softmax = nn.Softmax(dim = -1)
        #nn.init.xavier_uniform_(self.fc1.weight)
        #nn.init.xavier_uniform_(self.fc2.weight)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def get_backbone(self, name, pretrained):
        if name == "resne34":
            return resnet34(pretrained=pretrained)
        elif name == "resnet50":
            return resnet50(pretrained=pretrained)
        elif name == "resnet101":
            return resnet101(pretrained=pretrained)
        elif name == "resnet152":
            return resnet152(pretrained=pretrained)
        else:
            return resnet18(pretrained=pretrained) #default

    def forward(self, x_in): #(batch, n_frames, channels, w, h)
        #for t in range(x_in.size(1)):
        batch_size, n_frames, channels, w, h = x_in.shape
        x = torch.reshape(x_in,(batch_size*n_frames, channels, w, h))
        # Pass each frame through resnet 
        if(self.pretrained):
            with torch.no_grad():
                x = self.backbone_fixed(x) # image t = (batch*frames, n_features,1,1)
        else:
            x = self.backbone_fixed(x) # image t = (batch*frames, n_features, 1, 1)
        x = self.backbone_train(x)
        x = self.dropout(x)
        x = x.view(batch_size, n_frames, -1) # (batch, channels * w * h)
        # FC layers
        x = F.relu(self.fc1(x))
        self.lstm.flatten_parameters()
        rnn_out, (h_n, h_c) = self.lstm( x, None)  # (batch, n_frames, rnn_hidden)
        # h_n shape = h_c shape = (n_layers, batch, hidden_size)
        
        x = self.fc3(rnn_out[:, -1, :])# (batch,128) choose rnn_out at the last time step
        #x = self.fc3(x) # (batch,1,n_classes)
        return x

    def save(self, model_name = "model"):
        filename = os.path.join(self.save_dir, model_name+".pth")
        torch.save(self.state_dict(), filename )

    def load(self, filepath):
        self.load_state_dict(torch.load(filepath))