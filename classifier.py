import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import rnn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
import os

class ResNet18LSTM(nn.Module):
    def __init__(self, pretrained, n_classes, rnn_hidden = 254, num_layers = 4, \
                 fc1_hidden=254, fc2_hidden=512, fc3_hidden=254, dropout_rate = 0.2,
                 backbone= "resnet18", save_dir = "./trained_models/"):
        super().__init__()
        self.save_dir = save_dir
        self.pretrained = pretrained
        self.backbone_net = torch.nn.Sequential(*(list(self.get_backbone(backbone,pretrained).children())[:-1])).cuda()
        _output_len =  512 if backbone =="resnet18" or backbone =="resnet30" else 2048
        self.fc1 = nn.Linear(_output_len,fc1_hidden)
        #self.bn1 = nn.BatchNorm1d(fc1_hidden, momentum=0.01)
        self.fc2 = nn.Linear(fc1_hidden, fc2_hidden)
        self.fc3 = nn.Linear(fc2_hidden, fc3_hidden)
        self.lstm =  nn.LSTM(
                        input_size = fc3_hidden,
                        hidden_size = rnn_hidden,
                        num_layers = num_layers,
                        batch_first =  True 
                    )
        # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        self.fc4 = nn.Linear(rnn_hidden, fc2_hidden)
        self.fc5 = nn.Linear(fc2_hidden, n_classes, bias = True)
        self.dropout_rate = dropout_rate
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
        cnn_feat_seq = []
        for t in range(x_in.size(1)):
            # Pass each frame through resnet 
            if(self.pretrained):
                with torch.no_grad():
                    x = self.backbone_net(x_in[:, t, :, :, :]) # image t = (batch, channels, w, h)
            else:
                x = self.backbone_net(x_in[:, t, :, :, :]) # image t = (batch, channels, w, h)
            x = x.view(x.size(0), -1) # (batch, channels * w * h)
            # FC layers
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            #x = F.dropout(x, p=0.2, training=self.training)
            x = F.relu(self.fc3(x))
            cnn_feat_seq.append(x)

        cnn_feat_seq = torch.stack(cnn_feat_seq, dim=0) # (n_frames, batch, cnn features)
        cnn_feat_seq = torch.transpose(cnn_feat_seq,0,1) # (batch, n_frames, cnn features)
        #batch_size, seq_len = cnn_feat_seq.shape[0:2]
        #x = torch.flatten(cnn_feat_seq, 1)
        #self.lstm.flatten_parameters()
        rnn_out, (h_n, h_c) = self.lstm(cnn_feat_seq, None)  # (batch, n_frames, rnn_hidden)
        del cnn_feat_seq
        # h_n shape = h_c shape = (n_layers, batch, hidden_size) 
        x = F.relu(self.fc4(rnn_out[:, -1, :]))# (batch,128) choose rnn_out at the last time step
        #x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.fc5(x)) # (batch,1,n_classes)
        return x

    def save(self, model_name = "model"):
        filename = os.path.join(self.save_dir, model_name+".pth")
        torch.save(self.state_dict(), filename )

    def load(self, filepath):
        self.load_state_dict(torch.load(filepath))