import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import rnn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.resnet import resnet18, resnet50, resnet101, resnet152
import os

class ResNet18LSTM(nn.Module):
    def __init__(self, pretrained, n_classes, rnn_hidden = 256, num_layers = 4, \
                 backbone= "resnet18", save_dir = "./trained_models/"):
        super().__init__()
        self.save_dir = save_dir
        self.pretrained = pretrained
        self.backbone_net = IntermediateLayerGetter(self.get_backbone(backbone, pretrained=pretrained), {"avgpool": "out"}).cuda()
        self.fc1 = nn.Linear(512,512)
        self.lstm =  nn.LSTM(
                        input_size = 512,
                        hidden_size = rnn_hidden,
                        num_layers = num_layers,
                        batch_first =  True 
                    )
        # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        self.fc2 = nn.Linear(rnn_hidden, 128)
        self.fc3 = nn.Linear(128, n_classes, bias = True)
        #self.softmax = nn.Softmax(dim = -1)
        #nn.init.xavier_uniform_(self.fc1.weight)
        #nn.init.xavier_uniform_(self.fc2.weight)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def get_backbone(self, name, pretrained):
        if name == "resnet18":
            return resnet18(pretrained=pretrained)
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
                    x = self.backbone_net(x_in[:, t, :, :, :])['out']  # image t = (batch, channels, w, h)
            else:
                x = self.backbone_net(x_in[:, t, :, :, :])['out']  # image t = (batch, channels, w, h)
            x = x.view(x.size(0), -1) # (batch, channels * w * h)
            # FC layers
            x = F.relu(self.fc1(x))
            cnn_feat_seq.append(x)

        cnn_feat_seq = torch.stack(cnn_feat_seq, dim=0) # (n_frames, batch, cnn features)
        cnn_feat_seq = torch.transpose(cnn_feat_seq,0,1) # (batch, n_frames, cnn features)
        #batch_size, seq_len = cnn_feat_seq.shape[0:2]
        #x = torch.flatten(cnn_feat_seq, 1)
        #self.lstm.flatten_parameters()
        rnn_out, (h_n, h_c) = self.lstm(cnn_feat_seq, None)  # (batch, n_frames, rnn_hidden)
        del cnn_feat_seq
        # h_n shape = h_c shape = (n_layers, batch, hidden_size) 
        x = F.relu(self.fc2(rnn_out[:, -1, :]))# (batch,128) choose rnn_out at the last time step
        x = self.fc3(x) # (batch,1,n_classes)
        #x = F.dropout(x, p=0.2, training=self.training)
        return x

    def save(self, model_name = "model"):
        filename = os.path.join(self.save_dir, model_name+".pth")
        torch.save(self.state_dict(), filename )

    def load(self, filepath):
        self.load_state_dict(torch.load(filepath))
