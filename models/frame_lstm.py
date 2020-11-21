import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
import os
from models import backbones
from models.backbones import DilatedResNetBackbone, Bottleneck

class FrameLSTM(nn.Module):

    def __init__(self, n_classes, rnn_hidden = 2048, num_layers = 4, pool_fn='L2', \
                 dropout_rate = 0.2, backbone= "resnet18", save_dir = "./trained_models/"):
        super().__init__()
        self.num_classes = n_classes
        self.hidden_size = rnn_hidden
        self.num_layers = num_layers
        self.pool_fn = pool_fn
        self.dropout_rate = dropout_rate
        self.pretrained = True
        self.init_backbone(backbone)
        print ('Using backbone class: %s'%backbone)
        self.save_dir = save_dir

    def get_backbone(self, name, pretrained):
        if name == "resne34":
            wts = resnet34(pretrained=pretrained).state_dict()
        elif name == "resnet50":
            wts = resnet50(pretrained=pretrained).state_dict()
        elif name == "resnet101":
            wts = resnet101(pretrained=pretrained).state_dict()
        elif name == "resnet152":
            wts = resnet152(pretrained=pretrained).state_dict()
        else:
            wts = resnet18(pretrained=pretrained).state_dict() #default
        #net = DilatedResNetBackbone(Bottleneck, [3, 4, 6, 3], strides=[1,2,2,2], dilations=[1,1,1,1], pretrained=wts)
        net = DilatedResNetBackbone(Bottleneck, [3, 4, 6, 3], strides=[1,2,1,1], dilations=[1,1,2,4], pretrained=wts)
        return net

    def init_backbone(self, backbone_name="resnet50"):
        self.backbone_net = self.get_backbone(backbone_name, pretrained=self.pretrained)
        #self.backbone_net = _backbone()
        self.spatial_dim = self.backbone_net.spatial_dim
        self.rnn = nn.LSTM(input_size = self.backbone_net.feat_dim, 
                            hidden_size = self.hidden_size, 
                            num_layers = self.num_layers,
                            batch_first=True) # (B, T, num_maps)
        self.fc = nn.Linear(self.hidden_size, self.num_classes)

        # LSTM hidden state
        n_layers = 1
        h0 = torch.zeros(n_layers, 1, self.hidden_size)
        c0 = torch.zeros(n_layers, 1, self.hidden_size)
        nn.init.xavier_normal_(h0, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(c0, gain=nn.init.calculate_gain('relu'))
        self.h0 = nn.Parameter(h0, requires_grad=True)
        self.c0 = nn.Parameter(c0, requires_grad=True)

        print ('FrameLSTM created with out_ch: %d and spatial dim: %d'%(self.hidden_size, self.spatial_dim))

    # (B, T, X) --> (B*T, X) --module--> (B*T, Y) --> (B, T, Y)
    def flatten_apply(self, tensor, module):
        shape = tensor.shape
        flat = (shape[0]*shape[1], ) + shape[2:]
        tensor = tensor.view(flat)
        out = module(tensor)
        uflat = (shape[0], shape[1], ) + out.shape[1:]
        out = out.view(uflat)
        return out

    def pool(self, frame_feats):
        if self.pool_fn=='L2':
            pool_feats = F.lp_pool2d(frame_feats, 2, self.spatial_dim)
        else: #self.pool_fn=='avg':
            pool_feats = F.avg_pool2d(frame_feats, self.spatial_dim)
        return pool_feats

    def forward(self, x_in):
        #(batch, n_frames, channels, w, h)
        batch_size, n_frames, channels, w, h = x_in.shape
        frame_feats = self.flatten_apply(x_in, lambda t: self.backbone_net(t)) # (B, T, 2048, 28, 28)
        pool_feats = self.flatten_apply(frame_feats, lambda t: self.pool(t)).view(batch_size, n_frames, -1) # (B, T, 2048)
        self.rnn.flatten_parameters()
        rnn_out, (h_n, h_c) = self.rnn(pool_feats)#, self.get_hidden_state(B, frame_feats.device))
        # clip_feats = h_n[-1] #same as rnn_out[:, -1, :] in this case
        # preds = self.fc(clip_feats)
        out = F.relu(self.fc(rnn_out[:, -1, :]))
        
        return out

    def save(self, model_name = "model"):
        filename = os.path.join(self.save_dir, model_name+".pth")
        torch.save(self.state_dict(), filename )

    def load(self, filepath):
        self.load_state_dict(torch.load(filepath))