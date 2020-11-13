from __future__ import print_function, division
import os
from pandas.core import base
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2
import os
# # Ignore warnings
# import warnings
# warnings.filterwarnings("ignore")

class SthSthDataset(Dataset):
    """20something-something dataset."""
    "id, label, template(Masked words), placeholders(words labels)"

    def __init__(self, base_dir, labels_file = "something-something-v2-train.json",\
                    str2id_file = "something-something-v2-labels.json", n_frames= 10, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the videos.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels_dir = "%s/labels/"%base_dir
        self.data_dir = "%s/data/"%base_dir
        self.labels_frame = pd.read_json(os.path.join(self.labels_dir, labels_file))
        self.labels_id_frame = pd.read_json(\
                                os.path.join(self.labels_dir, str2id_file),\
                                typ='series')
        self.transform = transform
        self.n_frames = n_frames if n_frames<=20 else 20
        self.transform = transform
        self.new_ids = self.map_large2small()

    def calc_n_classes(self):
        return len(self.labels_frame["template"].unique())

    def map_large2small(self):
        keep_ids = [5,6] + list(range(8,25)) + [27,29] +\
            list(range(40,48)) + [49] +  list(range(53,59)) +\
            [60,62,69,83] + list(range(85,90)) + list(range(93,97))+\
            list(range(98,102)) + list(range(104,111))+\
            [118,121,122,123,129,130,148,151,152] +\
            list(range(155,161)) + [164,170,171,172,173]
        keep_ids.remove(159)
        keep_ids.remove(108)
        new_ids_lst = list(range(len(keep_ids)))
        old2new = dict(zip(keep_ids, new_ids_lst))
        return old2new

    def __len__(self):
        return len(self.labels_frame)

    def __getitem__(self, idx):
        curr_video = self.labels_frame.iloc[idx]
        if torch.is_tensor(idx):
            idx = idx.tolist()
        video_name = os.path.join(self.data_dir,
                                  str(curr_video["id"])) + ".webm"

        cap = cv2.VideoCapture(video_name)
        video_frames = []
        c = 0
        while(cap.isOpened() and c<50): #to avoid loop for ever
            ret, frame =  cap.read()
            if ret == False:
                break
            video_frames.append(frame)
            c+=1
        cap.release()
        cv2.destroyAllWindows()
        selected_indices = np.round(np.linspace(0, len(video_frames) - 1, self.n_frames)).astype(int)
        
        if self.transform:
            video_frames = [self.transform(video_frames[i]) for i in selected_indices]
        else:
            video_frames = [video_frames[i] for i in selected_indices]
        video_frames = np.stack(video_frames,axis=0)
        
        #return labels
        label = curr_video["template"].replace("[", "").replace("]", "")
        label_id = self.labels_id_frame[label]
        label_id = self.new_ids[label_id]
        return video_frames, label_id

class SthSthTestset(Dataset):
    """20something-something dataset."""
    "id, label, template(Masked words), placeholders(words labels)"

    def __init__(self, base_dir, ids_file = "something-something-v2-test.json",\
                 n_frames= 10, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the videos.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels_dir = "%s/labels/"%base_dir
        self.data_dir = "%s/data/"%base_dir
        self.video_ids_frame = pd.read_json(os.path.join(self.labels_dir, ids_file))
        self.transform = transform
        self.n_frames = n_frames if n_frames<=20 else 20
        self.transform = transform

    def __len__(self):
        return len(self.video_ids_frame)

    def __getitem__(self, idx):
        curr_video = self.video_ids_frame.iloc[idx]
        if torch.is_tensor(idx):
            idx = idx.tolist()
        video_name = os.path.join(self.data_dir,
                                  str(curr_video["id"])) + ".webm"

        cap = cv2.VideoCapture(video_name)
        video_frames = []
        c = 0
        while(cap.isOpened() and c<50): #to avoid loop for ever
            ret, frame =  cap.read()
            if ret == False:
                break
            video_frames.append(frame)
            c+=1
        cap.release()
        cv2.destroyAllWindows()
        selected_indices = np.round(np.linspace(0, len(video_frames) - 1, self.n_frames)).astype(int)
        
        if self.transform:
            video_frames = [self.transform(video_frames[i]) for i in selected_indices]
        else:
            video_frames = [video_frames[i] for i in selected_indices]
        video_frames = np.stack(video_frames,axis=0)
        
        return video_frames, video_name