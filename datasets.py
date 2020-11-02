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
# # Ignore warnings
# import warnings
# warnings.filterwarnings("ignore")

class SthSthDataset(Dataset):
    """20something-something dataset."""
    "id, label, template(Masked words), placeholders(words labels)"

    def __init__(self, labels_dir, data_dir, labels_file = "something-something-v2-train.json",skip_frames= 4, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the videos.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels_frame = pd.read_json( os.path.join(labels_dir, labels_file) )
        self.labels_id_frame = pd.read_json(\
                                os.path.join( labels_dir,"something-something-v2-labels.json"),\
                                typ='series')
        self.labels_dir = labels_dir
        self.data_dir = data_dir
        self.transform = transform
        self.skip_frames = skip_frames

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
            if(c % self.skip_frames == 0):
                video_frames.append(frame)
            c+=1
        cap.release()
        cv2.destroyAllWindows()
        video_frames = np.stack(video_frames,axis=0)
        
        #return labels
        label = curr_video["template"].replace("[", "").replace("]", "")
        label_id = self.labels_id_frame[label]
        
        return video_frames, label_id