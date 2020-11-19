from matplotlib.pyplot import show
from utils.utils import map_new2orig, id2str, map_orig2new
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import os

def get_data(labels_file = "78-classes_train.json", str2id_file = "78-classes_labels.json"):
    labels_dir = "./datasets/20bn-sth-sth-v2/labels/"
    data_info_path = os.path.abspath(os.path.join(labels_dir, labels_file))
    data_info_frame = pd.read_json(data_info_path)
    ids_frame = pd.read_json(os.path.join(labels_dir, str2id_file),\
                             typ='series')
    data_info_frame["template"] = data_info_frame["template"].str.replace("[", "")
    data_info_frame["template"] = data_info_frame["template"].str.replace("]", "")
    classes = ids_frame[data_info_frame["template"]]
    unique, counts = np.unique(classes, return_counts=True)
    old2new = map_orig2new()
    unique_str = ids_frame[unique]
    unique_new_ids = [ old2new[i] for i in unique]
    return unique_str, unique_new_ids, counts, classes

def plot_class_dist(labels_file="78-classes_train.json", fname = "hist", show_fig = False):
    unique_str, unique_new_ids, counts, classes = get_data(labels_file)
    plt.figure(figsize=(14,7))
    plt.title('Class distribution',fontsize=14)
    plt.xlabel('Classes', fontsize=14)
    plt.ylabel('Frequency',fontsize=14)
    #plt.hist(classes, bins = unique_new_ids, alpha=0.5, histtype='bar', ec='black')
    plt.bar(unique_new_ids, counts)
    plt.xticks(unique_new_ids, rotation='vertical') 
    plt.yticks() 
    if(show_fig):
        plt.show() 
    save_folder = "./figures/"
    if not os.path.exists(save_folder):
            os.makedirs(save_folder)
    plt.savefig(os.path.join(save_folder, fname+".png"), dpi=150)

if __name__ == "__main__":
    plot_class_dist("78-classes_train.json", fname="train_dist")
    plot_class_dist("78-classes_validation.json", fname ="eval_dist")