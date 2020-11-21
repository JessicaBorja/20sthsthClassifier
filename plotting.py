from matplotlib.pyplot import show
from utils.utils import map_new2orig, id2str, map_orig2new, get_class_dist
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import os

def plot_class_dist(labels_file="78-classes_train.json", fname = "hist", show_fig = False):
    unique_str, unique_new_ids, counts, classes = get_class_dist(labels_file)
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