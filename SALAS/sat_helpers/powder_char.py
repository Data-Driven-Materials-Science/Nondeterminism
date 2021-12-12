from imantics import Polygons, Mask
import pycocotools.mask as mask_util
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import json
## ampis
ampis_root = Path('../')
sys.path.append(str(ampis_root))

from ampis import data_utils
def clean_pred_instance(pred, instance_num):
    '''
    Currently does not work
    TODO: Determine how to delete a prediction mask from an instance object
    '''
    print(pred)
    data_instance = pred
    removed_index_list = []
    for i in range(len(data_instance['instances'].pred_masks)):
        #intermediary step to convert each RLE instance to a binary mask
        m1 = mask_util.decode(data_instance['instances'].pred_masks[i])[:, :]
        m2 = Mask(m1).polygons().points
        try:
            num_points = len(export_anno.split_array(m2[0])[0])
        except:
            print("Error: No points found, deleting annotation")
            num_points = 0
        if num_points < 4:
            removed_index_list.append(i)
    if len(removed_index_list) < 1:
        print('No Removals Required for instance', instance_num)
    else:
        print("Had to remove", len(removed_index_list), 'predicitons from the predicted detections on instance', instance_num)
    for i in range(len(removed_index_list)):
        data_instance['instances'].pred_masks.pop(removed_index_list[i]-i)
    return data_instance

        
def clean_pred_pickle_masks(pred):
    temp_pred = pred
    for i in range(len(pred)):
        temp_pred[i]['pred'] = clean_pred_instance(pred[i]['pred'], i)
    return temp_pred

    
def sort_and_clean(input_list):
    '''
    Takes in an unsorted list of 1 element lists, and returns a sorted list with up
    '''
    sort = []
    for i in input_list:
        sort.append(i)
    sort = np.sort(sort)
    Q1 = np.percentile(sort, 25, interpolation = 'midpoint') 
    Q2 = np.percentile(sort, 50, interpolation = 'midpoint') 
    Q3 = np.percentile(sort, 75, interpolation = 'midpoint') 
    IQR = Q3 - Q1
    low_lim = Q1 - 1.5 * IQR
    up_lim = Q3 + 1.5 * IQR
    index_cut_off = len(sort) - 1
    for i in range(len(sort)):
        if sort[i] > up_lim:
            index_cut_off = i
            break
    return sort[0:index_cut_off]

def generate_histogram(input_list, num_bins, title, x, y):
    '''
    Takes in a sorted list, number of bins, and strings for title, x and y axis
    Plots a histogram and displays it
    '''
    n, bins, patches = plt.hist(input_list, num_bins, facecolor='blue')
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()