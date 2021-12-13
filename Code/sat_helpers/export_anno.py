from imantics import Polygons, Mask
import pycocotools.mask as mask_util
from pathlib import Path
import sys
import os
import json
import cv2
## ampis
ampis_root = Path('../')
sys.path.append(str(ampis_root))

from ampis import data_utils

#This is purely a test of method usage.
def have_fun():
    print("Having Fun!")

#This is used to take detectron2 Tensor output, and convert them into polygon annotation
#Takes image path and output of detectron2 predictor
def export_tensor_to_polygon(img_path, pred):
    p = pred
    #intermediary step to convert tensor to RLE
    data_utils.format_outputs(img_path, dataset='test', pred=p)
    l = []
    for i in pred['instances'].pred_masks:
        #intermediary step to convert each RLE instance to a binary mask
        m1 = mask_util.decode(i)[:, :]
        #converts previous binary mask to polygon and appends to list with already converted segments
        l.append(Mask(m1).polygons().points)
    return l


#Takes an array of points in the form of [[x0,y0],[x1,y1]] and instead transitions into [[x1,x2],[y1,y2]]
def split_array(arr):
    return_list = [[], []]
    for i in arr:
        return_list[0].append(int(i[0]))
        return_list[1].append(int(i[1]))
    return return_list


#Takes a singluary array of points in format [[x1,x2],[y1,y2]] and converts into the format VIA accepts as instance notation
def make_individual_poly_annotation(array):
    inner = {"name":"polygon","all_points_x":array[0],"all_points_y":array[1]}
    outer = {"shape_attributes":inner}
    return outer


#Takes image filename, image path, and detectron2 predictions on image and returns a dictionary in VIA format
#This dictionary can then be imported as a json to VIA as if they were user made annotations
def make_VIA_file(image_filename, img_path, pred):
    image_size = os.path.getsize(img_path)
    outer = {}
    name = image_filename + str(image_size)
    ins = {'filename': image_filename, 'size': image_size}
    poly = export_tensor_to_polygon(img_path, pred)
    points = []
    for i in poly:
        points.append(split_array(i[0]))
    annos = []
    for i in points:
        temp = make_individual_poly_annotation(i)
        if len(temp['shape_attributes']['all_points_x']) > 5:
            annos.append(temp)
    ins['regions'] = annos
    outer = {name:ins}
    return outer


#Takes a filepath and a dictionary and writes the dictionary to disk in JSON format
def save_to_json(filepath, d):
    with open(filepath, 'w') as fp:
        json.dump(d, fp)

#Takes a list of a range of numbers and the total range those numbers can lie upon
#Switches those numbers to the inverse of it within that range
def invert_list(input_list, list_range):
    output_list = []
    for i in input_list:
        output_list.append(i)
    for i in range(len(output_list)):
        output_list[i] = list_range - output_list[i]
    return output_list


def invert_shape(input_dict, img_width, img_height, horizontal, vertical):
    if horizontal: 
        input_dict['shape_attributes']['all_points_x'] = invert_list(input_dict['shape_attributes']['all_points_x'], img_width)
    if vertical: 
        input_dict['shape_attributes']['all_points_y'] = invert_list(input_dict['shape_attributes']['all_points_y'], img_height)
    return input_dict


def invert_x_y_regions(input_list, img_width, img_height, horizontal, vertical):
    output_list = []
    for i in input_list:
        output_list.append(invert_shape(i, img_width, img_height, horizontal, vertical))
    return output_list


#(HARDCODED) TODO: Take in pathways
def invert_x_y_via(input_dict, horizontal, vertical, width, height):
    inverted_dict = {}
    modifier = ''
    if horizontal:
        modifier += 'x'
    if vertical:
        modifier += 'y'
    print(modifier)
    for i in input_dict:
        split_name = i.split('.png')
        flip_save_image(split_name[0], horizontal, vertical)
        size = os.path.getsize('Auto_annotate_images/' + split_name[0] + modifier + '.png')
        input_dict[i]['filename'] = split_name[0] + modifier + '.png'
        input_dict[i]['size'] = size
        input_dict[i]['regions'] = invert_x_y_regions(input_dict[i]['regions'], width, height, horizontal, vertical)
        inverted_dict[split_name[0]+modifier+'.png'+str(size)] = input_dict[i]
    return inverted_dict


#(HARDCODED) TODO: Take in pathways
def flip_save_image(name, horizontally, vertically, save=True):
    new_name = name
    img_path = Path('Auto_annotate_images', name +'.png')
    img = cv2.imread(str(img_path))
    if horizontally:
        new_name += 'x'
        img = cv2.flip(img, 1)
    if vertically:
        new_name += 'y'
        img = cv2.flip(img, 0)
    new_img_path = Path('Auto_annotate_images', new_name +'.png')
    if save:
        cv2.imwrite(str(new_img_path), img)
    return new_name
