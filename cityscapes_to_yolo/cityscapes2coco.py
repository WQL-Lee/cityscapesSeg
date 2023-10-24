# This file heavily borrows from https://github.com/facebookresearch/Detectron/tree/master/tools

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys

# Image processing
# Check if PIL is actually Pillow as expected
try:
    from PIL import __version__
except:
    print("Please install the module 'Pillow' for image processing, e.g.")
    print("pip install pillow")
    sys.exit(-1)

try:
    import PIL.Image     as Image
    import PIL.ImageDraw as ImageDraw
except:
    print("Failed to import the image processing packages.")
    sys.exit(-1)

import argparse
import json
import os
import cv2
import numpy as np

from utils.instance_class import *
from utils.labels import *
import pycocotools.mask as maskUtils


import pycocotools.mask as maskUtils
def annToMask(ann,h,w):
    """
    Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
    :return: binary mask (numpy 2D array)
    """
    rle = annToRLE(ann,h,w)
    m = maskUtils.decode(rle)
    return m, rle
    
def annToRLE(ann, h, w):
    """
    Convert annotation which can be polygons, uncompressed RLE to RLE.
    :return: binary mask (numpy 2D array)
    """
    # h, w = t['height'], t['width']
    segm = ann['segmentation']
    if type(segm) == list:
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(segm, h, w)
        rle = maskUtils.merge(rles)
    elif type(segm['counts']) == list:
        # uncompressed RLE
        rle = maskUtils.frPyObjects(segm, h, w)
    else:
        # rle
        rle = ann['segmentation']
    return rle

def poly_to_box(poly):
    """Convert a polygon into a tight bounding box."""
    x0 = min(min(p[::2]) for p in poly)
    x1 = max(max(p[::2]) for p in poly)
    y0 = min(min(p[1::2]) for p in poly)
    y1 = max(max(p[1::2]) for p in poly)
    box_from_poly = [x0, y0, x1, y1]
    return box_from_poly

def xyxy_to_xywh(xyxy_box):
    xmin, ymin, xmax, ymax = xyxy_box
    TO_REMOVE = 1
    xywh_box = (xmin, ymin, xmax - xmin + TO_REMOVE, ymax - ymin + TO_REMOVE)
    return xywh_box


full_categories={
        'unlabeled',
        'ego vehicle',
        'rectification border',
        'out of roi',
        'road',
        'sidewalk',
        'parking',
        'rail track',
        'person',
        'rider',
        'car',
        'truck',
        'bus',
        'on rails',
        'motorcycle',
        'bicycle',
        'caravan',
        'trailer',
        'building',
        'wall',
        'fence',
        'guard rail',
        'bridge',
        'tunnel',
        'pole',
        'pole group',
        'traffic sign',
        'traffic light',
        'vegetation',
        'terrain',
        'sky',
        'ground',
        'dynamic',
        'static',
}

def flatten_concatenation(matrix):
    flat_list = []
    for row in matrix:
        flat_list += row
    return flat_list

def convert_cityscapes(data_dir, out_dir):
    """Convert from cityscapes format to COCO instance seg format - polygons"""
    sets = [
        'leftImg8bit/train',
        'leftImg8bit/val'
    ]

    ann_dirs = [
        'gtFine/train',
        'gtFine/val',
    ]

    json_name = 'instancesonly_filtered_%s.json'
    polygon_json_file_ending = '_polygons.json'
    img_id = 0
    ann_id = 0


    # category_dict= {
    #     'road':1,
    #     'sidewalk':2,
    #     'parking':3,
    #     'rail track':4,
    #     'person':5,
    #     'rider':6,
    #     'car':7,
    #     'truck':8,
    #     'bus':9,
    #     'on rails':10,
    #     'motorcycle':11,
    #     'bicycle':12,
    #     'caravan':13,
    #     'trailer':14,
    #     'building':15,
    #     'wall':16,
    #     'fence':17,
    #     'guard rail':18,
    #     'bridge':19,
    #     'tunnel':20,
    #     'pole':21,
    #     'pole group':22,
    #     'traffic sign':23,
    #     'traffic light':24,
    #     'vegetation':25,
    #     'terrain':26,
    #     'sky':27,
    #     'ground':28,
    #     'dynamic':29,
    #     'static':30,
    # }
    
    category_dict= {
        'person':1,
        'rider':2,
        'car':3,
        'truck':4,
        'bus':5,
        'motorcycle':6,
        'bicycle':7,
        'caravan':8,
    }
    
    
    

    for data_set, ann_dir in zip(sets, ann_dirs):
        print('Starting %s' % data_set)
        ann_dict = {}
        images = []
        annotations = []

        for root, _, files in os.walk(os.path.join(data_dir, ann_dir)):
            for filename in files:
                if filename.endswith(polygon_json_file_ending):

                    if len(images) % 50 == 0:
                        print("Processed %s images, %s annotations, %s categories" % (len(images), len(annotations),len(category_dict)))
                        # print(list(sorted(category_dict)))

                    json_ann = json.load(open(os.path.join(root, filename)))

                    image = {}
                    image['id'] = img_id
                    img_id += 1
                    image['width'] = json_ann['imgWidth']
                    image['height'] = json_ann['imgHeight']
                    image['file_name'] = os.path.join("leftImg8bit",
                                                      data_set.split("/")[-1],
                                                      filename.split('_')[0],
                                                      filename.replace("_gtFine_polygons.json", '_leftImg8bit.png'))
                    image['seg_file_name'] = filename.replace("_polygons.json", "_instanceIds.png")
                    images.append(image)
                    
                    for obj in json_ann["objects"]:
                        obj_cls = obj["label"]
                        if obj_cls not in category_dict.keys():
                            continue
                        
                        ann = {}
                        ann['id'] = ann_id
                        ann_id += 1
                        ann['image_id'] = image['id']
                        ann['segmentation'] = [flatten_concatenation(obj["polygon"])]

                        ann['category_id'] = category_dict[obj_cls]
                        ann['iscrowd'] = 0
                        
                        _, rle = annToMask(ann,image['height'] ,image['width'])
                        ann['area']=float(maskUtils.area(rle))
                        # ann['area'] = obj['pixelCount']

                        xyxy_box = poly_to_box(ann['segmentation'])
                        xywh_box = xyxy_to_xywh(xyxy_box)
                        ann['bbox'] = xywh_box

                        annotations.append(ann)


        ann_dict['images'] = images
        categories = [{"id": category_dict[name], "name": name} for name in category_dict]
        ann_dict['categories'] = categories
        ann_dict['annotations'] = annotations
        print("Num categories: %s" % len(categories))
        print("Num images: %s" % len(images))
        print("Num annotations: %s" % len(annotations))
        if not os.path.exists(os.path.abspath(out_dir)):
            os.mkdir(os.path.abspath(out_dir))
        with open(os.path.join(out_dir, json_name % ann_dir.replace("/", "_")), 'w') as outfile:
            outfile.write(json.dumps(ann_dict))

def parse_args():
    parser = argparse.ArgumentParser(description='Convert dataset')
    parser.add_argument('--dataset', help="../dataset/cityscapes", default='cityscapes', type=str)
    parser.add_argument('--outdir', help="output dir for json files", default='/home/lwq/Code/Other/YoloSeg/dataset/cityscapes/annotations', type=str)
    parser.add_argument('--datadir', help="data dir for annotations to be converted", default="/home/lwq/Code/Other/YoloSeg/dataset/cityscapes", type=str)
    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.dataset == "cityscapes":
        convert_cityscapes(args.datadir, args.outdir)
    else:
        print("Dataset not supported: %s" % args.dataset)
