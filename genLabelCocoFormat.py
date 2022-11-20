import os

import pandas as pd
import numpy as np
from tqdm import tqdm
import json,itertools

from skimage import io, measure
import tifffile as tif
from pycocotools import mask
import argparse
from sklearn.model_selection import KFold

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_images_path", type=str, help="input images path"
    )
    parser.add_argument(
        "--input_labels_path", type=str, help="input labels path"
    )
    parser.add_argument(
        "--output_folder_path", type=str, help="output folder path"
    )

    args = parser.parse_args()
    return args

def read_image(img_path):
    if img_path.endswith('.tif') or img_path.endswith('.tiff'):
        img_data = tif.imread(img_path)
    else:
        img_data = io.imread(img_path)
    return img_data


def coco_structure(train_df, label_folder='TrainLabels', start_image_id=0, start_ann_id=0):
    cat_ids = {'cell': 1}    
    cats =[{'name':name, 'id':id} for name,id in cat_ids.items()]

    global_image_id = start_image_id
    global_inst_id = start_ann_id

    # image_name_id_mapper = dict()

    images = []
    annotations=[]
    for idx, row in tqdm(train_df.iterrows(), total=len(train_df)):
        images.append({'id':global_image_id, 'width':row.width, 'height':row.height, 'file_name':f'{row.img_name}'})

        img_name = row.img_name
        cell_id = img_name.split('.')[0]
        label_path = f'{label_folder}/{cell_id}.tiff'
        # img = read_image(img_path)
        label = read_image(label_path)
        instance_ids = np.unique(label)

        for ins_id in instance_ids:
            if ins_id == 0: # background
                continue 
            bin_mask = np.where(label == ins_id, 1, 0).astype('uint8')
            # plt.figure()
            # plt.imshow(bin_mask)
            # plt.show()

            # ann_id = img_name + '_' + str(ins_id) # coco evaluator needs the annotation id to be int

            fortran_ground_truth_binary_mask = np.asfortranarray(bin_mask)
            encoded_ground_truth = mask.encode(fortran_ground_truth_binary_mask)
            ground_truth_area = mask.area(encoded_ground_truth)
            ground_truth_bounding_box = mask.toBbox(encoded_ground_truth)
            contours = measure.find_contours(bin_mask, 0.5)

            annotation = {
                    "segmentation": [],
                    "area": ground_truth_area.tolist(),
                    "iscrowd": 0,
                    "image_id": global_image_id,
                     "bbox": ground_truth_bounding_box.tolist(),
                    "category_id": 1,
                    "id": global_inst_id
                }

            for contour in contours:
                contour = np.flip(contour, axis=1)
                segmentation = contour.ravel().tolist()
                if len(segmentation) >= 6: # make sure it is polygon (3 points)
                    annotation["segmentation"].append(segmentation)

            annotations.append(annotation)

            # update global instance id
            global_inst_id += 1

        # update global image id
        global_image_id += 1

        # break
    
    
    return {'categories':cats, 'images':images,'annotations':annotations}

def main(args):
    # os.makedirs('train_vis', exist_ok=True)

    train_img_names = []
    train_widths = []
    train_heights = []
    train_num_cells = []

    for img_name in tqdm(os.listdir(args.input_labels_path)):
        cell_id = img_name.split('.')[0]
        # img_path = f'TrainImagesPNG/{img_name}'
        label_path = f'{args.input_labels_path}/{cell_id}.tiff'

        label = read_image(label_path)
        h, w = label.shape[:2]
        train_widths.append(w)
        train_heights.append(h)
        train_num_cells.append(np.max(label))
        train_img_names.append(img_name)

    train_val_meta = pd.DataFrame({'img_name':train_img_names,
                'width':train_widths,
                'height':train_heights,
                'cell_count':train_num_cells})

    kfold = KFold(n_splits=5, shuffle=True, random_state=67)

    fold = 0
    for train_indices, valid_indices in kfold.split(train_val_meta['img_name']):
        train_val_meta.loc[valid_indices, 'fold'] = fold
        fold += 1

    chosen_fold = 0
    train_df = train_val_meta[train_val_meta.fold != chosen_fold]
    val_df = train_val_meta[train_val_meta.fold == chosen_fold]


    train_annotations = coco_structure(train_df, label_folder=args.input_labels_path)
    val_annotations = coco_structure(val_df, label_folder=args.input_labels_path)

    with open(f'{args.output_folder_path}/coco_annotations/train_annotations_fold{chosen_fold}.json', 'w') as f:
        json.dump(train_annotations, f)

    with open(f'{args.output_folder_path}/coco_annotations/val_annotations_fold{chosen_fold}.json', 'w') as f:
        json.dump(val_annotations, f)

if __name__ == "__main__":
    args = parse_args()
    main(args)