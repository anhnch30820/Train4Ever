import os

import pandas as pd
import numpy as np
from tqdm import tqdm

from skimage import io, exposure
import tifffile as tif
from pycocotools import mask

import argparse
import imagecodecs
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES=True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path", type=str, help="input path"
    )
    parser.add_argument(
        "--output_path", type=str, help="out path"
    )

    args = parser.parse_args()
    return args


def read_image(img_path):
    if img_path.endswith('.tif') or img_path.endswith('.tiff'):
        img_data = tif.imread(img_path)
    else:
        img_data = io.imread(img_path)
    return img_data

def normalize_channel(img, lower=1, upper=99):
    non_zero_vals = img[np.nonzero(img)]
    percentiles = np.percentile(non_zero_vals, [lower, upper])
    if percentiles[1] - percentiles[0] > 0.001:
        img_norm = exposure.rescale_intensity(img, in_range=(percentiles[0], percentiles[1]), out_range='uint8')
    else:
        img_norm = img
    return img_norm.astype(np.uint8)


def process_image(img_data):
    # normalize image data
    if len(img_data.shape) == 2:
        img_data = np.repeat(np.expand_dims(img_data, axis=-1), 3, axis=-1)
    elif len(img_data.shape) == 3 and img_data.shape[-1] > 3:
        img_data = img_data[:,:, :3]
    else:
        pass
    pre_img_data = np.zeros(img_data.shape, dtype=np.uint8)
    for i in range(3):
        img_channel_i = img_data[:,:,i]
        if len(img_channel_i[np.nonzero(img_channel_i)])>0:
            pre_img_data[:,:,i] = normalize_channel(img_channel_i, lower=1, upper=99)
    return pre_img_data

def main(args):

    os.makedirs(args.output_path, exist_ok=True)

    for img_name in tqdm(os.listdir(args.input_path)):
        cell_id = img_name.split('.')[0]
        
        img_path = f'{args.input_path}/{img_name}'
        print(img_path)
        img_data = read_image(img_path)
        img = process_image(img_data)
        
        fname = f'{args.output_path}/{cell_id}.png'
    
        io.imsave(fname, img)



if __name__ == "__main__":
    args = parse_args()
    main(args)




