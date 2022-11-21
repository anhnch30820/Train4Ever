import argparse
import os
import tifffile as tif
from skimage import io, exposure
import numpy as np
from tqdm.auto import tqdm
import cv2
from mmdet.apis import init_detector, inference_detector


MIN_SIDE_FOR_SLIDING = 4000
MIN_REQUIRED_INST_NUM = 5


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path", type=str, help="input path"
    )
    parser.add_argument(
        "--config_path", type=str, help="input path"
    )
    parser.add_argument(
        "--ckpt_path", type=str, help="input path"
    )

    parser.add_argument(
        "--output_path", type=str, help="out path"
    )

    args = parser.parse_args()
    return args


def get_patch_size(size):
    if size >= 2000 and size < 3000:
        return 256
    if size < 4000:
        return 512
    if size < 15000:
        return 1024
    if size >= 15000:
        return 2048
    return 1024


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


def sliding_window_prediction(im, model, window_size = 1024):
    H, W = im.shape[:2]
    n_rows = int(np.ceil(H / window_size))
    n_cols = int(np.ceil(W / window_size))

    pred_instance_mask = np.zeros((im.shape[0], im.shape[1]), dtype=np.int32)

    inst_id = 1
    for i in tqdm(range(n_cols)):
        for j in range(n_rows):
            start_x, end_x = window_size*i, np.minimum(window_size*(i+1), W)
            start_y, end_y = window_size*j, np.minimum(window_size*(j+1), H)
            patch = im[start_y:end_y, start_x:end_x]
            outputs = inference_detector(model, patch)
            for num, mask in enumerate(outputs[1][0]):              
              ys, xs = np.where(mask==1)
              ys += start_y
              xs += start_x
              pred_instance_mask[ys, xs] = inst_id
              inst_id += 1
    return pred_instance_mask

def main(args):
    os.makedirs(args.output_path, exist_ok=True)

    config_path = args.config_path
    ckpt_path = args.ckpt_path

    model = init_detector(config_path, ckpt_path, device='cuda')

    for fname in tqdm(sorted(os.listdir(args.input_path))):
        
        img_path = os.path.join(args.input_path, fname)

        im = read_image(img_path)
        im = process_image(im)
        
        # # convert to RGB
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        h, w = im.shape[:2]
        shortest_edge = np.min(im.shape[:2])
        outputs = inference_detector(model, im)
        
        
        if len(outputs[1][0]) <= MIN_REQUIRED_INST_NUM or shortest_edge > MIN_SIDE_FOR_SLIDING:
            patch_size = get_patch_size(shortest_edge)
            print('Image', fname, 'has predicted inst num =', len(outputs[1][0]), 
                'And size =', shortest_edge,
                '. Use sliding window infer with patch size:', patch_size)
            pred_instance_mask, result = sliding_window_prediction(im, model, patch_size)
            # np.savetxt(os.path.join(TUNING_SET_OUT_FOLDER, fname.split('.')[0] +'_label.txt'), result)
        else:
            pred_instance_mask = np.zeros((im.shape[0], im.shape[1]), dtype=np.int32)
            for i, mask in enumerate(outputs[1][0]):
                inst_id = i+1
                pred_instance_mask[mask] = inst_id
            # np.savetxt(os.path.join(TUNING_SET_OUT_FOLDER, fname.split('.')[0] +'_label.txt'), outputs[0][0])
        
        if not len(np.unique(pred_instance_mask)) > 5:
            print(fname)

        output_path = os.path.join(args.output_path, fname.split('.')[0] +'_label.tiff')
        tif.imwrite(output_path, pred_instance_mask, compression='zlib')


if __name__ == "__main__":
    args = parse_args()
    main(args)