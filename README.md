# Set up environment

```
git clone https://github.com/anhnch30820/Train4Ever.git
```

```
pip install -U openmim
mim install mmcv-full
```

```
cd Train4Ever
pip install -v -e .
```

Change permission to train

```
chmod 777 ./tools/dist_train.sh
```

# Preprocessing and Gen label format COCO

## Preprocessing
```
python preprocessing.py --input_path <path_to_input_data> --output_path ./data/TrainImagesPNG
```

## Gen label fomart COCO
```
python genLabelCocoFormat.py --input_labels_path <path_to_input_label_images> --output_folder_path ./data
```
To save time you can download our preprocessed data and generated annotation file [here](https://drive.google.com/drive/folders/1fZM3zDHC06K8kI9AJZNA3hJaLaaX36TH?usp=sharing)


# Training
```
tools/dist_train.sh configs/cbnet/mask_rcnn_cbv2_swin_tiny_patch4_window7_mstrain_780-1100_adamw_3x_coco.py 1
```

You can change file config [here](https://github.com/anhnch30820/Train4Ever/blob/master/configs/cbnet/mask_rcnn_cbv2_swin_tiny_patch4_window7_mstrain_780-1100_adamw_3x_coco.py) 

