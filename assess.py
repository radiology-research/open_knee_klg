import os
import torch
import numpy as np
import logging
import time
import json
from datetime import datetime
from dataclasses import dataclass
from monai.networks.nets import EfficientNetBN
from monai.data import NibabelReader, NumpyReader, PILReader
from monai.data.meta_tensor import MetaTensor

import dataloaders_knee_kl as dataloaders

def image_to_int8(image):
        image = np.asarray(image).astype(np.float16)
        image -= np.min(image)
        image /= np.max(image)
        image *= 255
        image = image.astype(np.uint8)
        return image

def get_images(imgs_path):
    '''
    Get list of cropped image file paths from the specified directory.
    '''

    image_files = [os.path.join(imgs_path, png) for png in os.listdir(imgs_path) if (png.endswith(".png") and png.startswith('crop_'))]

    # sanitize input filenames to list
    if isinstance(image_files, str):
        imgs = [image_files]
    elif isinstance(image_files, list):
        imgs = image_files
    else:
        imgs = list(image_files)

    return imgs

@dataclass
class default_klg_args:
    model_arch_str: str = 'efficientnet-b7'
    img_size: int = 600
    pad_size: int = 600
    noise_prob = 0.0
    noise_range = (0,0)
    batch_size: int = 1
    number_of_classes: int = 6
    img_spatial_dims: int = 2
    in_channels: int = 1
    img_format: str = 'png'
    img_label: str = 'image'
    model_ckpt_f = 'klg_classifier.pth'


def klg_inference(path_to_images, output_dir, args, inference_device='cuda'):
    '''
    Run KLG inference on cropped knee images.
    '''
    # get inference device
    torch.multiprocessing.set_sharing_strategy('file_system')
    if inference_device == 'cuda':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    # input image
    infer_data = get_images(path_to_images)

    # create image transforms
    inference_transforms = dataloaders.get_padded_validation_transforms(
                                                img_resize=args.img_size, 
                                                pad_size=(args.pad_size,args.pad_size),
                                                noise_prob=args.noise_prob,
                                                noise_gauss=args.noise_range)

    # create appropiate image reader
    img_format = args.img_format.lower()
    if img_format=='png' or img_format=='jpg' or img_format=='bmp':
        img_reader = PILReader()
    elif img_format=='npz':
        img_reader = dataloaders.get_npz_image_reader(img_keys=args.img_label, ch=None)
    elif img_format=='nii':
        img_reader = NibabelReader()
    else:
        print("Image format not supported.")
        return

    # create inference data loader
    inference_loader, inference_ds = dataloaders.get_inference_loaders(infer_data, 
                                                    inference_transforms,
                                                    img_reader,
                                                    batch_size=args.batch_size)

    # Create EfficientNet
    model = EfficientNetBN(args.model_arch_str, 
                                            pretrained=True,
                                            spatial_dims=args.img_spatial_dims, 
                                            in_channels=args.in_channels, 
                                            num_classes=args.number_of_classes,
                                            ).to(device)
    # load model checkpoint
    torch.serialization.add_safe_globals([MetaTensor])

    checkpoint = torch.load(args.model_ckpt_f, weights_only=False, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    ### Run inference on images ###

    # set model to evaluation mode
    label_preds = []
    metadata = {}
    model.eval()

    # iterate through inference data loader
    for batch_data in inference_loader:
        meta = {}
        batch_images = batch_data.to(device)
        with torch.no_grad():
            batch_outputs = model(batch_images)
            argmax_preds = list(batch_outputs.argmax(dim=1).cpu().numpy().flatten())[0]
            # TODO: Switch below line(s) to identify left/right
            label_preds.extend([argmax_preds])
            metadata[batch_data.meta['filename_or_obj'][0]] = int(argmax_preds) 

            meta = {
                'filename': batch_data.meta['filename_or_obj'][0],
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'grade': int(argmax_preds)
            } 
            # write knee-level metadata to a JSON file
            with open(os.path.join(output_dir, f"metadata_assess-knees_{batch_data.meta['filename_or_obj'][0].split('_')[-1].replace('.png', '')}.json"), "w") as f:
                json.dump(meta, f, indent=4) 

    # return label predictions
    with open(os.path.join(output_dir, "preds.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    return label_preds, metadata