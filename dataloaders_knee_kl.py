import numpy as np

from monai.config import print_config
from monai.data import DataLoader, ImageDataset, NumpyReader, PILReader
from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    AsDiscrete,
    Compose,
    LoadImage,
    RandFlip,
    RandRotate,
    RandRotate90,
    Resize,
    RandAxisFlip,
    RandZoom,
    ScaleIntensity,
    ResizeWithPadOrCrop,
    RandGaussianNoise,
    RandSpatialCrop,
    HistogramNormalize,
    RandAffine,
    SpatialPad,
    ResizeWithPadOrCrop,
    ScaleIntensity,
    RandScaleIntensity,
    RandShiftIntensity,
    RandAdjustContrast,
    ThresholdIntensity,
    NormalizeIntensity,
)

def get_training_transforms(img_resize=(256,256), rotate_prob=0.95, flip_prob=0.5, affine_prob=0.75, noise_prob=0.95, rotate_range=np.pi/4, translate_range=(10,10), noise_gauss=(0,0.05)):
    
    train_transforms = Compose([
        EnsureChannelFirst(),
        RandRotate(range_x=rotate_range, prob=rotate_prob, keep_size=True, padding_mode='zeros'),
        RandFlip(prob=flip_prob, spatial_axis=1),
        Resize(img_resize), 
        RandAffine(prob=affine_prob, rotate_range=None, shear_range=None, translate_range=translate_range, padding_mode='zeros'),
        RandGaussianNoise(prob=noise_prob, mean=noise_gauss[0], std=noise_gauss[1])
        ])
    
    return train_transforms

def get_validation_transforms(img_resize=(256,256), noise_prob=0.95, noise_gauss=(0,0.05)):
    
    validation_transforms = Compose([
        EnsureChannelFirst(), 
        Resize(img_resize), 
        RandGaussianNoise(prob=noise_prob, mean=noise_gauss[0], std=noise_gauss[1])
        ])
    
    return validation_transforms

def get_padded_training_transforms(img_resize=256, pad_size=(256,256), rotate_prob=0.95, flip_prob=0.5, affine_prob=0.75, noise_prob=0.95, shift_prob=0.2, contrast_prob=0.2, rotate_range=np.pi/4, translate_range=(10,10), noise_gauss=(0,0.05), randscale=(0.5,1.5), randshift=(0.01,0.1), gamma_range=(0.5, 2), scale=(0,1), normalize=(0.485,0.229), dtype=np.float32):
    
    train_transforms = Compose([
        EnsureChannelFirst(),
        Resize(img_resize, size_mode='longest'), 
        SpatialPad(pad_size, mode='constant'),
        RandRotate(range_x=rotate_range, prob=rotate_prob, keep_size=True, padding_mode='zeros'),
        RandFlip(prob=flip_prob, spatial_axis=1),
        RandAffine(prob=affine_prob, rotate_range=None, shear_range=None, translate_range=translate_range, padding_mode='zeros'),
        RandGaussianNoise(prob=noise_prob, mean=noise_gauss[0], std=noise_gauss[1]),
        ScaleIntensity(minv=scale[0],maxv=scale[1], dtype=dtype),
        RandScaleIntensity(randscale, prob=shift_prob),
        RandShiftIntensity(randshift, prob=shift_prob, safe=True),
        RandAdjustContrast(prob=contrast_prob, gamma=gamma_range, invert_image=False, retain_stats=False), 
        NormalizeIntensity(subtrahend=normalize[0], divisor=normalize[1], dtype=dtype),
        ])
    
    return train_transforms

def get_padded_validation_transforms(img_resize=256, pad_size=(256,256), noise_prob=0.95, noise_gauss=(0,0.05), scale=(0,1), normalize=(0.485,0.229), dtype=np.float32):
    
    validation_transforms = Compose([
        EnsureChannelFirst(), 
        Resize(img_resize, size_mode='longest'), 
        SpatialPad(pad_size, mode='constant'),
        RandGaussianNoise(prob=noise_prob, mean=noise_gauss[0], std=noise_gauss[1]),
        ScaleIntensity(minv=scale[0],maxv=scale[1], dtype=dtype),
        NormalizeIntensity(subtrahend=normalize[0], divisor=normalize[1], dtype=dtype),
        ])
    
    return validation_transforms

def get_npz_image_reader(img_keys=['wb_max_intensity'], ch=None):
    '''
    img_keys = [diff, max_intensity, wb_diff, wb_max_intensity, fgt_max_intensity]
    '''
    
    img_reader = NumpyReader(npz_keys=img_keys, channel_dim=ch)
    
    return img_reader

def get_inference_loaders(img_infer, infer_transforms, img_reader, batch_size=1):

    # create a validation data loader
    infer_ds = ImageDataset(image_files=img_infer, transform=infer_transforms, reader=img_reader)
    infer_loader = DataLoader(infer_ds, batch_size=batch_size, num_workers=1, pin_memory=False)
    
    return infer_loader, infer_ds