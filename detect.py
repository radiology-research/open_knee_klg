import os
import json
from datetime import datetime
import torch
from ultralytics import YOLO
from pydicom import dcmread
import skimage
import numpy as np
from PIL import Image

# Function to convert DICOM to PNG and save it with the correct filename
def preprocess_dcm(dcm_file_path):
    '''
    Read DICOM file and preprocess image. 
    '''
    
    # Read the DICOM file
    dcm = dcmread(dcm_file_path)
    # Get the pixel data from the DICOM file
    pixel_array = dcm.pixel_array

    # Normalize the pixel values based on histogramequalization
    pixel_array = skimage.exposure.equalize_hist(pixel_array)
    pixel_array = (pixel_array * 255).astype(np.uint8)
    
    # Convert the array to a PIL image
    img = Image.fromarray(pixel_array)

    return img

def detect_knees(input_src, output_dest, mdl_weights='knee_yolo.pt', conf_threshold=0.5):
    '''
    Run AP knee joint detection
    '''

    # Load DICOM and preprocess image
    img = preprocess_dcm(input_src)

    # Load model weights
    torch.serialization.add_safe_globals([tasks.DetectionModel, torch.nn.Sequential])
    model = YOLO(mdl_weights)

    # Run knee joint detection
    results = model.predict(source=img, name = "ucsf-ci2-klnet", conf=conf_threshold, max_det=4, save=False, save_txt=False, agnostic_nms=True)


    assert len(results) == 1 # only 1 image should be fed in at a time

    # save results
    results[0].save(filename=os.path.join(output_dest, "detect_result.jpg"))

    # Extract bounding box coords from tensors and log resutls
    boxes = results[0].boxes
    xyxy = boxes.xyxy.cpu().numpy().astype(int)
    cls_ids = boxes.cls.cpu().numpy().astype(int)         # (N,)
    decode_cls_ids = results[0].names
    confs   = boxes.conf.cpu().numpy()                    # (N,)

    if len(boxes) == 0:
        print('No knees detected.')
    elif len(boxes) <= 2:
        print('Detected', len(boxes), 'knees.')
    else:
        raise Exception('Too many boxes found:', len(boxes))
    for box in boxes.xyxy.cpu():
        x1, y1, x2, y2 = box
        print(f"Bounding Box: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

    # Use bounding box coords to return cropped subsections of originalimage,
            
    crops = []
    for i, coords in enumerate(xyxy):
        
        x1, y1, x2, y2 = coords
        cls_id = int(cls_ids[i])
        label  = decode_cls_ids[cls_id]
        conf   = float(confs[i])

        # Generate Metadata
        meta = {
            'filename': input_src,
            'timestampe': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            cls_id: cls_id,
            label: labels,
            conf: conf,
            'bbox_coords': coords.tolist()
        }
        # write metadata to a JSON file
        with open(os.path.join(output_dest, f"metadata_detect-knees_{decode_cls_ids[cls_id]}.json"), "w") as f:
            json.dump(meta, f, indent=4)

        # Save image of cropped knee
        crop = np.array(img)[y1:y2, x1:x2]  # note: [y1:y2, x1:x2] (row, col)
        crops.append(crop)
        # if output_dest:
        Image.fromarray(crop).save(os.path.join(output_dest, f"crop_{decode_cls_ids[cls_id]}.png"))

    return crops, xyxy