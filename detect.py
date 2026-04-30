import os
import json
from datetime import datetime
import torch
# from ultralytics import YOLO
# from ultralytics.nn import tasks
import onnxruntime as ort
from pydicom import dcmread
import skimage
import numpy as np
# from PIL import Image
import cv2


decode_cls_ids = {
     0: 'right',
     1: 'left'
}

# Function to convert DICOM to PNG and save it with the correct filename
def preprocess_dcm(dcm_file_path, output_shape: tuple[int, int] = (640, 640)):
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
    # img = Image.fromarray(pixel_array)

    # Convert to 3 channel image
    img = np.stack([pixel_array, pixel_array, pixel_array], axis=0)

    # Pad image to input size of model
    img, pad = letterbox(img, output_shape)

    # Normalize the image data to 0-1 range
    img = np.array(img) / 255.0

    # Expand the dimensions of the image data to match the expected input shape
    img = img[None].astype(np.float32)

    return img, pad

def letterbox(img: np.ndarray, new_shape: tuple[int, int] = (640, 640)) -> tuple[np.ndarray, tuple[int, int]]:
        """Resize and reshape images while maintaining aspect ratio by adding padding.

        Args:
            img (np.ndarray): Input image to be resized.
            new_shape (tuple[int, int]): Target shape (height, width) for the image.

        Returns:
            img (np.ndarray): Resized and padded image.
            pad (tuple[int, int]): Padding values (top, left) applied to the image.
        """
        shape = img.shape[:2]  # current shape [height, width]

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        new_unpad = round(shape[1] * r), round(shape[0] * r)
        dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2  # wh padding

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = round(dh - 0.1), round(dh + 0.1)
        left, right = round(dw - 0.1), round(dw + 0.1)
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        return img, (top, left)

def postprocess(output: list[np.ndarray], input_shape: tuple[int, int], pad: tuple[int, int], confidence_thres=0.5, iou_thres=0.7) -> np.ndarray:
        """Perform post-processing on the model's output to extract and visualize detections.

        This function processes the raw model output to extract bounding boxes, scores, and class IDs. It applies
        non-maximum suppression to filter overlapping detections and draws the results on the input image.

        Args:
            output (list[np.ndarray]): The output arrays from the model.
            pad (tuple[int, int]): Padding values (top, left) used during letterboxing.

        Returns:
            (np.ndarray): The input image with detections drawn on it.
        """
        # Transpose and squeeze the output to match the expected shape
        outputs = np.transpose(np.squeeze(output[0]))

        # Get the number of rows in the outputs array
        rows = outputs.shape[0]

        # Lists to store the bounding boxes, scores, and class IDs of the detections
        boxes = []
        scores = []
        class_ids = []

        # Lists to store final bounding boxes, scores, and class IDs after non-maximum suppression
        final_boxes = []
        final_scores = []
        final_class_ids = []

        # Calculate the scaling factors for the bounding box coordinates
        gain = min(input_shape[0] / input_shape[0], input_shape[1] / input_shape[1])
        outputs[:, 0] -= pad[1]
        outputs[:, 1] -= pad[0]

        # Iterate over each row in the outputs array
        for i in range(rows):
            # Extract the class scores from the current row
            classes_scores = outputs[i][4:]

            # Find the maximum score among the class scores
            max_score = np.amax(classes_scores)

            # If the maximum score is above the confidence threshold
            if max_score >= confidence_thres:
                # Get the class ID with the highest score
                class_id = np.argmax(classes_scores)

                # Extract the bounding box coordinates from the current row
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # Calculate the scaled coordinates of the bounding box
                left = int((x - w / 2) / gain)
                top = int((y - h / 2) / gain)
                width = int(w / gain)
                height = int(h / gain)

                # Add the class ID, score, and box coordinates to the respective lists
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, scores, confidence_thres, iou_thres)

        # Iterate over the selected indices after non-maximum suppression
        for i in np.array(indices).flatten():
            # Get the box, score, and class ID corresponding to the index
            final_boxes.append(boxes[int(i)])
            final_scores.append(scores[int(i)])
            final_class_ids.append(class_ids[int(i)])

        # Return the final results
        return final_boxes, final_scores, final_class_ids

def xywh2xyxy(xywh, input_shape=(640, 640)):
    """Convert bounding box format from (x_left, y_top, width, height) to (x1, y1, x2, y2).

    Args:
        xywh (list[float]): A list containing the bounding box coordinates in the format (x_left, y_top, width, height).
        input_shape (tuple[int, int]): The shape of the input image.

    Returns:
        list[float]: A list containing the bounding box coordinates in the format (x1, y1, x2, y2).
    """

    x_left, y_top, width, height = xywh
    x1 = int(x_left*input_shape[1])
    y1 = int(y_top*input_shape[0])
    x2 = int((x_left + width)*input_shape[1])
    y2 = int((y_top + height)*input_shape[0])

    return [x1, y1, x2, y2]

def detect_knees(input_src, output_dest, mdl_weights='knee_yolo.onnx', conf_threshold=0.5):
    '''
    Run AP knee joint detection
    '''

    # Load model weights
    # torch.serialization.add_safe_globals([tasks.DetectionModel, torch.nn.Sequential])
    # model = YOLO(mdl_weights)
    session = ort.InferenceSession(mdl_weights)

    # Get the model inputs
    model_inputs = session.get_inputs()
    input_shape = model_inputs[0].shape

    # Load DICOM and preprocess image
    img, pad = preprocess_dcm(input_src, output_shape=(input_shape[2], input_shape[3]))

    # Run knee joint detection
    # results = model.predict(source=img, name = "ucsf-ci2-klnet", conf=conf_threshold, max_det=4, save=False, save_txt=False, agnostic_nms=True)
    # Run inference using the preprocessed image data
    outputs = session.run(None, {model_inputs[0].name: img})

    # Postprocess the outputs
    boxes, confs, cls_ids = postprocess(img, outputs, (input_shape[2], input_shape[3]), pad, confidence_thres=conf_threshold)

    assert len(results) == 1 # only 1 image should be fed in at a time

    # save results
    results[0].save(filename=os.path.join(output_dest, "detect_result.jpg"))

    # Extract bounding box coords from tensors and log resutls
    # boxes = results[0].boxes
    # xyxy = boxes.xyxy.cpu().numpy().astype(int)
    # cls_ids = boxes.cls.cpu().numpy().astype(int)         # (N,)
    # decode_cls_ids = results[0].names
    # confs   = boxes.conf.cpu().numpy()                    # (N,)

    if len(boxes) == 0:
        print('No knees detected.')
    elif len(boxes) <= 2:
        print('Detected', len(boxes), 'knees.')
    else:
        raise Exception('Too many boxes found:', len(boxes))
    
    for box in boxes:
        x1, y1, w, h = box
        print(f"Bounding Box: x_left={x1}, y_top={y1}, width={w}, height={h}")

    # Use bounding box coords to return cropped subsections of original image
            
    crops = []
    for i, coords in enumerate(boxes):
        
        x1, y1, x2, y2 = xywh2xyxy(coords, input_shape=(input_shape[2], input_shape[3]))
        cls_id = int(cls_ids[i])
        label  = decode_cls_ids[cls_id]
        conf   = float(confs[i])

        # Generate Metadata
        meta = {
            'filename': input_src,
            'timestampe': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'cls_id': cls_id,
            'label': label,
            'conf': conf,
            'bbox_coords': coords,
        }
        # write metadata to a JSON file
        with open(os.path.join(output_dest, f"metadata_detect-knees_{label}.json"), "w") as f:
            json.dump(meta, f, indent=4)

        # Save image of cropped knee
        gray_img = img[0,0,:,:] * 255.0
        crop = np.array(gray_img)[y1:y2, x1:x2]  # note: [y1:y2, x1:x2] (row, col)
        crops.append(crop)

        Image.fromarray(crop).save(os.path.join(output_dest, f"crop_{label}.png"))

    return crops, boxes