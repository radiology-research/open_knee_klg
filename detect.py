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
from PIL import Image
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
    
    # Pad image to input size of model
    img, pad = letterbox(pixel_array, output_shape)

    # Normalize the image data to 0-1 range
    img = np.array(img) / 255.0

    # Convert to 3 channel image
    img = np.stack([img, img, img], axis=0)

    # Expand the dimensions of the image data to match the expected input shape
    img = img[None].astype(np.float32)

    return img, pixel_array, pad

def letterbox(img: np.ndarray, new_shape: tuple[int, int] = (640, 640)):
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

def save_detection_result(
    img,
    boxes,
    confs,
    cls_ids,
    output_path,
    line_width=None,
):
    """
    Draw detected knee bounding boxes and save an annotated image.
    """

    annotated = img.copy()
    h, w = annotated.shape[:2]

    if line_width is None:
        line_width = max(round((h + w) / 2 * 0.003), 2)

    font_scale = line_width / 3
    font_thickness = max(line_width - 1, 1)

    # Ultralytics-like color palette
    palette = [
        (255, 56, 56),
        (255, 157, 151),
        (255, 112, 31),
        (255, 178, 29),
        (207, 210, 49),
        (72, 249, 10),
        (146, 204, 23),
        (61, 219, 134),
        (26, 147, 52),
        (0, 212, 187),
        (44, 153, 168),
        (0, 194, 255),
        (52, 69, 147),
        (100, 115, 255),
        (0, 24, 236),
        (132, 56, 255),
        (82, 0, 133),
        (203, 56, 255),
        (255, 149, 200),
        (255, 55, 199),
    ]

    assert len(boxes) == len(confs) and len(confs) == len(cls_ids)
    for i in range(0, len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        conf, class_id = confs[i], cls_ids[i]

        x1 = int(round(x1))
        y1 = int(round(y1))
        x2 = int(round(x2))
        y2 = int(round(y2))
        class_id = int(class_id)

        # Ultralytics palette is RGB; OpenCV expects BGR
        rgb = palette[class_id % len(palette)]
        color = rgb[::-1]

        if decode_cls_ids is None:
            decode_cls_id = str(class_id)
        elif isinstance(decode_cls_ids, dict):
            decode_cls_id = decode_cls_ids[class_id]
        else:
            decode_cls_id = decode_cls_ids[class_id]

        label = f"{decode_cls_id} {conf:.2f}"

        # Bounding box
        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            color,
            thickness=line_width,
            lineType=cv2.LINE_AA,
        )

        # Label dimensions
        (text_w, text_h), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            font_thickness,
        )

        # Put label above box if there is room, otherwise inside
        outside = y1 >= text_h + 3

        if outside:
            label_y1 = y1 - text_h - 6
            label_y2 = y1
            text_y = y1 - 4
        else:
            label_y1 = y1
            label_y2 = y1 + text_h + 6
            text_y = y1 + text_h + 3

        cv2.rectangle(
            annotated,
            (x1, label_y1),
            (x1 + text_w + 4, label_y2),
            color,
            thickness=-1,
        )

        cv2.putText(
            annotated,
            label,
            (x1 + 2, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA,
        )

    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
        
    if not cv2.imwrite(output_path, annotated):
        raise RuntimeError(f"Failed to save image to {output_path}")

    return annotated

def postprocess(output: list[np.ndarray], input_shape: tuple[int, int], pred_shape: tuple[int, int], pad: tuple[int, int], confidence_thres=0.5, iou_thres=0.7):
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
        coords = []
        scores = []
        class_ids = []

        # Lists to store final bounding boxes, scores, and class IDs after non-maximum suppression
        final_boxes = []
        final_coords = []
        final_scores = []
        final_class_ids = []

        # Calculate the scaling factor for the bounding box coordinates
        gain = min(
            pred_shape[0] / input_shape[0],
            pred_shape[1] / input_shape[1]
        )

        # Iterate over each row in the outputs array
        for i in range(rows):
            # Unpack
            x, y, w, h = outputs[i][:4]
            classes_scores = outputs[i][4:]
            assert len(classes_scores) == 2

            # Find the maximum score among the class scores
            max_score = np.amax(classes_scores)

            # Skip if the maximum score is less than the confidence threshold
            if max_score < confidence_thres:
                continue

            # Otherwise read and rescale bbox

            # Get the class ID with the highest score
            class_id = np.argmax(classes_scores)

            # Unpad
            x -= pad[1]
            y -= pad[0]

            # Undo resize
            x /= gain
            y /= gain
            w /= gain
            h /= gain

            # Calculate the scaled coordinates of the bounding box
            left = int(x - w / 2)
            top = int(y - h / 2)
            width = int(w)
            height = int(h)

            # Get x1, x2, y1, y2
            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)

            # Clip to original image boundaries
            orig_h, orig_w = input_shape

            x1 = max(0, min(x1, orig_w - 1)) # NOTE: this could be asserts now, since this won't happen
            y1 = max(0, min(y1, orig_h - 1))
            x2 = max(0, min(x2, orig_w - 1))
            y2 = max(0, min(y2, orig_h - 1))
            xyxy = [x1, y1, x2, y2]

            # Add the class ID, score, and box coordinates to the respective lists
            class_ids.append(class_id)
            scores.append(max_score)
            boxes.append([left, top, width, height])
            coords.append(xyxy)

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, scores, confidence_thres, iou_thres)

        # Iterate over the selected indices after non-maximum suppression
        for i in np.array(indices).flatten():
            # Get the box, score, and class ID corresponding to the index
            final_boxes.append(boxes[int(i)])
            final_coords.append(coords[int(i)])
            final_scores.append(scores[int(i)])
            final_class_ids.append(class_ids[int(i)])

        # Return the final results
        return final_boxes, final_coords, final_scores, final_class_ids

# def xywh2xyxy(xywh, input_shape=(640, 640)):
#     """Convert bounding box format from (x_left, y_top, width, height) to (x1, y1, x2, y2).

#     Args:
#         xywh (list[float]): A list containing the bounding box coordinates in the format (x_left, y_top, width, height).
#         input_shape (tuple[int, int]): The shape of the input image.

#     Returns:
#         list[float]: A list containing the bounding box coordinates in the format (x1, y1, x2, y2).
#     """

#     x_left, y_top, width, height = xywh
#     x1 = int(x_left*input_shape[1])
#     y1 = int(y_top*input_shape[0])
#     x2 = int((x_left + width)*input_shape[1])
#     y2 = int((y_top + height)*input_shape[0])

#     return [x1, y1, x2, y2]

# def detect_knees(input_src, output_dest, mdl_weights='knee_yolo.onnx', conf_threshold=0.5):
def detect_knees(input_src, output_dest, mdl_weights='best.onnx', conf_threshold=0.5):
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
    img, img_src, pad = preprocess_dcm(input_src, output_shape=(input_shape[2], input_shape[3]))

    save_detection_result(
        img[0,0],
        [[1,2,3,4]],
        [1],
        [1],
        os.path.join(output_dest, "img.jpg"),
    )
    save_detection_result(
        img_src,
        [[1,2,3,4]],
        [0],
        [0],
        os.path.join(output_dest, "img_src.jpg"),
    )

    # Run inference using the preprocessed image data
    outputs = session.run(None, {model_inputs[0].name: img})


    # Postprocess the outputs: save image, extract boundng box coords, log results
    boxes, coords, confs, cls_ids = postprocess(outputs, img_src.shape, img.shape[-2:], pad, confidence_thres=conf_threshold)

    assert ((boxes == []) or (len(boxes[0]) == 4) or (len(boxes) == 2 and len(boxes[0]) == 4 and len(boxes[1]) == 4)) # ensure output is 0, 1 or 2 boxes
    assert len(boxes) == len(coords) and len(coords) == len(confs) and len(confs) == len(cls_ids)

    # save results
    # results[0].save(filename=os.path.join(output_dest, "detect_result.jpg"))
    save_detection_result(
        img_src,
        coords,
        confs,
        cls_ids,
        os.path.join(output_dest, "detect_result.jpg"),
    )

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
    for i, coord in enumerate(coords):
        x1, y1, x2, y2 = coord
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
            'bbox_coords': coord,
        }
        # write metadata to a JSON file
        with open(os.path.join(output_dest, f"metadata_detect-knees_{label}.json"), "w") as f:
            json.dump(meta, f, indent=4)

        # Save image of cropped knee
        crop = np.array(img_src)[y1:y2, x1:x2]  # note: [y1:y2, x1:x2] (row, col)
        crops.append(crop)

        Image.fromarray(crop).save(os.path.join(output_dest, f"crop_{label}.png"))

    return crops, boxes