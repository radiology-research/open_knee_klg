import os
import argparse
import logging
import time
import json

from detect import detect_knees
from assess import klg_inference, default_klg_args

def run_inference(input, output, verbose=False):
    '''
    Run KLG inference on AP Knee Radiograph DICOM file.
    '''

    # Set up outputs for knee detection and KLG inference
    detect_output_dir = os.path.join(output, "detect_output")
    klg_output_dir = os.path.join(output, "klg_output")
    os.makedirs(detect_output_dir, exist_ok=True)
    os.makedirs(klg_output_dir, exist_ok=True)

    # Call the knee detection model
    bbox_imgs, bbox_arrs = detect_knees(input_src=input, output_src=detect_output_dir)
    if verbose:
        print(f"Detected {len(bbox_imgs)} knees in the input DICOM file.")
        print(f"Bounding box coordinates for detected knees: {bbox_arrs}")

    # Call the KLG inference model on the detected knee crops
    klg_args = default_klg_args()
    klg_results, klg_dict = klg_inference(path_to_images=detect_output_dir, output_dir=klg_output_dir, args=klg_args)
    if verbose:
        print(f"KLG Inference: {klg_dict}")

if __name__ == "__main__":
    # Parse
    parser = argparse.ArgumentParser(description="Detect knees from MRI scans")
    parser.add_argument('-i', '--input', required=True, help="Path to the input DICOM file")
    parser.add_argument('-o', '--output', required=True, help="Path to the output directory")
    parser.add_argument('--verbose', action='store_true', default=False, help="Print verbose output to console")

    args = parser.parse_args()
    
    # Set Up
    logging.basicConfig(
        filename=os.path.join(args.output, "knee_klg.log"),    # file to write to
        filemode="a", 
        level=logging.INFO, 
    )
    start_time = time.time()
    logging.info(f"Start Time: {start_time:.4f} seconds")
   
    # Call the main function
    run_inference(input=args.input, output=args.output, verbose=args.verbose)

    # Stop timer
    end_time = time.time()
    logging.info(f"End Time: {end_time:.4f} seconds")
