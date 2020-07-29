import os
import sys
import time
import socket
import json
import cv2
import numpy as np
from argparse import ArgumentParser

from model import Model_X
from input_feeder import InputFeeder


def build_argparser():
    """
    Parse command line arguments.
    
    Return:
        args: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m",
                        "--model",
                        required=True,
                        type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i",
                        "--input",
                        required=True,
                        type=str,
                        help="Path to image or video file")
    parser.add_argument("-l",
                        "--cpu_extension",
                        required=False,
                        type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                        "Absolute path to a shared library with the"
                        "kernels impl.")
    parser.add_argument("-d",
                        "--device",
                        type=str,
                        default="CPU",
                        help="Specify the target device to infer on: "
                        "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                        "will look for a suitable plugin for device "
                        "specified (CPU by default)")
    parser.add_argument("-pt",
                        "--prob_threshold",
                        type=float,
                        default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def infer_on_stream(args):
    """
    Handle input Stream and perform inference Frame by Frame

    Check if input file is supported by OpenCV
    Load All models
    Perform inference Frame By Frame
        Detect and Crop face in Frame
    
    Parameters:
        args: Values of command line arguments

    Returns:
        None
    """

    ### Handle the input stream ###
    # extenstion of input file
    input_extension = os.path.splitext(args.input)[1].lower()
    # supported extensions
    supported_vid_exts = ['.mp4', '.mpeg', '.avi', '.mkv']
    supported_img_exts = [
        ".bmp", ".dib", ".jpeg", ".jp2", ".jpg", ".jpe", ".png", ".pbm",
        ".pgm", ".ppm", ".sr", ".ras", ".tiff", ".tif"
    ]
    # if input is camera
    if args.input.upper() == 'CAM':
        input_type = "cam"

    # if input is video
    elif input_extension in supported_vid_exts:
        input_type = "video"

    # if input is image
    elif input_extension in supported_img_exts:
        input_type = "image"
    else:
        sys.exit("FATAL ERROR : The format of your input file is not supported" \
                    "\nsupported extensions are : " + ", "\
                        .join(supported_img_exts + supported_vid_exts))
    ### Load All Models
    ## Load Face Detector Model
    face_detector = Model_X(args.model, args.device, args.cpu_extension)
    face_detector.load_model()

    ### Initialize Input Feeder
    input_feeder = InputFeeder(input_type, args.input)
    (initial_w, initial_h) = input_feeder.load_data()

    f_count = 0
    ### Iterate through input file frame by frame
    ### see `InputFeeder.next_batch` method for more detail
    for ret, frame in input_feeder.next_batch():
        # break if no next frame present
        if not ret:
            break
        f_count = f_count + 1
        print("Processing Frame: ", f_count)

        ### Detect Face in Frame
        output = face_detector.predict(frame)
        face = face_detector.preprocess_output(output, 0.6, frame, initial_w,
                                               initial_h)

        # skip frame if face not found
        if not np.any(face):
            print("Face Not found in Frame")
            continue
        cv2.imwrite('output.png', face)

    print("Input File is complete \nClearing Resources")
    input_feeder.close()
    del face_detector


def main():
    """
    Load the network and parse the output.

    Returns:
        None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Perform inference on the input stream
    infer_on_stream(args)


if __name__ == '__main__':
    main()