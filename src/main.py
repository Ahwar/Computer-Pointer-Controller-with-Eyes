import os
import sys
import time
import socket
import json
import cv2
import numpy as np
import logging
from pathlib import Path
from argparse import ArgumentParser

from face_detector import FaceDetector
from head_pose_estimator import HeadposeEstimator
from landmark_detector import LandmarkDetector
from gaze_estimator import GazeEstimator
from input_feeder import InputFeeder

from mouse_controller import MouseController
import logging

#Create and configure logger
logging.basicConfig(filename="main.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')

#Creating an object
logger = logging.getLogger()

#Setting the threshold of logger to DEBUG
logger.setLevel(logging.DEBUG)


def build_argparser():
    """
    Parse command line arguments.
    
    Return:
        args: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-ftm",
                        "--face_det_m",
                        required=True,
                        type=str,
                        help="Path to an xml file of Face Detection Model.")
    parser.add_argument("-ldm",
                        "--lmar_det_m",
                        required=True,
                        type=str,
                        help="Path to an xml file of Landmark Detection model")
    parser.add_argument(
        "-hem",
        "--h_pose_m",
        required=True,
        type=str,
        help="Path to an xml file of Head Pose Estimation model.")
    parser.add_argument("-gem",
                        "--g_est_m",
                        required=True,
                        type=str,
                        help="Path to an xml file of Gaze Estimation Model.")
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
    parser.add_argument(
        "-pt",
        "--prob_threshold",
        type=float,
        default=0.5,
        help="Probability threshold for face detections filtering"
        "(0.5 by default)")
    parser.add_argument(
        "-pr",
        "--preview",
        action='store_true',
        help="Use this flag if you want to preview visualizations on person face")
    return parser


def infer_on_stream(args):
    """
    Handle input Stream and perform inference Frame by Frame

    Check if input file is supported by OpenCV
    Load All models
    Perform inference Frame By Frame
        Detect and Crop face in Frame
    Detect left, right Eye
    Detect Head Pose
    Estimate Gaze 
    Move Mouse according to Gaze
    
    Parameters:
        args: Values of command line arguments

    Returns:
        None
    """
    # Check if all input files are present
    for _ in [
            args.face_det_m, args.lmar_det_m, args.h_pose_m, args.g_est_m,
            args.input
    ]:
        if not Path(_).is_file():
            error_message = "This file is not Present: \"{}\" Check the file please".\
                  format(_)
            logger.error(error_message)
            sys.exit(error_message)
        else:
            logger.info(
                "input files: {} is available on specified path".format(_))

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
        logger.error("Input file: {} is not Supported".format(args.input))

        sys.exit("FATAL ERROR : The format of your input file is not supported" \
                    "\nsupported extensions are : " + ", "\
                        .join(supported_img_exts + supported_vid_exts))

    ### Load All Models
    ## Load Face Detector Model
    face_detector = FaceDetector(args.face_det_m, args.device,
                                 args.cpu_extension)
    face_detector.load_model()

    logger.info("Face Detection model loaded successfully")
    ## Load Headpose Estimator Model
    headpose_estimator = HeadposeEstimator(args.h_pose_m, args.device,
                                           args.cpu_extension)
    headpose_estimator.load_model()
    logger.info("Headpose Estimator model loaded successfully")

    ## Load Landmark Detector Model
    landmark_detector = LandmarkDetector(args.lmar_det_m, args.device,
                                         args.cpu_extension)
    landmark_detector.load_model()
    logger.info("Landmark Detector model loaded successfully")

    ## Load Gaze Estimation Model
    gaze_estimator = GazeEstimator(args.g_est_m, args.device,
                                   args.cpu_extension)
    gaze_estimator.load_model()
    logger.info("Gaze Estimation model loaded successfully")
    ### Initialize Input Feeder
    input_feeder = InputFeeder(input_type, args.input)
    (initial_w, initial_h) = input_feeder.load_data()
    logger.info("Input Feeder loaded successfully")
    f_count = 0
    ### Iterate through input file frame by frame
    ### see `InputFeeder.next_batch` method for more detail
    for ret, frame in input_feeder.next_batch():
        # break if no next frame present
        if not ret:
            break
        f_count += 1
        logger.info("Processing Frame: {}".format(f_count))
        print("\nProcessing Frame: {}".format(f_count))

        ### Detect Face in Frame
        output = face_detector.predict(frame)
        ## Crop Face
        face, face_coords = face_detector.preprocess_output(
            output, args.prob_threshold, frame, initial_w, initial_h)

        # skip frame if face not found
        if not np.any(face):
            print("Face Not found in Frame\tSkipping Frame")
            logger.warning("Face Not found in Frame\tSkipping Frame")
            continue
        ### Estimate HeadPose
        head_pose = headpose_estimator.predict(face)
        logger.info("Head Pose Estimation complete")

        ### Detect Face Landmarks
        landmarks = landmark_detector.predict(face)
        logger.info("Face Landmarks detected")
        ## Crop left and right Eye
        left_eye, left_eye_coords, right_eye, right_eye_coords = landmark_detector.preprocess_output(
            landmarks, face)

        ## Skip frame if any eye is not cropped correctly
        if 0 in left_eye.shape or 0 in right_eye.shape:
            print("Issue in Eye Cropping. \nSkipping this Frame ...")
            logger.warning("Issue in Eye Cropping. \nSkipping this Frame ...")
            continue
        logger.info("Both Eyes cropped successfuly")
        ### Estimate Gaze
        gaze = gaze_estimator.predict(left_eye, right_eye, head_pose)
        logger.info("Gaze Estimated successfully")
        ## Get mouse coords (x, y)
        mouse_coords = gaze_estimator.preprocess_output(gaze, head_pose)
        logger.info("New mouse coordinates: {}".format(mouse_coords))
        # Show Preview of input with drawn predictions
        if (args.preview):
            # function draw rectangel around eye
            def rectange_eyes(frame, face_coords, eye_coords):
                """Draw bounding box around Eye"""
                eye_start = (
                    (face_coords[0][0] + eye_coords[0][0]),  # x_min + x_min
                    (face_coords[0][1] + eye_coords[0][1]))  # y_min _ y_min
                eye_end = (
                    (face_coords[0][0] + eye_coords[1][0]),  # x_min + x_max
                    (face_coords[0][1] + eye_coords[1][1]))  # y_min + y_max

                return cv2.rectangle(frame, eye_start, eye_end, (0, 0, 255), 2)

            # draw box around face
            image = cv2.rectangle(frame, face_coords[0], face_coords[1],
                                  (0, 0, 255), 2)
            # draw box around left eye
            image = rectange_eyes(image, face_coords, left_eye_coords)
            # draw box around right eye
            image = rectange_eyes(image, face_coords, right_eye_coords)
            # show head pose values on image
            cv2.putText(
                image,
                "Head Pose: Yaw: {:.2f}, Pitch: {:.2f}, Roll: {:.2f}".format(
                    head_pose["angle_y_fc"][0][0],
                    head_pose["angle_y_fc"][0][0],
                    head_pose["angle_y_fc"][0][0],
                ), (40, 40), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 1)
            # show head pose values on image
            cv2.putText(
                image,
                "Gaze: X-axis: {:.2f}, Y-axis: {:.2f}, Z-axis: {:.2f}".format(
                    gaze[0][0], gaze[0][1], gaze[0][2]), (40, 70),
                cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 1)

            cv2.imshow('Preview | press q to close', image)
            # break loop if q is pressed on output window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        print("New mouse coordinates: {}\n\n".format(mouse_coords))
        ### Move Mouse
        mouse_controler = MouseController("medium", "medium")
        mouse_controler.move(mouse_coords[0], mouse_coords[1])
        # go to next frame

    ### Processing Complete delete resources
    print("Input File is complete \nClearing Resources")
    logger.info("Input File is complete Clearing Resources")
    input_feeder.close()
    del face_detector
    del landmark_detector
    del headpose_estimator
    del gaze_estimator
    logger.info("Most Heavy Resources are Free now")


def main():
    """
    Load the network and parse the output.

    Returns:
        None
    """
    # Grab command line args
    logger.info("Starting Program")
    args = build_argparser().parse_args()
    # Perform inference on the input stream
    infer_on_stream(args)
    logger.info("Every Thing Complete Exiting Program")


if __name__ == '__main__':
    main()