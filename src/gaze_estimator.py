import os
import sys
import cv2
import math
import numpy as np
import logging as log
from openvino.inference_engine import IECore
'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
'''


class GazeEstimator:
    '''
    Class for the Gaze Estimation Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        ''' Use this to set your instance variables. '''
        self.core = None
        self.network = None
        self.output_blob = None
        self.exec_network = None
        self.infer_request = None
        self.device = device
        self.extenstions = extensions
        self.model_xml = model_name

    def load_model(self):
        '''
        Load Model file and create Executable Network

        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins (e.g. CPU extensions), this is where you can load them.
        Initialize Core
        Read model in to IENetwork and add any necessary Extension
        Check for Supported Layers
        '''
        self.model_weights = os.path.splitext(self.model_xml)[0] + ".bin"
        ### Load the model ###
        # Take model .xml and .bin file and create IENetwork
        self.core = IECore()
        self.network = self.core.read_network(model=self.model_xml,
                                              weights=self.model_weights)

        ### Add any necessary extensions ###
        try:
            if self.extenstions and self.device == 'CPU':
                self.core.add_extension(extension_path=self.extenstions,
                                        device_name=self.device)
        except:
            "Error in Loading {} Extension".format(self.device)

        ### Check model for unsupported layers
        self.check_model()

        # retrieve name of model's output layer
        self.output_blob = next(iter(self.network.outputs))

        ### load IENetwork to Executable Network ###
        ### Note: You may need to update the function parameters. ###
        self.exec_network = self.core.load_network(network=self.network,
                                                   device_name=self.device,
                                                   num_requests=1)

    def predict(self, left_eye, right_eye, head_pose):
        '''
        Perform Inference on Image and return Raw Output.

        This method is meant for running predictions on the input image.
        Parameters:
            image (numpy.ndarray): Frame from input file
        
        Returns:
            model_output (numpy.ndarray): Raw Model Output
        '''
        ### PreProcess input image according to model Requirement
        left_eye = self.preprocess_input(left_eye)
        right_eye = self.preprocess_input(right_eye)
        ### run inference and return output
        # Start Async Inference Request
        poses = [[
            head_pose["angle_y_fc"][0][0],  # Estimated Head yaw (in degrees)
            head_pose["angle_p_fc"][0][0],  # Estimated Head pitch (in degrees)
            head_pose["angle_r_fc"][0][0]  # Estimated Head roll (in degrees)
        ]]

        # print("aksdfjdskfajskdfjasdfk", v)
        infer_request_handle = self.exec_network.start_async(
            request_id=0,
            # Input dictionary
            inputs={
                # image of left eye
                "left_eye_image": left_eye,
                # image of right eye
                "right_eye_image": right_eye,
                # head pose angles
                "head_pose_angles": poses
            })

        # wait for the output and return.
        if infer_request_handle.wait(-1) == 0:
            model_output = infer_request_handle.outputs[self.output_blob]
        return model_output

    def check_model(self):
        """Check for supported layers"""
        layers_map = self.core.query_network(network=self.network,
                                             device_name=self.device)

        unsupported_layers = [
            l for l in self.network.layers.keys() if l not in layers_map
        ]

        if (unsupported_layers != []):
            sys.exit("Those mention layers in your model are not supported by OpenVino Inference Engine:" \
                     " \n\t" + "\n\t".join(unsupported_layers))

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        (b, c, h, w) = self.get_input_shape()
        image = cv2.resize(image, (w, h))
        image = np.transpose(image, (2, 0, 1))
        image = image.reshape(b, c, h, w)

        return image

    def preprocess_output(self, outputs, hpe_cords):
        """
        Model output is dictionary like this
        {'gaze_vector': array([[ 0.51141196,  0.12343533, -0.80407059]], dtype=float32)}
        containing Cartesian coordinates of gaze direction vector
        We need to get this value and convert it in required format
        hpe_cords which is output of head pose estimation is in radian
        It needed to be converted in catesian cordinate
        """
        gaze_vector = outputs[0]
        mouse_cord = (0, 0)
        try:
            angle_r_fc = hpe_cords["angle_r_fc"]
            sin_r = math.sin(angle_r_fc * math.pi / 180.0)
            cos_r = math.cos(angle_r_fc * math.pi / 180.0)
            x = gaze_vector[0] * cos_r + gaze_vector[1] * sin_r
            y = -gaze_vector[0] * sin_r + gaze_vector[1] * cos_r
            mouse_cord = (x, y)
        except Exception as e:
            print(
                "Error While preprocessing output in Gaze Estimation Model" +
                str(e))
        return mouse_cord

    def get_input_shape(self):
        """ Return the shape of the input layer """
        return self.network.inputs["left_eye_image"].shape