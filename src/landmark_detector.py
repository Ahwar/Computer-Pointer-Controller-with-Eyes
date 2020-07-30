import os
import sys
import cv2
import numpy as np
import logging as log
from openvino.inference_engine import IECore
'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
'''


class LandmarkDetector:
    '''
    Class for the Landmarks Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        ''' Use this to set your instance variables. '''
        self.core = None
        self.network = None
        self.input_blob = None
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

        # retrieve name of model's input layer
        self.input_blob = next(iter(self.network.input_info))

        # retrieve name of model's output layer
        self.output_blob = next(iter(self.network.outputs))

        ### load IENetwork to Executable Network ###
        ### Note: You may need to update the function parameters. ###
        self.exec_network = self.core.load_network(network=self.network,
                                                   device_name=self.device,
                                                   num_requests=1)

    def predict(self, image):
        '''
        Perform Inference on Image and return Raw Output.

        This method is meant for running predictions on the input image.
        Parameters:
            image (numpy.ndarray): Face image
        
        Returns:
            model_output (numpy.ndarray): Raw Model Output
        '''
        ### PreProcess input image according to model Requirement
        input_img = self.preprocess_input(image)
        ### run inference and return output
        # Start Async Inference Request
        infer_request_handle = self.exec_network.start_async(
            request_id=0, inputs={self.input_blob: input_img})

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

    def preprocess_output(self, outputs, image):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''

        x_left_eye, y_left_eye = outputs[0][0][0][0], outputs[0][1][0][0]
        x_right_eye, y_right_eye = outputs[0][2][0][0], outputs[0][3][0][0]

        left_eye = self.crop_eyes(x_left_eye, y_left_eye, image)
        right_eye = self.crop_eyes(x_right_eye, y_right_eye, image)
        # print(left_eye.shape, right_eye.shape)
        # cv2.imwrite("left.png", left_eye)
        # cv2.imwrite("right.png", right_eye)
        return left_eye, right_eye

    def crop_eyes(self, x_axis, y_axis, image):
        w, h = image.shape[1], image.shape[0]
        x_min = int(x_axis * w) - 30
        y_min = int(y_axis * h) - 30
        x_max = int(x_axis * w) + 30
        y_max = int(y_axis * h) + 30

        cropped_eye = image[y_min:y_max, x_min:x_max]
        return cropped_eye

    def get_input_shape(self):
        """ Return the shape of the input layer """
        return self.network.inputs[self.input_blob].shape