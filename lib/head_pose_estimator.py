import cv2
import numpy as np
import openvino as ov

"""
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
"""


class HeadposeEstimator:
    """
    Class for the Face Detection Model.
    """

    def __init__(self, model_name, device="CPU"):
        """Use this to set your instance variables."""
        self.core = None
        self.compiled_model = None
        self.infer_request = None
        self.device = device
        self.model_xml = model_name

    def load_model(self):
        """
        Load Model file and create Executable Network

        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins (e.g. CPU extensions), this is where you can load them.
        Initialize Core
        Read model in to IENetwork and add any necessary Extension
        Check for Supported Layers
        """
        ### Load the model ###
        # Take model .xml and .bin file and create IENetwork
        self.core = ov.Core()
        self.compiled_model = self.core.compile_model(self.model_xml, "AUTO")

        self.infer_request = self.compiled_model.create_infer_request()

    def predict(self, image):
        """
        Perform Inference on Image and return Raw Output.

        This method is meant for running predictions on the input image.
        Parameters:
            image (numpy.ndarray): Image of Detected Face

        Returns:
            model_output (numpy.ndarray): Raw Model Output
        """
        model_output = {}
        ### PreProcess input image according to model Requirement
        input_img = self.preprocess_input(image)
        input_tensor = ov.Tensor(input_img, shared_memory=False)
        ### run inference and return output
        # Start Async Inference Request
        self.infer_request.set_input_tensor(input_tensor)
        self.infer_request.start_async()
        self.infer_request.wait()

        model_output["yaw"] = self.infer_request.get_output_tensor(0).data[0][0]
        model_output["pitch"] = self.infer_request.get_output_tensor(1).data[0][0]
        model_output["role"] = self.infer_request.get_output_tensor(2).data[0][0]
        return model_output

    def preprocess_input(self, image):
        """
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        """
        (b, c, h, w) = self.get_input_shape()
        image = cv2.resize(image, (w, h))
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)

        return image

    def get_input_shape(self):
        """Return the shape of the input layer"""
        return self.compiled_model.inputs[0].shape
