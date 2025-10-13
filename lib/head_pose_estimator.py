import cv2
import numpy as np
import openvino as ov

from lib.CVModel import CVModel

"""
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
"""


class HeadposeEstimator(CVModel):
    """
    Class for the Face Detection Model.
    """

    

    def predict(self, image):
        """
        Perform Inference on Image and return Raw Output.

        This method is meant for running predictions on the input image.
        Parameters:
            image (numpy.ndarray): Image of Detected Face

        Returns:
            model_output (numpy.ndarray): Raw Model Output
        """
        super().predict(image)
        model_output = {}
        model_output["yaw"] = self.infer_request.get_output_tensor(0).data[0][0]
        model_output["pitch"] = self.infer_request.get_output_tensor(1).data[0][0]
        model_output["role"] = self.infer_request.get_output_tensor(2).data[0][0]
        return model_output

