import cv2
import numpy as np
import openvino as ov

from lib.CVModel import CVModel

"""
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
"""


class LandmarkDetector(CVModel):
    """
    Class for the Landmarks Detection Model.
    """

    def predict(self, image):
        super().predict(image)

        model_output = self.infer_request.get_output_tensor().data

        return model_output

    def postprocess_output(self, outputs, image):
        """
        Before feeding the output of this model to the next model,
        you might have to postprocess the output. This function is where you can do that.
        """

        x_left_eye, y_left_eye = outputs[0][0][0][0], outputs[0][1][0][0]
        x_right_eye, y_right_eye = outputs[0][2][0][0], outputs[0][3][0][0]
        # make cropped eye and its coordinates
        left_eye, left_coords = self.crop_eyes(x_left_eye, y_left_eye, image)
        right_eye, right_coords = self.crop_eyes(x_right_eye, y_right_eye, image)
        return left_eye, left_coords, right_eye, right_coords

    def crop_eyes(self, x_axis, y_axis, image):
        w, h = image.shape[1], image.shape[0]
        x_min = int(x_axis * w) - 30
        y_min = int(y_axis * h) - 30
        x_max = int(x_axis * w) + 30
        y_max = int(y_axis * h) + 30

        cropped_eye = image[y_min:y_max, x_min:x_max]
        return cropped_eye, ((x_min, y_min), (x_max, y_max))
