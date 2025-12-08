import numpy as np
import openvino as ov
import math

from lib.CVModel import CVModel


class GazeEstimator(CVModel):
    """
    Class for the Gaze Estimation Model.
    """

    def predict(self, left_eye, right_eye, head_pose):
        """
        Perform Inference on Image and return Raw Output.

        This method is meant for running predictions on the input image.
        Parameters:
            image (numpy.ndarray): Frame from input file

        Returns:
            model_output (numpy.ndarray): Raw Model Output
        """
        ### PreProcess input image according to model Requirement
        left_eye = self.preprocess_input(left_eye)
        right_eye = self.preprocess_input(right_eye)
        left_eye = ov.Tensor(array=left_eye, shared_memory=False)
        right_eye = ov.Tensor(array=right_eye, shared_memory=False)
        ### run inference and return output
        # Start Async Inference Request
        poses = np.array(
            [
                [
                    head_pose["yaw"],  # Estimated Head yaw (in degrees)
                    head_pose["pitch"],  # Estimated Head pitch (in degrees)
                    head_pose["role"],  # Estimated Head roll (in degrees)
                ]
            ],
            dtype=np.float32,
        )
        poses = ov.Tensor(array=poses, shared_memory=False)

        self.infer_request.set_input_tensors(
            {
                # image of left eye
                0: left_eye,
                # image of right eye
                1: right_eye,
                # head pose angles
                2: poses,
            }
        )
        # run inference
        self.infer_request.start_async()
        self.infer_request.wait()
        # Get output tensor for model with one output
        output = self.infer_request.get_output_tensor()
        output_buffer = output.data
        return output_buffer

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
            angle_r_fc = hpe_cords["role"]
            sin_r = math.sin(angle_r_fc * math.pi / 180.0)
            cos_r = math.cos(angle_r_fc * math.pi / 180.0)
            x = gaze_vector[0] * cos_r + gaze_vector[1] * sin_r
            y = -gaze_vector[0] * sin_r + gaze_vector[1] * cos_r
            mouse_cord = (x, y)
        except Exception as e:
            print("Error While preprocessing output in Gaze Estimation Model" + str(e))
        return mouse_cord
