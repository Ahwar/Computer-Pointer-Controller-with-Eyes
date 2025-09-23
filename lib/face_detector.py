import cv2
import numpy as np
import openvino as ov

"""
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
"""


class FaceDetector:
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
            image (numpy.ndarray): Frame from input file

        Returns:
            model_output (numpy.ndarray): Raw Model Output
        """

        ### PreProcess input image according to model Requirement
        input_img = self.preprocess_input(image)
        input_tensor = ov.Tensor(input_img, shared_memory=False)
        ### run inference and return output
        # Start Async Inference Request
        self.infer_request.set_input_tensor(input_tensor)
        self.infer_request.start_async()
        self.infer_request.wait()

        model_output = self.infer_request.get_output_tensor().data

        return model_output

    def preprocess_input(self, image):
        """
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        """
        (b, c, h, w) = self.get_input_shape()
        image = cv2.resize(image, (w, h))
        image = np.transpose(image, (2, 0, 1))
        image = image.reshape(b, c, h, w)
        image = image.astype(np.float32)
        image = image / 255.0

        return image

    def postprocess_output(self, outputs, threshold, image, image_w, image_h):
        """
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        """
        for out in outputs[0][0]:
            conf = out[2]
            maxi = 0
            det = None
            cropped_face = None
            if conf > threshold and conf > maxi:
                det = out
                maxi = conf
            if det is not None:
                (x_min, y_min) = (int(det[3] * image_w), int(det[4] * image_h))
                (x_max, y_max) = (int(det[5] * image_w), int(det[6] * image_h))
                cropped_face = image[y_min:y_max, x_min:x_max]
            return cropped_face, ((x_min, y_min), (x_max, y_max))

    def get_input_shape(self):
        """Return the shape of the input layer"""
        return self.compiled_model.inputs[0].shape


if __name__ == "__main__":
    detector = FaceDetector(r"bin\models\1\face-detection-retail-0004.xml")
    detector.load_model()
    image = cv2.imread("bin/face.png")
    print("Model Input shape:", detector.get_input_shape())
    print("image size", image.shape)
    output = detector.predict(image)
    print("Model Output:", output.shape)
