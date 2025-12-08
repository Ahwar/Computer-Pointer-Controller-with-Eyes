import cv2
import openvino as ov

from lib.CVModel import CVModel

"""
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
"""


class FaceDetector(CVModel):

    def predict(self, image):
        super().predict(image=image)

        model_output = self.infer_request.get_output_tensor().data
        return model_output

    def postprocess_output(self, outputs, threshold, image, image_w, image_h):
        """
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        """
        maxi = 0
        det = None
        cropped_face = None
        for out in outputs[0][0]:
            conf = float(out[2])
            if conf > threshold and conf > maxi:
                det = out
                maxi = conf
        if det is not None:
            (x_min, y_min) = (int(det[3] * image_w), int(det[4] * image_h))
            (x_max, y_max) = (int(det[5] * image_w), int(det[6] * image_h))
            cropped_face = image[y_min:y_max, x_min:x_max]
            return cropped_face, ((x_min, y_min), (x_max, y_max))
        return None, None

    def get_input_shape(self):
        """Return the shape of the input layer"""
        return self.compiled_model.inputs[0].shape


if __name__ == "__main__":
    detector = FaceDetector("bin/models/face-detection-retail-0004.xml")
    detector.load_model()
    image = cv2.imread("bin/face.png")
    print("Model Input shape:", detector.get_input_shape())
    print("image size", image.shape)
    output = detector.predict(image)
    print("Model Output:", output.shape)
    cropped_face, face_coords = detector.postprocess_output(
        output, 0.1, image, image.shape[1], image.shape[0]
    )
    if cropped_face is not None:
        cv2.imshow("t", cropped_face)
        cv2.waitKey()
    else:
        print("No face founded")
