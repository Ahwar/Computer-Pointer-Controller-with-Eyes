"""
Just  a sample code that demonstrate inference using OpenVino
"""

import openvino as ov
import cv2
import numpy as np

## download models
# curl --create-dirs https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/face-detection-retail-0004/FP32/face-detection-retail-0004.bin -o bin/models/1/face-detection-retail-0004.bin
# curl --create-dirs https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/face-detection-retail-0004/FP32/face-detection-retail-0004.xml -o bin/models/1/face-detection-retail-0004.xml

core = ov.Core()

compiled_model = core.compile_model(
    "bin/models/1/face-detection-retail-0004.xml", "AUTO"
)

infer_request = compiled_model.create_infer_request()

print(infer_request.get_input_tensor())
print(infer_request.get_output_tensor())


## read image and preprocessing
# Read image
img = cv2.imread("bin/face.png")
# Resize image
img = cv2.resize(img, (300, 300))
# Convert to float32
img = img.astype(np.float32)
# Normalize pixel values to [0, 1]
img = img / 255.0
# Convert to numpy array
img = np.array(img)
img = img.reshape(3, 300, 300)
img = np.expand_dims(img, axis=0)


## Inference
# Create tensor from external memory
input_tensor = ov.Tensor(array=img, shared_memory=True)
# Set input tensor for model with one input
infer_request.set_input_tensor(input_tensor)
# run inference
infer_request.start_async()
infer_request.wait()


## output post processing
# get output
# Get output tensor for model with one output
output = infer_request.get_output_tensor()
output_buffer = output.data
print(output_buffer.shape)
