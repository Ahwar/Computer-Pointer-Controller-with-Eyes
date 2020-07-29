# Computer Pointer Controller

*TODO:* Write a short introduction to your project

## Project Set Up and Installation
*TODO:* Explain the setup procedures to run your project. For instance, this can include your project directory structure, the models you need to download and where to place them etc. Also include details about how to install the dependencies your project requires.
### Requirements

#### Hardware

* 6th to 10th generation Intel® Core™ processor with Iris® Pro graphics or Intel® HD Graphics.
* OR use of Intel® Neural Compute Stick 2 (NCS2)

#### Software

*   Intel® Distribution of OpenVINO™ toolkit 2020.4 release
*   Python Version 3.5 or 3.6

#### Python Dependencies

Python dependencies are defined in `requirement.txt` file on home directory.  

### Installations

#### OpenVino Toolkit Version 2020.4
To Install the OpenVino Toolkit Version 2020.4 refer to this [OpenVino Toolkit Installation Guide](https://docs.openvinotoolkit.org/2020.4/openvino_docs_install_guides_installing_openvino_linux.html#install-openvino)  
  
#### Python 3.5 or 3.6

> Note: This version is only supported by Python 3.5 or 3.6.

#### Virtual Environment
Installing and activating virtual environment will isolate your packages from the whole system.   

To Install **Python 3.5 or 3.6 Virtual Environment** refer to this [Virtual Environment Installation Guide](https://thepythonguru.com/python-virtualenv-guide/)  

#### Python dependencies
To install them first activate your virtual environment and then go to project home directory and run below command  
```
pip3 install -r requirements.txt
```  

### Download OpenVino models.
For this project we need these list of models.  
*  [face-detection-adas-binary-0001](https://docs.openvinotoolkit.org/2020.4/omz_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html).  

### Download the model
Below command is used to download models files.  
first install pre requisites:  
```
python3 -mpip install --user -r /[openvino/installation/directory]/openvino/deployment_tools/tools/model_downloader/requirements.in
```  

Then download the models files with below command.  
```
python3 /[openvino/installation/directory]/openvino/deployment_tools/tools/model_downloader/downloader.py --name [model_name]  --output_dir my/download/directory --precisions [precision]
```  
In above command replace [model_name] with the name of the model from above models list.  
In the above command replace [precision] with the model precision you want to use.
-  For running model on CPU use precisions as FP32.
-  For running model on MYRIAD VPU or IGPU use precisions as FP16.
## Demo
### Run the code

Open a new terminal to run the code. 

#### Setup the environment

You must configure the environment to use the Intel® Distribution of OpenVINO™ toolkit one time per session by running the following command:
```
source /opt/intel/openvino/bin/setupvars.sh -pyver 3.5
```

You should also be able to run the application with Python 3.6, although newer versions of Python will not work with the app.

#### Running the program

Though by default application runs on CPU, this can also be explicitly specified by ```-d CPU``` command-line argument:
run below command to run application.
```
python src/main.py -i bin/demo.mp4 -m your-face-detection-model.xml -d CPU -pt 0.6
```

To Understand all command line argument see below help.
> To change the device you want your model to run see description of `-d` argument below. 

```
usage: main.py [-h] -m MODEL -i INPUT [-l CPU_EXTENSION] [-d DEVICE]
               [-pt PROB_THRESHOLD] 

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Path to an xml file with a trained model.
  -i INPUT, --input INPUT
                        Path to image or video file
  -l CPU_EXTENSION, --cpu_extension CPU_EXTENSION
                        MKLDNN (CPU)-targeted custom layers.Absolute path to a
                        shared library with thekernels impl.
  -d DEVICE, --device DEVICE
                        Specify the target device to infer on: CPU, GPU, FPGA
                        or MYRIAD is acceptable. Sample will look for a
                        suitable plugin for device specified (CPU by default)
  -pt PROB_THRESHOLD, --prob_threshold PROB_THRESHOLD
                        Probability threshold for detections filtering(0.5 by
                        default)
```

##### Using a camera stream instead of a video file

To get the input video from the camera, use the `-i CAM` command-line argument.  

##### Running on the Intel® Neural Compute Stick

To run on the Intel® Neural Compute Stick, use the ```-d MYRIAD``` command-line argument.
> computation on MYRIAD Intel NCS requires model of precision `FP16`. for downloading model with that precision refere to model downloading section

##### Running on the Integrated Graphical Processing Unit (IGPU)

To run on the IGPU, use the ```-d GPU``` command-line argument.
> computation on IGPU requires model of precision `FP16`. for downloading model with that precision refere to model downloading section
## Documentation
*TODO:* Include any documentation that users might need to better understand your project code. For instance, this is a good place to explain the command line arguments that your project supports.

## Benchmarks
*TODO:* Include the benchmark results of running your model on multiple hardwares and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc.

## Results
*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.

## Stand Out Suggestions
This is where you can provide information about the stand out suggestions that you have attempted.

### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.

### Edge Cases
There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project and how you solved them to make your project more robust.
