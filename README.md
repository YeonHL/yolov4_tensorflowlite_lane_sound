# Yolov4_Lane_Tensorflowlite_Alarm



## 1. Install



### 1) Conda Environment (Recommended)

To get started, install the proper dependencies either via Anaconda or Pip. I recommend Anaconda route for people using a GPU as it configures CUDA toolkit version for you.

```python
# Tensorflow CPU
conda env create -f conda-cpu.yml
conda activate yolov4-cpu

# Tensorflow GPU
conda env create -f conda-gpu.yml
conda activate yolov4-gpu
```



### 2) Pip & CUDA

(TensorFlow 2 packages require a pip version >19.0.)

```python
# TensorFlow CPU
pip install -r requirements.txt

# TensorFlow GPU
pip install -r requirements-gpu.txt
```



#### Nvidia CUDA

Make sure to use CUDA Toolkit version 10.1 as it is the proper version for the TensorFlow version used in this repository. [CUDA Toolkit 10.1 update2 Archive | NVIDIA Developer](https://developer.nvidia.com/cuda-10.1-download-archive-update2)



## 2. Download Official YOLOv4 Pre-trained Weights

For easy demo purposes we will use the pre-trained weights for our tracker.

https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights



Copy and paste yolov4-tiny.weights from your downloads folder into the 'data' folder of this repository.



## 3. Convert to tflite

To implement the object tracking using YOLOv4-tiny, first we convert the .weights into the corresponding TensorFlow model which will be saved to a checkpoints folder. Then we convert tf model to tflite model. All we need to do is run the object_tracker.py script to run our object tracker with YOLOv4-tiny, DeepSort and TensorFlowlite.

```python
# Save tf model for tflite converting
python save_model.py --weights ./data/custom-yolov4-tiny-detector_best.weights --output ./checkpoints/yolov4-tiny-416 --input_size 416 --model yolov4 --tiny --framework tflite

# Convert tf model to tflite model
python convert_tflite.py --weights ./checkpoints/yolov4-tiny-416 --output ./checkpoints/yolov4-tiny-416.tflite

```



### 4. Running the Tracker

Video file for test

https://drive.google.com/file/d/1X84PFatOcdlVXvRG0ngw8oAZKxj3GsEk/view?usp=sharing

```python
# Run yolov4-tiny object tracker
python object_tracker.py --weights ./checkpoints/yolov4-tiny-416.tflite --model yolov4 --video ./data/video/test_sample.mp4 --output ./outputs/.avi --tiny --info
```



## References



[GitHub - nicedaddy/yolov4_deepsort_lane_detection](https://github.com/nicedaddy/yolov4_deepsort_lane_detection)

[GitHub - haroonshakeel/tensorflow-yolov4-tflite](https://github.com/haroonshakeel/tensorflow-yolov4-tflite)




