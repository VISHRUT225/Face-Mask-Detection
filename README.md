# Face-Mask-Detection
This repository contains code for face detection using YOLOv5. The project is built on the PyTorch deep learning framework and uses the YOLOv5 object detection algorithm for detecting faces.

<center><img align="center" src="./Detected Images/mask_gif.gif" alt="1" style="width: 300px; height: auto;"></center>

## 📑 Tech Stack/Frameworks

- Python
- YOLOv5
- PyTorch
- OpenCV
- NumPy

## :key: Prerequisites
1. A Google account: You'll need a Google account to access Google Colab.

2. Dataset: You'll need a face mask dataset with bounding box annotations. You can use the face-mask-detection dataset from Roboflow, which contains annotated images of people wearing or not wearing masks. You can import the dataset into Google Colab using the Roboflow API or by uploading it to your Google Drive.

3. YOLOv5 code: You'll need the YOLOv5 code to train and test the face detection model which is availabe to below.

 All the dependencies and required libraries are included in the file <code>requirements.txt</code>

## 🚀 Getting Started
1. Clone Repo and install all dependencies
```
!git clone https://github.com/ultralytics/yolov5
!pip install -qr yolov5/requirements.txt
%cd yolov5

import torch
from IPython.display import Image, clear_output


clear_output()
```

2. Download the custom data-set that you are interested in (using the Roboflow api) Link to datasets 
```
%cd /content
!curl -L "Enter your API key/url" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip
```

3. Create the custom model configuration file
```
#extracting information from the roboflow file
%cat data.yaml

# define number of classes based on data.yaml
import yaml
with open("data.yaml", 'r') as stream:
    num_classes = str(yaml.safe_load(stream)['nc'])

```

4. Download pre-trained weights
```
!wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt
```

5. Train yolov5 on the custom images using the custom configuration file
```
# train yolov5s on custom data for 100 epochs
# time its performance
%%time
%cd /content/yolov5/
!python train.py --img 416 --batch 16 --epochs 100 --data '../data.yaml' --weights /content/yolov5/yolov5n.pt --cache
```
6. Inference on trained model
```
from utils.plots import plot_results
plot_results('/content/yolov5/runs/train/exp/results.csv')  # plot 'results.csv' as 'results.png'

# display plot results
from IPython.display import Image
Image(filename='/content/yolov5/runs/train/exp/results.png')
```



## 🔎 Detection
1. Run yolov5 detection on images.
```
#Don't forget to copy the location of the weights file and replace it in the code below
!python detect.py --weights /content/yolov5/runs/train/exp/weights/best.pt --img 416 --conf 0.4 --source ../test/images
```

2. Show mask-detected images
```
import glob
from IPython.display import Image, display

for imageName in glob.glob('/content/yolov5/runs/detect/exp2/*.jpg'):
    display(Image(filename=imageName))
    print("\n")
```

## 📷 Detected Images

<img src="./Detected Images/1.jpg" alt="1" style="width: 300px; height: auto;"> &nbsp;<img src="./Detected Images/2.jpg" alt="1" style="width: 300px; height: auto;"> &nbsp; <img src="./Detected Images/5.jpg" alt="1" style="width: 300px; height: auto;"> &nbsp;<img src="./Detected Images/3.jpg" alt="1" style="width: 300px; height: auto;"> &nbsp;<img src="./Detected Images/4.jpg" alt="1" style="width: 300px; height: auto;"> &nbsp;


## 👍🏻 Acknowledgements
The YOLOv5 implementation used in this project was adapted from the Ultralytics YOLOv5 repository. Thanks to the authors for making their code available under an open source license.
