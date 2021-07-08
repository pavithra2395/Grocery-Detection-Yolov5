## Grocery_Product Detection using YOLOv5x
In this repository, I present an application of the latest version of YOLO i.e. YOLOv5x, to detect items present in a GroceryDataset//shelfImages. This application can be used to keep track of inventory of items simply using images of the items on shelf.

![Result image](result.jpg)

## Introduction
Object detection is a computer vision task that requires object(s) to be detected, localized and classified. In this task, first we need our machine learning model to tell if any object of interest is present in the image. If present, then draw a bounding box around the object(s) present in the image. In the end, the model must classify the object represented by the bounding box. This task requires Single shot object detection so that it can be implemented  in real-time. One of its major applications is its use in real-time object detection in self-driving vehicles.

Joseph Redmon, et al. originally designed YOLOv1, v2 and v3 models that perform real-time object detection. YOLO "You Only Look Once" is a state-of-the-art real-time deep learning algorithm used for object detection, localization and classification in images and videos. This algorithm is very fast, accurate and at the forefront of object detection based projects. 

Each of the versions of YOLO kept improving the previous in accuracy and performance. Then came YOLOv4 developed by another team, further adding to performance of model and finally the YOLOv5x model was introduced by Glenn Jocher in June 2020. This model significantly reduces the model size (YOLOv4 on Darknet had 244MB size where as YOLOv5 Extra-Large model is of 30MB). YOLOv5x also claims a faster accuracy and more frames per second than YOLOv4 as shown in graph below, taken from Roboflow.ai's website.


!![yolo_vs_detnet.png] Comparison of YOLOv5 vs EfficientDetNet

In this article, I will only focus on the use of YOLOv5x for Grocery item detection.

## Objective
To use YOLOv5x to draw bounding boxes over retail products in pictures using open_source_dataset.

![Original image](image)[Result image](result.png)

Grocery shelf image (on left) vs desired output with bounding box drawn on objects (right)

## Dataset
To do this task, first I downloaded the  Shelf image dataset from the following link: 
https://storage.googleapis.com/open_source_datasets/ShelfImages.tar.gz
The open_source_dataset is based on images of retail objects in a densely packed setting. It provides training and test set images and the corresponding .csv files which contain information for bounding box locations of all objects in those images. The .csv files have object bounding box information written in the following columns: 

image_name,x1,y1,x2,y2,class,image_width,image_height

where x1,y1 are top left co-ordinates of bounding box and x2,y2 are bottom right co-ordinates of bounding box, rest of parameters are self-explanatory. An example of parameters of C1_P12_N1_S4_1_JPG image for one bounding box, is shown below. There are several bounding boxes for each image, one box for each object.

C1_P12_N1_S4_1_JPG (Annotated.png)
In the Shelf image dataset, we have 283 images in the test set, 73 images. Each image can have varying number of objects, hence, varying number of bounding boxes.

## Methodology
Image annotated by Roboflow:
From the dataset, I took 283 images from the training set and went to Roboflow.ai website which provides online image annotation service in different formats including YOLOv5 supported format. The reason for picking only 283 images from training set is that the Roboflow.ai's image annotation service is free for the first 1000 images only.


### Automatic Annotation
On Roboflow.ai website, the bounding box annotation .csv file and images from training set are uploaded and Roboflow.ai's annotation service automatically draws bounding boxes on images using the annotations provided in the .csv files as shown in image above. 

### Data Generation
Roboflow also gives option to generate a dataset based on user defined split. I used 70–20–10 training-validation-test set split. After the data is generated on Roboflow, we get the original images as well as all bounding box locations for all annotated objects in a separate text file for each image, which is convenient.
Finally, we get a link to download the generated data with label files. This link contains a key that is restricted to only your account and is not supposed to be shared.

### Hardware Used
The model was trained on Google Colab Pro notebook with AMD Radson 4GB Graphics Card. Google Colab notebook can also be used which is free but usage session time is limited.

## Code
The code is present in colab notebook in attached files. However, it is recommended to copy the whole code in Google Colab notebook.

It is originally trained for COCO dataset but can be tweaked for custom tasks which is what I did. I started by cloning YOLOv5 and installing the dependencies mentioned in requirements.txt file. Also, the model is built for Pytorch, so I import that.

```
!git clone https://github.com/ultralytics/yolov5  # clone repo
I have Uploaded the model in to my google drive
!pip install -r /content/drive/MyDrive/YOLOV5/yolov5-master/yolov5-master/requirements.txt
```

Next, I download the dataset that I created at Roboflow.ai. The following code will download training, test and validation set and annotations too. It also creates a .yaml file which contains paths for training and validation set as well as what classes are present in our data. If you use Roboflow for data, then dont forget to enter the key in code as it is unique per user.

```
# Export code snippet and paste here
%cd /content
!curl -L "ADD THE KEY OBTAINED FROM ROBOFLOW" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip
```

This file tells the model the location path of training and validation set images alongwith the number of classes and the names of classes. For this task, number of classes is "11" and the name of class is "['0', '1', '10', '2', '3', '4', '5', '6', '7', '8', '9']" as we are only looking to predict bounding boxes. data.yaml file can be seen below:
![yaml](data.yaml)

### Network Architecture
Next let's define the network architecture for YOLOv5x. It is the same architecture used by the author Glenn Jocher for training on COCO dataset. I didnt change anything in the network. However, few tweaks were needed to change bounding box size, color and also to remove labels otherwise labels would jumble the image because of so many boxes. These tweaks were made in detect.py and utils.py file. The network is saved as custom_yolov5.yaml file.

```
I used coco128.yaml file for class names with number of classes .
```

## Training
Now I start the training process. I defined the image size (img) to be 640, batch size 8 and the model is run for 100 epochs. If we dont define weights, they are initialized randomly.

```
# train yolov5x on custom data for 100 epochs
%cd /content/drive/MyDrive/YOLOV5/yolov5-master/yolov5-master

!python train.py  --img 640 --batch 8 --epochs 100 --data coco128.yaml --weights /content/drive/MyDrive/YOLOV5/yolov5-master/yolov5-master/yolov5x.pt --nosave --cache
```
After the training is complete, model's weights are saved in train folder in exp(experiment)folder as last_yolov5_results.pt.

It took 0.639 hours for training to complete on a GPU provided by Google Colab .


## Observations
We can visualize important evaluation metrics after the model has been trained using the following code:

```
The following 3 parameters are commonly used for object detection tasks:
· GIoU is the Generalized Intersection over Union which tells how close to the ground truth our bounding box is.
· Objectness shows the probability that an object exists in an image. Here it is used as loss function.
· mAP is the mean Average Precision telling how correct are our bounding box predictions on average. It is area under curve of precision-recall curve.
It is seen that Generalized Intersection over Union (GIoU) loss and objectness loss decrease both for training and validation. Mean Average Precision (mAP) however is at 0.7 for bounding box IoU threshold of 0.5. Recall stands at 0.8 as shown below:


Now comes the part where we check how our model is doing on test set images using the following code:
```
# when we ran this, we saw 20.575s inference time.
I have modified the code in detect.py file to save the results, the results are stored in detect folder with name of exp(experiment).
!python detect.py --weights runs/train/exp3/weights/last.pt --conf 0.2 --source /content/drive/MyDrive/YOLOV5/yolov5-master/yolov5-master/dataset/images/test

## Results
Following images show the result of our YOLOv5x algorithm trained to draw bounding boxes on objects. The results are pretty good. 

![results](Detect_testdata_exp7)

Assignment Requriments:
mAp , precision , Recall Results are stored in Mtrics.json file.
Quetion & Answers Q&A file
Training results are stored in exp folder with source code with ipynb file.
Test results are stored in exp folder.


REMEMBER: The model that I have attached was only trained on 283 images for optimum results.

## Conclusion
Controversies aside, YOLOv5 performs well and can be customized to suit our needs. However, training the model can take significant GPU power and time. It is recommended to use atleast Google Colab with 16GB GPU or preferably a TPU to speed up the process for training the large dataset.

This retail object detector application can be used to keep track of store shelf inventory or for a smart store concept where people pick stuff and get automatically charged for it. YOLOv5'x Extra large weight size and good frame rate will pave its way to be first choice for embedded-system based real-time object detection tasks.
