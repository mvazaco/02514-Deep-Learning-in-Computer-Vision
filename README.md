# 02514 Deep Learning in Computer Vision

Set of the 3 projects carried out indicidually for the course _02514 Deep Learning in Computer Vision_ at Tehnical University of Denmark (DTU). Everything trained in the DTU High-Performance Cluster (HPC).

## Image Classification

In this project, we will recreate Jian-Yangs (from Silicon Valley on HBO) hotdog/not hotdog algorithm. The purpose of the algorithm is to classify images into two classes: hotdog or not hotdog. This algorithm classifies images into one of two categories, hence it is an instance of a Binary Classifier.
    
A dataset of images containing either hotdogs or something that is not a hotdog have been provided. The images come from the ImageNet categories: pets, furniture, people, food, frankfurter, chili-dog, hotdog. The dataset is already divided in train and test, where in total we can find around 3000 images.

![Image Classification Poster](https://github.com/mvazaco/02514-Deep-Learning-in-Computer-Vision/blob/main/ImageClassification/poster.jpg)

## Image Segmentation

Two different datasets for image segmentation:

- PH2: The skin lesion segmentation dataset.
- DRIVE: The retinal blood vessel segmentation dataset.

Our task is to design a generic segmentation architecture that you apply to both datasets, and perform a thorough validation of your algorithm, including multiple metrics for segmentation performance as well as an ablation study illustrating how different choices of parameters, loss functions, etc impact the performance of your network at your given task. Finally, we compare our architecture to the Segment Anything Model algorithm on a small test set.

![Image Segmentation Poster](https://github.com/mvazaco/02514-Deep-Learning-in-Computer-Vision/blob/main/ImageSegmentation/poster2.jpg)

## Object Detection

Our project addresses the critical issue of environmental litter by developing a deep learning object detection system. The system aims to automatically identify and locate trash in wild environments, enabling deployment in robotic machines for efficient cleanup of beaches, forests, and roads. The challenge lies in recognizing deformable, transparent, aged, fragmented, occluded, and camouflaged litter, necessitating a model that understands the diverse features of the natural world.

![Object Detection Poster](https://github.com/mvazaco/02514-Deep-Learning-in-Computer-Vision/blob/main/ObjectDetection/poster3.jpg)
