Korean Character Recognition
==============================
## Overview
This project is building a model to recognize 2,350 handwritten hangul labels using VGG-19 CNN model. We use transfer learning method to save times for the training.
Over 2,000,000 of handwritten characters were used to train the model.

## Getting Started
* Install the python libraries. (See requirements.txt)
	pip install -r requirements.txt
* Download the dateset
	You can download the dataset with permission from the EIRIC.
	Link: [EIRIC](https://www.eiric.or.kr/special/special.php#)

## Workflow
1. Preprocess the images
The images are variant in resolutions while the model needs a consistent input size, 224 × 224. We need to remove the noises in the images and crop the images into 224 x 224.
* Removing noises with median filters
* Cropping images

<img src='pics/rawimages.png' width=320> <img src='pics/preprocessed.png' width=320>
Before						After


