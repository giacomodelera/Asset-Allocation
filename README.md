# Asset Allocation

# AI for Blood Cell Identification

## Abstract
Blood cell identification is critical in medical diagnostics, particularly for understanding diseases affecting the blood.

This project focuses on the classification of different types of blood cell images into eight distinct categories using neural networks and deep learning techniques. 
The primary objective is to develop a model capable of achieving high accuracy in this task using image data.

The project began with the construction by hand of a Convolutional Neural Network (CNN), which provided baseline results. To improve performance, it has been decided to use tranfer learning techniques using pre-trained models and it has been found that the EfficientNet family was to best one to employ, with Efficient-NetV2S[3], towards techniques of data augmentation, achieving the best performance.

Additionally, other dataset augmentation techniques were applied using keras-cv to enhance the training dataset and further improve model robustness. 

In the end, also test time augmentation techinques were employed to improve the accuracy.

## Context
This project is an AI-based solution developed for blood cell identification as part of a competition. The goal was to accurately classify various types of blood cells from images. 
The project utilizes deep learning models and image processing to achieve high classification accuracy. 

We are proud to have ranked in the top 5 of the competition, demonstrating the effectiveness of the solution.

## Repository Structure
* [Report](Report.pdf)
  contains a deeper description of the problem and a detailed explanation of the methodologies.
* **`Code/`**
  
  **`1_dataset_augmentation`** contains code for augmenting the dataset by applying various transformations (e.g., rotation, flipping, scaling) to the images.
  
  **`2_model_training`** is responsible for training the AI model. It loads the augmented dataset, sets up the deep learning architecture, and trains the model using the appropriate loss function and optimization     technique.
  
  **`3_zip_for_submission`** is used to format the model and weights in order to easily submit results to the competition platform.


