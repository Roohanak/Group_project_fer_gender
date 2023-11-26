# KNN-Based Facial Emotion Recognition System

## Introduction
This Python script employs the K-Nearest Neighbors (KNN) algorithm for facial emotion recognition. It processes images, applies gray scaling, resizes them, and uses KNN for emotion classification. The script supports various functionalities, including image preprocessing, KNN model testing with different parameters, and multithreading for improved performance.

## Features
- Image processing (gray scaling and resizing)
- Emotion classification using KNN
- Testing KNN with different k-values and metrics
- Multithreading for accelerated computing
- Graph generation for model accuracy
- Option to predict emotions from user-provided images

## Installation
Before running the script, install the required packages:
```bash
pip install opencv-python
pip install scikit-learn
pip install matplotlib
```

## Usage:

### Prepare the Data
Place your training and validation image datasets in respective folders. Update training_file_path and validation_file_path in the script with the paths to these folders.

### Run the Script
Execute the script. It will automatically process the images, train the KNN model, and test it with different parameters.

###  View Results 
After execution, accuracy graphs are saved in the specified output directory. The script prints details about the best accuracies achieved with different metrics.

## Script Structure
### Image Processing: 
The get_images_from_folder function is responsible for loading and processing images from a directory.

### KNN Model Training and Testing: 
The test_k_values and test_metric functions are used to test the KNN model with various K values and distance metrics.

### Graph Generation: 
The get_graphed_result_w_test_size function generates graphs depicting model performance.


### Multithreading
The script includes commented-out sections for multithreading. If your system can handle it, uncomment these sections for faster processing. Be cautious, as improper use may affect system performance.

## Note
The script assumes a specific directory structure for image data. Ensure your data is organized accordingly.
Multithreading is optional and should be used based on system capabilities.
References
Various resources were consulted during the development of this script, including documentation on Scikit-Learn's KNN classifier and studies on facial emotion recognition using machine learning.
