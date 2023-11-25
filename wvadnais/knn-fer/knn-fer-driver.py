#################################################################################################################
# imports
#################################################################################################################
# pip install opencv-python
# pip install scikit-learn
# pip install matplotlib
import numpy as np
import cv2
from sklearn.neighbors import KNeighborsClassifier, kneighbors_graph
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time
import threading
from concurrent.futures import ThreadPoolExecutor

# NOTE: multithreading is being used to accelerate computing,
# if you are facing trouble please switch to single threading, see provided comments below.

# filePath/to/train-folder
training_file_path = 'C:/Users/William/PycharmProjects/Group_project_fer_gender/wvadnais/knn-fer/knn-fer-images/images/train'

# filePath/to/validation-folder
validation_file_path = 'C:/Users/William/PycharmProjects/Group_project_fer_gender/wvadnais/knn-fer/knn-fer-images/images/validation'


#################################################################################################################
# setting up methods
#################################################################################################################

# load images from a directory and convert them into a usable format
# -Apply grey scale
# -Resize
# -Get label
def get_images_from_folder(base_folder):
    images = []
    labels = []
    for emotion_label in os.listdir(base_folder):
        emotion_folder = os.path.join(base_folder, emotion_label)
        if os.path.isdir(emotion_folder):
            for filename in os.listdir(emotion_folder):
                img = cv2.imread(os.path.join(emotion_folder, filename), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (48, 48))  # Resize images to 48x48
                    images.append(img.flatten())
                    labels.append(emotion_label)  # Use folder name as label
    return np.array(images), np.array(labels)


# testing different k values
# x=images,y=labels, metric= distance metric used in knn
def test_k_values(X_training, y_training, X_test, y_test, metric):
    # testing different k values
    # can set k up to one less than the number of samples in the training set but wouldnt recommend
    k_values = [1, 3, 7, 10, 20, 50, 100]
    accuracies = {}

    # # for single threading
    # for k in k_values:
    #     run(X_test, X_training, accuracies, k, metric, y_test, y_training)

    # for multithread processing, BE CAREFUL! dont apply multithread anywhere else unless you understand your system
    def run_thread(k):
        run(X_test, X_training, accuracies, k, metric, y_test, y_training)

    with ThreadPoolExecutor(max_workers=8) as executor:
        time.sleep(1)
        executor.map(run_thread, k_values)

    return accuracies


# just a helper method for the test_k_values method
# x=images,y=labels, metric= distance metric used in knn
def run(X_test, X_training, accuracies, k, metric, y_test, y_training):
    start_time = time.time()
    knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
    knn.fit(X_training, y_training)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies[k] = accuracy


# testing different metrics
# x=images,y=labels
def test_metric(X_training, y_training, X_test, y_test, size):
    metrics = ['nan_euclidean', 'manhattan', 'correlation', 'euclidean', 'chebyshev', 'braycurtis',
               'cosine', 'minkowski', 'sqeuclidean', 'canberra', 'hamming']

    best_accuracies = {}
    all_accuracies = {}

    # for single thread processing
    for metric in metrics:
        run_knn(X_test, X_training, all_accuracies, best_accuracies, metric, size, y_test, y_training)

    # # for multithread processing, BE CAREFUL! dont apply multithread anywhere else unless you understand your system
    # def run_knn_thread(metric):
    #     run_knn(X_test, X_training, all_accuracies, best_accuracies, metric, size, y_test, y_training)

    # with ThreadPoolExecutor(max_workers=2) as executor:
    #     time.sleep(2)
    #     executor.map(run_knn_thread, metrics)
    return best_accuracies, all_accuracies


# just a helper method for the test_metric method
# x=images,y=labels, metric= distance metric used in knn
def run_knn(X_test, X_training, all_accuracies, best_accuracies, metric, size, y_test, y_training):
    start_time = time.time()
    try:
        accuracies = test_k_values(X_training, y_training, X_test, y_test, metric)
        best_k = max(accuracies, key=accuracies.get)
        best_accuracy = accuracies[best_k]
        best_accuracies[metric] = (best_k, best_accuracy)
        all_accuracies[metric] = accuracies
        print(
            f'Best accuracy for metric {metric}: {best_accuracy * 100}% with k={best_k} for test size: {size} '
            f'taking {time.time() - start_time} seconds')
    except ValueError:
        print(f'Metric {metric} is not valid or not applicable')


# generate graph for a given size
# x=images,y=labels
def get_graphed_result_w_test_size(X_training, y_training, size):
    # splitting the dataset into the training set and test set
    # test_size = 0.01 means use 1% of the data, 0.1 use 10%, 1 use 100%
    # random_state = 0  to ensure reproducibility and reliability,
    #      w/o random_state default uses numpy.random and produce different results each time.
    X_training, X_test, y_training, y_test = train_test_split(X_training, y_training, test_size=size, random_state=0)

    # getting results..
    best_accuracies, all_accuracies = test_metric(X_training, y_training, X_test, y_test, size)

    # default wasnt giving enough color options.
    colors = list(mcolors.TABLEAU_COLORS)

    # graph the results
    # plt.ion()
    plt.figure(figsize=(15, 10))
    for i, metric in enumerate(all_accuracies.keys()):
        ks, accs = zip(*all_accuracies[metric].items())
        plt.plot(ks, accs, marker='o', linestyle='-', label=metric, color=colors[i % len(colors)])
    plt.title('KNN Accuracy vs. K Value for Different Metrics with test size: ' + str(size))
    plt.xlabel('K Value')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.draw()
    plt.savefig('knn-fer-images/output-graphs/graph_sample_sizing-' + str(size * 35000) + '.png')
    time.sleep(3)
    print('\n\n#############  STARTING NEXT TEST SIZE  #############\n\n')


#################################################################################################################
# main
#################################################################################################################

# get training data
# x=images,y=labels
X_training, y_training = get_images_from_folder(training_file_path)  # filePath/to/train-folder

# get validation data
# x=images,y=labels
X_validation, y_validation = get_images_from_folder(validation_file_path)  # filePath/to/validation-folder

# due to large sample extra decimals were added to improve performance
test_sizes = [0.00003, 0.00006, 0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.1, 0.5, 1]
for size in test_sizes:
    get_graphed_result_w_test_size(X_training, y_training, size)

#################################################################################################################
# Code to input your own picture against the model and see if it predicts the correct emotion
#################################################################################################################

# # predict emotion from an user provided image
# def predict_emotion(image_path, knn_model):
#     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     img = cv2.resize(img, (48, 48))
#     emotion = knn_model.predict([img.flatten()])
#     return emotion[0]
#
#
# # open file dialog then predict emotion
# def open_file_and_predict(knn_model):
#     root = tk.Tk()
#     root.withdraw()  # Hide the main window
#     file_path = filedialog.askopenfilename()  # Open the file dialog
#     if file_path:
#         print(f"Selected file: {file_path}")
#         emotion = predict_emotion(file_path, knn_model)
#         print(f"Predicted Emotion: {emotion}")
#
# # setup KNN model
# # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
# k = 20
# knn_model = KNeighborsClassifier(n_neighbors=k, metric='minkowski')
# knn_model.fit(X_training, y_training)
#
# open_file_and_predict(knn_model)

#################################################################################################################
# references
#################################################################################################################

# from Scikit:
##  - https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
##  - https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
#
# Viewed to understand fer2013 data set usage:
## - https://medium.com/@birdortyedi_23820/deep-learning-lab-episode-3-fer2013-c38f2e052280
#
# Viewed to understand more about KNN for FER:
## - https://medium.com/shecodeafrica/performing-face-recognition-using-knn-fe71d87ab619#:~:text=K%2DNearest%20Neighbors%20(KNN)%20is%20a%20powerful%20algorithm%20used,choice%20for%20face%20recognition%20tasks.
#
# Viewed to understand KNeighborsClassifier with FER:
## - https://github.com/codinghappiness-web/Face-recognition-with-knn/blob/main/liverec.py
#
