# Group_project_fer_gender
Overview
This README provides instructions on how to use the provided Python code to classify gender using three different machine learning models: Convolutional Neural Networks (CNN), Support Vector Machines (SVM), and K-Nearest Neighbors (KNN). The code is designed to run on Google Colab with GPU support.

Pre-requisites
Google Colab account.

Dataset
We use the CelebAMask-HQ dataset for gender classification. Before running the code, you need to download the dataset and upload it to Google Drive.

Dataset setup
Visit the dataset URL: CelebAMask-HQ Dataset.
Download the dataset to your local machine.
Unzip the dataset.
Upload Dataset to Google Drive
Log in to your Google Drive account.
Create a new folder, e.g., gender_classification_dataset.
Upload the unzipped dataset to this folder.

Setting Up Google Colab
Open Google Colab: Google Colab.
Sign in with your Google account.
Create a new notebook.
GPU Runtime Setup
To ensure faster computation, enable GPU in your Colab notebook:
In your Colab notebook, go to Runtime > Change runtime type.
Under Hardware accelerator, select GPU.
Click Save.

Mount Google Drive in Colab
To access the dataset from Google Drive, mount it in Colab:
from google.colab import drive
drive.mount('/content/drive')
Follow the on-screen instructions to authorize Colab to access your Google Drive.

Running the Code
Set the dataset path in the code to point to the location where you uploaded the dataset in Google Drive, e.g., /content/drive/My Drive/gender_classification_dataset.
Copy the Python code into a new cell in your Google Colab notebook.
Run the cell to execute the code.
The code will automatically train the models and display the results including accuracy, training time, and memory usage for each model.

Conclusion
By following these instructions, you can easily run the code to compare the performance of CNN, SVM, and KNN models in gender classification using the CelebAMask-HQ dataset on Google Colab with GPU support.
