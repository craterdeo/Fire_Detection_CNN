# Fire_Detection_CNN
Implementation of CNN algorithm (Deep Learning) for Image Detection.

This Fire Detection Repository Covers:
  - Getting a Dataset
  - Setting Up the Directory
  - Setting up the Coding Environment (IDE)
  - Implementing the Convolutional Neural Network (CNN) Algorithm
  - Analysing the Output 

## 1) Getting a Dataset
This code is functional on a dataset with two classes or class labels of images. Any dataset with fire and non fire images will be suitable to train the model for fire detection.
Depending on the application and the need/interest , the dataset can either be self developed or an already existing dataset.
### 1.1)Creating Your Own Dataset:
A dataset for image detection needs to have a decent amount of images so that the model is able to identify different scenarios and conditions to be able to find the required thing in the image, may it be any kind of image.

Thus while creating the dataset, it is necessary to take pictures(take yourself) or get pictures that have the object in the image to be detected (like fire for fire detection) as well as making sure that the pictures include different kinds of scenarios so that the model doesnt get wrongly trained.For Example, say the fire dataset is being trained . If the images in the dataset are containing only fire and not the surrounding area or other details, it is possible that the system gets trained to detect fire only using colour which may cause it to detect normal sunset images also as fire images.

Next, it is necessary to have images that show the object and images that don't so that the dataset is able to classify the images while testing into two different categories.

### 1.2) Using an existing Dataset
Datasets for general purposes like fire detection or vehicle detection are already available as these datasets have been used for training existing models. Using these datasets prevents the need of looking for correct images and classifying them. Further, using these saves a lot of time as creating a dataset is a long and tedious process. 
However, existing datasets might not be able to satisfy the requirement of the project/model that is newly being implemented or is using a different approach. 

Such existing datasest can be found on the internet with sufficiet amount of searching and filtering. 

The dataset used in this Fire Detection implementation can be found [here](https://www.kaggle.com/datasets/christofel04/fire-detection-dataset).

## 2) Setting Up the Directory
While writing the code , it will be required to load the correct dataset files and directories into the code so that the model can be trained. Thus, keeping all the required resources in one single directory makes the coding easier. 
To set up a directory, navigate to the folder on the system where you want to save the files and code and create a folder of any name (say `Fire_Detection`).The dataset is to be stored in this directory .

## 3)Setting Up the Coding Environment (IDE)
In this implementation, we will be using VS Code IDE to write our codes and implement the image detection.

### Installing VS Code
To install Visual Dtudio Code, you can directly type in *install vs code* on google and install the latest version. 
Here, you can find the tutorial to install VS code and setting up python extension for [Windows](https://www.youtube.com/watch?v=MlIzFUI1QGA) and [Linux](https://code.visualstudio.com/docs/setup/linux).


### Setting Up Jupyter Notebook Extension
To install the jupyter notebook extension, go to the extensions icon in vscode and search *jupyter*. Install the first extension Jupyter by microsoft. 

![image](https://user-images.githubusercontent.com/81915404/162585795-ae35acb2-39c8-4495-8c27-d5ddf4d56cb9.png)

Once jupyter noteebook has been installed , go to the dirctory Fire_Detection and open VS code in that directory. This can be done by going to open in file tab in vscode and typin path of the directory **OR** by going t that directoy, right clicking and choosing the option **open with code**.

For Linux, the following command can be run to open code in the specific directory:
    
    cd <path to the directory>
    code .

So , say my directory is stored in documents an =d is named Fire_Detection. Then, the command will be:

    cd /Documents/Fire_Detection
    code .

After doing this, the open directory in the vs code window will show an empty directory named Fire_Detection.

![image](https://user-images.githubusercontent.com/81915404/162586381-cc4d9315-fe44-45c4-85f7-2081e2e904a7.png)

## Downloading the Required Files:


