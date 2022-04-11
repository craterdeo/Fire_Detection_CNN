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

Next, we need to install the required dependencies (python libraries) to be able to implement our code. 
In this code, we are making use of Tensorflow Library . We will also use numpy to create arrays of images and os to refer our directory thorugh code.

To install these on VS code, go to the terminal and enter the following commands:
    
    pip install tensorflow 
    pip install numpy
    pip install os

## Downloading the Required Files:
The files block of the repository contains three files `fire_detection.ipynb` , `fire_detection.txt` and `fire_detection.py`.

To implement the algorithm, only the jupyter notebook is required i.e. `fire_detection.ipynb`. The other two can be used as references . The text file shows the complete training process steps and the python file contains the complete code as a single file though running it will not give the same output as vscode cannot process images in its terminal.

To set the directory up for implementation, download `fire_detection.ipynb` file in the `Fire_Detection` directory that was initially set up. Next, we need to set up the **Training , Testing** and **Validation** datasets. 

Splitting the dataset into these three sets is a very important task as it decides whether the model will be able to learn the pattern perfectly and be able to identify them or not. The dataset can be split in any way , some of which are shown below. Choosing the right validation dataset is necessary to prevent overfitting as well as underfitting of the model. To understand more about splitting the dataset, you can refer to [this](https://www.v7labs.com/blog/train-validation-test-set) website.

![image](https://user-images.githubusercontent.com/81915404/162608173-322a266b-d620-4a97-baf8-756081cf6ab3.png)

To split the dataset, create afolder in the `Fire_Detection` directry and name it `Dataset_Fire` or any other name as per your choice. Create three folders in this folder `Training` , `Testing` and `Validation`. In the training folder, add 70 - 75 % of your dataset images and add 10 - 15 % of them to the validation dataset. Try to put a high amount of varied images in the validation dataset for better fitting. Add the rest of the images in the testing dataset. 

In the `Trainng` and `Vaidation` folders, seperate the fire and non fire images into `fire` and `non_fire` folders while in the `Testing` folder, let there be all images at the same place.

This allows the model to train to be able to classify the images as fire or non fire.

After finishing this, the directory will look like this:

![image](https://user-images.githubusercontent.com/81915404/162608510-8bfab761-0b0c-4339-8bec-8e388901a241.png)

Now lets see how to implement the code in vs code.

## 4)Implementing the Convolutional Neural Network (CNN) Algorithm

To implement the code, click on the `fire_detection.ipynb` file in the open editor in vs code shown above. Then, start doing `shift + enter` on all the blocks indivisually or press run all button on top to execute the program.

When the block with model.fit() command is executed, the training will start and will continue for a while depending on size of the dataset as well as the number of epochs it is being trained for. The output/training steps will look like wats shown in the `fire_detection.txt` file which was output for 50 epochs.

![image](https://user-images.githubusercontent.com/81915404/162609237-298402ce-f019-4806-b0c5-9041893120a8.png)

Finally, when the last code block is executed, the model will take the testing dataset images as input and for each of them, print if the image has fire or not.

The output obtained from the above code was :

![image](https://user-images.githubusercontent.com/81915404/162609311-9481c66d-a870-4c63-9d0f-92ba7430c9c7.png)

![image](https://user-images.githubusercontent.com/81915404/162609318-ceaa9cc9-e671-469a-aaa6-d80e6a319428.png)

As we can see, the model is able to correctly detect if the image contains fire or not.

## 5) Analysing the Output 

The journey of image detection doesnt end with correct outputs, The output now has to be analysed to see if the model is able to get the correct output for different scenarios and conditions in the images. 

Our output on the testing dataset looks as follows:

![image](https://user-images.githubusercontent.com/81915404/162609564-da1455b7-f2a5-4094-8fb9-5d7e03f66a6e.png) ![image](https://user-images.githubusercontent.com/81915404/162609570-ac401dc9-386a-4190-957a-a72a370d5da9.png) ![image](https://user-images.githubusercontent.com/81915404/162609604-7d42fcb4-4b08-4f94-8a4d-c27ffc0c7ecf.png)

![image](https://user-images.githubusercontent.com/81915404/162609634-15b394c4-2aaf-488e-9d2c-19c7448502d9.png) ![image](https://user-images.githubusercontent.com/81915404/162609645-11cb0516-47aa-48b0-8d59-cba2ee01bb2d.png)

### The Exception:
As stated above, the model can only predict based on whatever has been fed as data to it. In CNN algorithm, the model finds specific pstterns n the images and filters then through multiple neural netwrok layers to come up with the best aggregate of pattern that can detect the image. 

For our dataset, the model is able to detect most of the images as fire or non fire correctly. However, in some case as shown below, where the condition satisfies the pattern found by the model for fire but the existnce of fire is not clearly seen, we cannot say for sure if fire is there in the image. However, based on the patterns the model has learned through training, it detects the image to have fire. 

![image](https://user-images.githubusercontent.com/81915404/162609819-7fe2fc9d-b1ee-4971-ace6-8c176a002ad9.png)


Thus as stated earlier, detection depends on the way the dataset was split into the three sets. Checking for better outputs by trying multiple different split percentages is also a way to see which one fits best . However, since all possible combinations cannot be tried with all different images, we can do this for a limited number of times and find the best fit we are able to get.

