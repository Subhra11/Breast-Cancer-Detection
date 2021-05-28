# Breast-Cancer-Detection
Study of Image Processing Techniques to Improve Quality of  Mammographic Images for Breast Cancer Detection.
The project's aim is to develop and compare a variety of image processing techniques for automatically removing the background clutter, separating breast regions, and enhancing breast contrast. A pre-trained classification model named Inception-ResNet-V2 is given which is trained on natural Image data. The project demands the same pre-trained model to be re-trained on mammographic processed data as well as the raw data. In the end, the evaluation metrics (like AUC, ROC) will measure the effectiveness of the developed image processing techniques by comparing the classification performance between the model re-trained on the original data and on the enhanced image data.

Image_Enhancement.ipynb gives the detailed information of how to read the dataset and how to implement the Image pre-processing methods.

train_inception_org.py and train_inception_cropped.py are the python scripts used to train the classifier.

The dataset folder can be found in the link as provided in the appendix manual.
