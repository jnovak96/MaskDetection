# Python Mask Detector
This program was written in Python as an introduction to machine learning algorithms and convolutional neural network solutions. The program
implements the PyTorch library and uses a dataset that can be found here: https://www.kaggle.com/andrewmvd/face-mask-detection, but can be technically be used
with other datasets if the images have corresponding .xml files that store the same metadata. The set mentioned has 800+ images, however the program is
designed to split images that feature several faces into their own unique images for easier procssing. After the images have been split, the sample size is
increased to 4000+ images. The program can be easily modified to run at a higher cycle count, but with the current configuration I was able to get an accuracy result of
over 90% with the trained model.

# example of cropped sub-images
![alt text](https://raw.githubusercontent.com/jnovak96/MaskDetection/john/charts/Untitled.png)

# Accuracy scores based on class
![alt text](https://github.com/jnovak96/MaskDetection/blob/john/charts/accuracy.png?raw=true)

# Traning loss chart based on batch count
![alt text](https://github.com/jnovak96/MaskDetection/blob/john/charts/training_loss.png?raw=true)
