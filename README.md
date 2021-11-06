# Python Mask Detector
This program was written in Python as an introduction to machine learning algorithms and convolutional neural network solutions. The program
implements the PyTorch library and uses a dataset that can be found here: https://www.kaggle.com/andrewmvd/face-mask-detection, but can be technically be used
with other datasets if the images have corresponding .xml files that store the same metadata. The set mentioned has 800+ images, however the program is
designed to split images that feature several faces into their own unique imaages for easier procssing. After the images have been split, the sample size is
increased to 4000+ images. The program can be easily modified to run at a higher cycle count, but with the current configuration I was able to get an accuracy result of
over 90% with the trained model.
