# Human-Activity-Recognition-for-Anomaly-Detection

This project works on ICPR 2010 Human Action Dataset and uses Convolutional Neural Networks (CNN) to detect action and simple rules to identify (ab)normality of the action.

divframes.py splits the given video into individual frames. The frames are then individually split into training and testing files.
harfad.py uses these frames to recognize actions. The CNN used is based on VGG16 pre-trained model and the top classifier model is built on top of this. 


Packages required:

OpenCV 3.1.0

Theano 0.8.2

Tensorflow 0.12.1

numpy 1.12.0

Keras 1.2.1
