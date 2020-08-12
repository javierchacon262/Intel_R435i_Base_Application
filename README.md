# Intel R435i with YOLOv3 based Object Segmentation, Classification and Volume Regression

This repository contains the implementation of a fairly simple but very functional application making use of an Intel R435i sensor, it has a 1080p RGB camera @ 30 fps and a depth sensor based on a low-power infrared laser, which delivers precise measurements. even in unbalanced environments.

Two deep learning models were chosen for the processing of the images acquired by the sensor, the first is an implementation of YOLO-v3 based on TensorFlow 2, this is used to segment and classify RGB shapes. The second model is a regressor whose input is the combination of RGB and depth images. This is in charge of returning a weight matrix whose sum results in an approximation of the mass belonging to the objects captured by the sensor.

Both data capture and processing are carried out in threads independent of the processor and the main interface is in charge of displaying the captured images in real time as well as the classification and regression results that are updated every 5 seconds, 1 minute and 1 hour.

It is important to take into account the library requirements as well as the environment in which it was developed, Anaconda python 3.7.
