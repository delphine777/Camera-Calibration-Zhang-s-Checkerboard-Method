# CameraCalibration

In this project, I implemented Camera Calibration using OPENCV library with Zhengyou Zhang's camera calibration technique. See the detail of this method here: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr98-71.pdf

Reprojection from 3D points to 2D points are calculated and reprojection errors are measured in both left and right images.
I also calculated 2D to 2D reprojection error with the Fundamental matrix measurement. 
See the detail of Fundamental matrix here: 
https://arxiv.org/pdf/1706.07886.pdf

We can rectify the left and right images after camara calibration. 
