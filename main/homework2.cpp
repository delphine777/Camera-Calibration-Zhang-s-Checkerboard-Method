#include <iostream>
#include <opencv.hpp>
#include <opencv2/core.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/calib3d.hpp"
#include <opencv2/highgui.hpp>
#include <string>
#include <vector>
#include <fstream>
using namespace std;
using namespace cv;


#define PATH "C:/Users/nicky/source/repos/StereoCalibration/Calib_Test_Pictures/"
#define NUM 20


int main(int argc, char* argv[])
{
    Mat leftImage, rightImage;   //input image
    vector<string> leftFileList, rightFileList;   //file name and path
    vector<Mat> rotateMat[2], translateMat[2];
    
    Mat leftCameraMatrix = Mat::eye(3, 3, CV_64F);  //camera matrix
    Mat rightCameraMatrix = Mat::eye(3, 3, CV_64F);
    
    Mat leftDistCoeffs = Mat::zeros(9, 1, CV_64F);
    Mat rightDistCoeffs = Mat::zeros(9, 1, CV_64F);
    int flag = 0;

    vector<Point2f> leftCorners, rightCorners;   //2d corner points
    vector<vector<Point2f>> cornersLeft, cornersRight;   //3d corner points
    vector<Point3f> worldPoints;
    vector<vector<Point3f>> worldPoints2;

    //*******************Read all images from the file****************************
    for (int i = 1; i <= NUM; i++)
    {
        stringstream str;
        str << PATH <<  setfill('0') << i << "left.png";
        //cout << str.str() << endl;
        leftFileList.push_back(str.str());
    }
    for (int i = 1; i <= NUM; i++)
    {
        stringstream str;
        str << PATH << setfill('0') << i << "right.png";
        //cout << str.str() << endl;
        rightFileList.push_back(str.str());
    }


    //*******************generate object points in real world*********************
    for (int j = 0; j < 8; j++)
    {
        for (int k = 0; k < 11; k++)
        {
            worldPoints.push_back(Point3f(j * 1.0, k * 1.0, 0.0f));
        }
    } 

    //*******************Find the corner points***********************************
    
    for (int i = 0; i < leftFileList.size(); i++) {
        leftImage = imread(leftFileList[i], IMREAD_COLOR);
        rightImage = imread(rightFileList[i], IMREAD_COLOR);
        /*
        if (leftImage.empty()) {
            std::cout << "failed to open img.jpg" << std::endl;
        }
        else {
            imshow("origin", leftImage);
        }
        waitKey(10);
        */
        bool leftFound = findChessboardCorners(leftImage, Size(11, 8), leftCorners, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
        bool rightFound = findChessboardCorners(rightImage, Size(11, 8), rightCorners, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
        //find the number of points 
        cornersLeft.push_back(leftCorners);
        cornersRight.push_back(rightCorners);
       // drawChessboardCorners(leftImage, Size(11, 8), leftCorners, leftFound);  //show the image
        //drawChessboardCorners(rightImage, Size(11, 8), rightCorners, rightFound);

        worldPoints2.push_back(worldPoints);  //push 2d points in world coordinate into 3d vector
    }

    //******************* calibrate camera******************************************
    calibrateCamera(worldPoints2, cornersLeft, leftImage.size(), leftCameraMatrix, leftDistCoeffs, rotateMat[0], translateMat[0], flag);
    calibrateCamera(worldPoints2, cornersRight, leftImage.size(), rightCameraMatrix, rightDistCoeffs, rotateMat[1], translateMat[1], flag);
    //cout << "rotate size" << rotateMat.size() << endl;
    //cout << "translate size" << translateMat.size() << endl;
    Mat R, T, E, F;
    


    double rms = stereoCalibrate(worldPoints2, cornersLeft, cornersRight, leftCameraMatrix, leftDistCoeffs, rightCameraMatrix, rightDistCoeffs, leftImage.size(), R, T, E, F);
   // composeRT(rotateMat[0], translateMat[0], rotateMat[1], translateMat[1], rotate, translate);
   // cout << "new rotate size" << rotateMat[0].size() << endl;
   // cout << "new translate size" << translateMat[0].size() << endl;
    //cout<< "Left cameraMatrix" << leftCameraMatrix.size() << "Left distCoeffs" << leftDistCoeffs.size();
   // cout << "Right cameraMatrix" << rightCameraMatrix.size() << "Right distCoeffs" << rightDistCoeffs.size();

    cout << F.size() << endl;
    //****************************Calculate reprojection error 3D to 2D************************
    vector<Point2f> leftImagePoints, rightImagePoints;
    vector<double> leftErr, rightErr;
    double totalErr = 0.0, err = 0.0;
    
    for (int i = 0; i < leftFileList.size(); i++) 
    {
        projectPoints(worldPoints2[i], rotateMat[0][i], translateMat[0][i], leftCameraMatrix, leftDistCoeffs, leftImagePoints);
        err = norm(cornersLeft[i], leftImagePoints, NORM_L1)/leftImagePoints.size();
        leftErr.push_back(err);
        totalErr += err;
    }

    double leftMeanErr = totalErr / leftFileList.size();
    double leftStd = 0.0;

    for (int i = 0; i < leftFileList.size(); i++)
    {
        leftStd += (leftErr[i] - leftMeanErr) * (leftErr[i] - leftMeanErr);
    }
    leftStd = sqrt(leftStd / leftFileList.size());

    totalErr = 0.0;
    for (int i = 0; i < rightFileList.size(); i++)
    {
        projectPoints(worldPoints2[i], rotateMat[1][i], translateMat[1][i], rightCameraMatrix, rightDistCoeffs, rightImagePoints);
        err = norm(cornersRight[i], rightImagePoints, NORM_L1)/rightImagePoints.size();
        rightErr.push_back(err);
        totalErr += err;
    }
    double rightMeanErr = totalErr / rightFileList.size();
    double rightStd = 0.0;
    for (int i = 0; i < rightFileList.size(); i++)
    {
        rightStd += (rightErr[i] - rightMeanErr) * (rightErr[i] - rightMeanErr);
    }
    rightStd = sqrt(rightStd / rightFileList.size());

    cout << "Left Mean Error: " << leftMeanErr << " " << "Right Mean Error: " << rightMeanErr << endl;

    cout << "Left Standard Deviation: " << leftStd << " " << "Right Standard Deviation: " << rightStd << endl;

    
 
    //******************************Calculation reprojection from 2D to 2D*******************************************
    float err2d = 0.0;
    for (int i = 0; i < leftFileList.size() ; i++)
    {
        Mat F = findFundamentalMat(cornersLeft[i], cornersRight[i], FM_RANSAC, 3, 0.99);
        //cout<< float(*(F.data)) <<" "<< float(*(F.data + 1)) << " " << float(*(F.data+2)) << " " << float(*(F.data+3)) << " " << float(*(F.data+4))
       // << " " << float(*(F.data + 5)) << " " << float(*(F.data + 6)) << " " << float(*(F.data + 7)) << " " << float(*(F.data + 8)) << " " << float(*(F.data + 9)) << " " << float(*(F.data + 10)) << endl;
        
        for (int j = 0; j < cornersLeft.size(); j++)
        {

            err2d+= abs((float(F.at<double>(0,0)) * cornersLeft[i][j].x + float(F.at<double>(0, 1)) * cornersLeft[i][j].y + float(F.at<double>(0, 2))) * cornersRight[i][j].x + (cornersLeft[i][j].x
                * float(F.at<double>(1, 0)) + cornersLeft[i][j].y * float(F.at<double>(1, 1)) + float(F.at<double>(1, 2))) * cornersRight[i][j].y +
                (float(F.at<double>(2, 0)) * cornersLeft[i][j].x + float(F.at<double>(2,1)) * cornersLeft[i][j].y + float(F.at<double>(2,2))));
            /*
         err2d += ((float (*(F.data))) * cornersLeft[i][j].x + (float(*(F.data + 1))) * cornersLeft[i][j].y +  (float(*(F.data+2))))* cornersRight[i][j].x +( cornersLeft[i][j].x
          * (float(*(F.data+3))) + cornersLeft[i][j].y * (float(*(F.data+4)))  + (float(*(F.data+5))))* cornersRight[i][j].y+
          + (float(*(F.data+6)) * cornersLeft[i][j].x + (float(*(F.data+7))) * cornersLeft[i][j].y + (float(*(F.data+8))));
          */
        }
    }

    err2d /= (leftFileList.size() * cornersLeft.size());

    cout << "2d Error: " << err2d <<  endl;
    //*******************Write to XML file******************************************
    FileStorage fs("calibration.xml", FileStorage::WRITE);
    fs << "frameCount" << 20;
    time_t rawtime;
    time(&rawtime);

    fs << "LeftCameraMatrix" << leftCameraMatrix << "LeftDistCoeffs" << leftDistCoeffs;
    fs << "RightCameraMatrix" <<  rightCameraMatrix << "RightDistCoeffs" << rightDistCoeffs;
 
    fs << "RotateMatrix" << R;
  

    fs << "TranslateMatrix" << T;
   

    fs.release();
    
    return 0;
}

