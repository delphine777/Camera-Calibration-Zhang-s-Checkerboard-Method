---

## Clone this repository

Use these steps to clone from SourceTree, our client for using the repository command-line free. Cloning allows you to work on your files locally. If you don't yet have SourceTree, [download and install first](https://www.sourcetreeapp.com/). If you prefer to clone from the command line, see [Clone a repository](https://confluence.atlassian.com/x/4whODQ).

1. You’ll see the clone button under the **Source** heading. Click that button.
2. Now click **Check out in SourceTree**. You may need to create a SourceTree account or log in.
3. When you see the **Clone New** dialog in SourceTree, update the destination path and name if you’d like to and then click **Clone**.
4. Open the directory you just created to see your repository’s files.

---

## Use CMAKE to build the project

The path of the source code is "yourPath/StereoCalibration"
Build path is "yourPath/StereoCalibation/build"
Build in VS (both debug and release)

---

## Open the solution and change the image path into your own path

change:

#define PATH "C:/Users/nicky/source/repos/StereoCalibration/Calib_Test_Pictures/"
into 
#define PATH "yourPath/StereoCalibration/Calib_Test_Pictures/"


Now you can run it and see the result!