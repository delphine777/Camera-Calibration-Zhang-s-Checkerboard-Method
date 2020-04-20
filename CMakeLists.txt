cmake_minimum_required (VERSION 3.0)

project(StereoCalibration)

FIND_PACKAGE ( Threads REQUIRED )

###########################################
#       Environment definitions
###########################################

# Init env variables
set(WIN_ENV false)
set(ANDROID_ENV false)
set(LX_ENV false)

# Set win32/64 flag
if(WIN32 OR WIN64)
    if( CMAKE_SIZEOF_VOID_P EQUAL 8 )
        set(WIN64_ENV true)
    else()
        set(WIN32_ENV true)
    endif()
endif()

# Update environment
if(WIN32_ENV)
    set(WIN_ENV true)
    message("Windows32 Environment Detected")
    add_definitions(-DWIN_ENV)
    add_definitions(-DWIN32_ENV)

elseif(WIN64_ENV)
    set(WIN_ENV true)
    message("Windows64 Environment Detected")
    add_definitions(-DWIN_ENV)
    add_definitions(-DWIN64_ENV)

elseif(ANDROID32)
    set(ANDROID_ENV true)
    message("Android Environment Detected")
    add_definitions(-DANDROID_ENV)

elseif(ANDROID64)
    set(ANDROID_ENV true)
    message("Android64 Environment Detected")
    add_definitions(-DANDROID_ENV)

elseif(UNIX)
    set(LX_ENV true)
    message("Linux Environment Detected")
    add_definitions(-DLX_ENV)

endif()

###########################################
#       Features per OS
###########################################

# Init all external dependencies to false
set(USE_OPENCV false)

set( STATIC_BUILD false )
set(OS_NAME "")


# Update features per OS
if(WIN_ENV AND WIN32_ENV)
    set(OS_NAME "win32")
    set(USE_OPENCV true)


elseif(WIN_ENV AND WIN64_ENV)
    set(OS_NAME "win64")
	set(USE_OPENCV true)
	set(USE_QT true)


elseif(ANDROID_ENV AND ANDROID32)
    set(OS_NAME "android")

	set(USE_OPENCV true)

elseif(ANDROID_ENV AND ANDROID64)
    set(OS_NAME "android64")


elseif(LX_ENV)
    set(OS_NAME "lx")
	set(USE_OPENCV true)

endif()


###########################################
#       External dependencies
###########################################

    # OpenCV header files
if(USE_OPENCV)
    include_directories(dep/opencv/include/calib3d)
    include_directories(dep/opencv/include/core)
    include_directories(dep/opencv/include/dnn)
    include_directories(dep/opencv/include/features2d)
    include_directories(dep/opencv/include/flann)
    include_directories(dep/opencv/include/gapi)
    include_directories(dep/opencv/include/highgui)
    include_directories(dep/opencv/include/imgcodecs)
    include_directories(dep/opencv/include/imgproc)
    include_directories(dep/opencv/include/ml)
    include_directories(dep/opencv/include/objdetect)
    include_directories(dep/opencv/include/photo)
    include_directories(dep/opencv/include/stitching)
    include_directories(dep/opencv/include/ts)
    include_directories(dep/opencv/include/video)
    include_directories(dep/opencv/include/videoio)
    include_directories(dep/opencv/include/xfeatures2d)
    include_directories(dep/opencv/include/ximgproc)
include_directories(dep/opencv/include/opencv2)
endif()

    # List OpencvLibs
if(WIN_ENV)
    if(USE_OPENCV)
		set (OPENCV_LIBS
			optimized
			opencv_calib3d412
			optimized
			opencv_core412
			optimized
			opencv_dnn412
			optimized
			opencv_features2d412
			optimized
			opencv_flann412
			optimized
			opencv_gapi412
			optimized
			opencv_highgui412
			optimized
			opencv_imgcodecs412
			optimized
			opencv_imgproc412
			optimized
			opencv_ml412
			optimized
			opencv_objdetect412
			optimized
			opencv_photo412
			optimized
			opencv_stitching412
			optimized
			opencv_ts412
			optimized
			opencv_video412
			optimized
			opencv_videoio412
			optimized
			opencv_xfeatures2d412
            optimized
            opencv_ximgproc412

			debug
			opencv_calib3d412d
			debug
			opencv_core412d
			debug
			opencv_dnn412d
			debug
			opencv_features2d412d
			debug
			opencv_flann412d
			debug
			opencv_gapi412d
			debug
			opencv_highgui412d
			debug
			opencv_imgcodecs412d
			debug
			opencv_imgproc412d
			debug
			opencv_ml412d
			debug
			opencv_objdetect412d
			debug
			opencv_photo412d
			debug
			opencv_stitching412d
			debug
			opencv_ts412d
			debug
			opencv_video412d
			debug
			opencv_videoio412d
			debug
			opencv_xfeatures2d412d
            debug
            opencv_ximgproc412d
		)
	endif()
elseif(LX_ENV)
	if(USE_OPENCV)
		set( OPENCV_LIBS
			opencv_calib3d
			opencv_core
			opencv_dnn
			opencv_features2d
			opencv_flann
			opencv_gapi
			opencv_highgui
			opencv_imgcodecs
			opencv_imgproc
			opencv_ml
			opencv_objdetect
			opencv_photo
			opencv_stitching
			opencv_video
			opencv_videoio
		   )
	endif()
elseif(ANDROID_ENV)
	if(USE_OPENCV)
		set( OPENCV_LIBS
			opencv_core
			opencv_highgui
			opencv_imgproc
			opencv_imgcodecs
			opencv_videoio
			opencv_features2d
			opencv_flann
			opencv_calib3d
		   )
	endif()
endif()



if(USE_QT)

    if(LX_ENV)

        set(QT_LIBS
            Qt5Core
            )
    endif()

endif()

###########################################
#       Compilation flags
###########################################
if(NOT WIN_ENV)
    add_definitions(-std=c++11)
    add_definitions(-pthread)
    #add_definitions(-zlib)
    add_definitions(-Wall)
    add_definitions(-O2)
endif()

# ARM compilation flags for Android
if( ANDROID_ENV AND ANDROID32)
    add_definitions(-march=armv7-a -mfpu=neon -mfloat-abi=softfp) # Originally: mfpu=vfpv3-d16
    add_definitions(-lsupc++) # todo: check if relevant
endif()

if( ANDROID_ENV AND ANDROID64)
    add_definitions(-march=armv8-a -fPIE)
    add_definitions(-lsupc++) # todo: check if relevant
endif()

if(USE_QT)
    add_definitions(-DUSE_QT)
endif()


###########################################
#       Link to external libs
###########################################

if(USE_OPENCV)
    set(OPENCV_PATH ${CMAKE_SOURCE_DIR}/dep/opencv/lib/${OS_NAME})
    link_directories(${OPENCV_PATH})
else()
    link_directories(${OPENCV_COMPILE_DIR})
endif()


if(USE_QT)
    set(QT_PATH ${CMAKE_SOURCE_DIR}/dep/qt5/lib/${OS_NAME})
    link_directories(${QT_PATH})
endif()
###########################################
#       Configure destination folders
###########################################

    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    set(CMAKE_INSTALL_PREFIX "bin")


###########################################
#       Default options
###########################################

if( STATIC_BUILD)
    option(BUILD_SHARED_LIBS "Build shared libraries (DLLs)." OFF)
else()
    option(BUILD_SHARED_LIBS "Build shared libraries (DLLs)." ON)
endif()



add_subdirectory(main)
