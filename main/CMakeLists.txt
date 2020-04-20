cmake_minimum_required (VERSION 3.0)

project(main)

###########################################
#		Include directories
###########################################


    include_directories(../include)


###########################################
#		Source files & libs
###########################################
	set (SRC_FILES
		 homework2.cpp
		 )



###########################################
#		Executable
###########################################

	add_executable(main ${SRC_FILES})


###########################################
#		Compilation & Linkage flags
###########################################
    
if(ANDROID_ENV AND ANDROID32)
	# Add path to gnustl_shared lib
	# The variable	ROOTPATH is set in android.cmake
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L${ROOTPATH}/arm-linux-androideabi/lib/armv7-a/")
	
	# Add arm related flags. Recommended in Android-NDK documentation
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -march=armv7-a -Wl,--fix-cortex-a8")
endif()

if(ANDROID_ENV AND ANDROID64)

    # Add path to gnustl_shared lib
    # The variable  ROOTPATH is set in android.cmake
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -L${ROOTPATH}/aarch64-linux-android/lib/")

    # Add arm related flags. Recommended in Android-NDK documentation
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -march=armv8-a -pie")
endif()
    
if(WIN_ENV)
    #message(${CMAKE_BINARY_DIR})
    # Attempt for a work around to resolve linkage issue
    #add_custom_command(TARGET main
    #    POST_BUILD    
    #    COMMAND ${CMAKE_COMMAND} -E copy ${OPENCV_PATH}/debug/*.dll ${CMAKE_BINARY_DIR}
    #    )
endif()


###########################################
#		Link libs
###########################################

	# Link project libs
	target_link_libraries(main)


	# Android special libs from the toolchain
if( ANDROID_ENV )
	target_link_libraries(main ${GNUSTL_LIBS} log) # Removed: stdc++
endif()

	# Threads lib	
	target_link_libraries (main ${CMAKE_THREAD_LIBS_INIT})

    # link opencv libs
    target_link_libraries(main  ${OPENCV_LIBS})
    
if(USE_QT)
    target_link_libraries(main ${QT_LIBS})
endif()
