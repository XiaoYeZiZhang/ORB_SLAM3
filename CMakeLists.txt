cmake_minimum_required(VERSION 2.8)
project(ORB_SLAM3)

add_definitions(-w)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_C_FLAGS_DEBUG  "${CMAKE_C_FLAGS_DEBUG} -O0 -g -ggdb")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g -ggdb")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O2")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2")

# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wno-deprecated -O3 -march=native ")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-deprecated -O3 -march=native")

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

SET(OpenCV_DIR /home/zhangye/Downloads/install/opencv-3.4.0/install/share/OpenCV)
find_package(OpenCV 3)
if(NOT OpenCV_FOUND)
   message(FATAL_ERROR "OpenCV3 not found.")
endif()

MESSAGE("OPENCV VERSION:")
MESSAGE(${OpenCV_VERSION})

find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${PROJECT_SOURCE_DIR}/include/CameraModels
${EIGEN3_INCLUDE_DIR}
${Pangolin_INCLUDE_DIRS}
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(${PROJECT_NAME} SHARED
src/ORBSLAM3/System.cc
src/ORBSLAM3/Tracking.cc
src/ORBSLAM3/LocalMapping.cc
src/ORBSLAM3/LoopClosing.cc
src/ORBSLAM3/ORBextractor.cc
src/ORBSLAM3/ORBmatcher.cc
src/ORBSLAM3/FrameDrawer.cc
src/ORBSLAM3/Converter.cc
src/ORBSLAM3/MapPoint.cc
src/ORBSLAM3/KeyFrame.cc
src/ORBSLAM3/Atlas.cc
src/ORBSLAM3/Map.cc
src/ORBSLAM3/MapDrawer.cc
src/ORBSLAM3/Optimizer.cc
src/ORBSLAM3/PnPsolver.cc
src/ORBSLAM3/Frame.cc
src/ORBSLAM3/KeyFrameDatabase.cc
src/ORBSLAM3/Sim3Solver.cc
src/ORBSLAM3/Initializer.cc
src/ORBSLAM3/Viewer.cc
src/ORBSLAM3/ImuTypes.cc
src/ORBSLAM3/G2oTypes.cc
src/CameraModels/Pinhole.cpp
src/CameraModels/KannalaBrandt8.cpp
src/ORBSLAM3/OptimizableTypes.cpp
src/ORBSLAM3/MLPnPsolver.cpp
include/ORBSLAM3/System.h
include/ORBSLAM3/Tracking.h
include/ORBSLAM3/LocalMapping.h
include/ORBSLAM3/LoopClosing.h
include/ORBSLAM3/ORBextractor.h
include/ORBSLAM3/ORBmatcher.h
include/ORBSLAM3/FrameDrawer.h
include/ORBSLAM3/Converter.h
include/ORBSLAM3/MapPoint.h
include/ORBSLAM3/KeyFrame.h
include/ORBSLAM3/Atlas.h
include/ORBSLAM3/Map.h
include/ORBSLAM3/MapDrawer.h
include/ORBSLAM3/Optimizer.h
include/ORBSLAM3/PnPsolver.h
include/ORBSLAM3/Frame.h
include/ORBSLAM3/KeyFrameDatabase.h
include/ORBSLAM3/Sim3Solver.h
include/ORBSLAM3/Initializer.h
include/ORBSLAM3/Viewer.h
include/ORBSLAM3/ImuTypes.h
include/ORBSLAM3/G2oTypes.h
include/CameraModels/GeometricCamera.h
include/CameraModels/Pinhole.h
include/CameraModels/KannalaBrandt8.h
include/ORBSLAM3/OptimizableTypes.h
include/ORBSLAM3/MLPnPsolver.h
include/ORBSLAM3/TwoViewReconstruction.h
include/ObjectRecognition/Utility/Camera.h
include/ObjectRecognition/Utility/Utility.h
src/ORBSLAM3/TwoViewReconstruction.cc
include/ObjectRecognition/ObjectRecognitionSystem/ObjectRecognitionManager.h
include/ObjectRecognition/ObjectRecognitionSystem/ObjectRecognitionSystem.h
include/ObjectRecognition/Interface/HandleBase.h
include/ObjectRecognition/Utility/Thread/LockAnnotaion.h
include/ObjectRecognition/Utility/Thread/MessageQueue.h
include/ObjectRecognition/Utility/Thread/Mutex.h
include/ObjectRecognition/Utility/Thread/TaskQueue.h
include/ObjectRecognition/Utility/Thread/Thread.h
include/ObjectRecognition/Utility/Thread/ThreadBase.h
src/ObjectRecognition/ObjectRecognitionSystem/ObjectRecognitionManager.cc
src/ObjectRecognition/ObjectRecognitionSystem/ObjectRecognitionSystem.cc
src/ObjectRecognition/Utility/Thread/MessageQueue.cc
src/ObjectRecognition/Utility/Thread/Mutex.cc
src/ObjectRecognition/Utility/Thread/Runconfig.cc
src/ObjectRecognition/Utility/Thread/TaskQueue.cc
src/ObjectRecognition/Utility/Thread/Thread.cc
src/ObjectRecognition/Utility/Thread/ThreadBase.cc
include/ObjectRecognition/Utility/Performance.h
src/ObjectRecognition/Utility/Performance.cc
include/ObjectRecognition/Utility/Tools.h
)

add_subdirectory(Thirdparty/g2o)

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
-lboost_serialization
-lcrypto
)


# Build examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Monocular)

add_executable(mono_tum
Examples/Monocular/mono_tum.cc)
target_link_libraries(mono_tum ${PROJECT_NAME})

add_executable(mono_kitti
Examples/Monocular/mono_kitti.cc)
target_link_libraries(mono_kitti ${PROJECT_NAME})

add_executable(mono_euroc
Examples/Monocular/mono_euroc.cc)
target_link_libraries(mono_euroc ${PROJECT_NAME})

add_executable(mono_tum_vi
Examples/Monocular/mono_tum_vi.cc)
target_link_libraries(mono_tum_vi ${PROJECT_NAME})


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Monocular-Inertial)

add_executable(mono_inertial_euroc
Examples/Monocular-Inertial/mono_inertial_euroc.cc)
target_link_libraries(mono_inertial_euroc ${PROJECT_NAME})

add_executable(mono_inertial_objectRecognition_test
        Examples/Monocular-Inertial/mono_inertial_objectRecognition_test.cc)
target_link_libraries(mono_inertial_objectRecognition_test ${PROJECT_NAME})

add_executable(mono_inertial_tum_vi
Examples/Monocular-Inertial/mono_inertial_tum_vi.cc)
target_link_libraries(mono_inertial_tum_vi ${PROJECT_NAME})

