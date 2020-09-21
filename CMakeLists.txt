cmake_minimum_required(VERSION 2.8)
project(ORB_SLAM3)
set(CMAKE_CXX_STANDARD 14)
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


find_package(OpenCV 3)
if(NOT OpenCV_FOUND)
   message(FATAL_ERROR "OpenCV3 not found.")
endif()

MESSAGE("OPENCV VERSION:")
MESSAGE(${OpenCV_VERSION})

find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(gflags REQUIRED)
find_package(glog REQUIRED)

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${PROJECT_SOURCE_DIR}/include/CameraModels
${EIGEN3_INCLUDE_DIR}
${Pangolin_INCLUDE_DIRS}
${GFLAS_INCLUDE_DIRS}
${GLOG_INCLUDE_DIRS}
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

include_directories(${PROJECT_SOURCE_DIR}/include/ceres-solver/include)
include_directories(${PROJECT_SOURCE_DIR}/include/ObjectRecognition)
include_directories(${PROJECT_SOURCE_DIR}/include/ObjectRecognition/Utility/FeatureExtractor)
include_directories(${PROJECT_SOURCE_DIR}/include/PoseSolver)
include_directories(${PROJECT_SOURCE_DIR}/Thirdparty)

set (ORBSLAM3_FILES
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
        include/ORBSLAM3/OptimizableTypes.h
        include/ORBSLAM3/MLPnPsolver.h
        include/ORBSLAM3/TwoViewReconstruction.h
        include/ObjectRecognition/Utility/Camera.h
        include/ObjectRecognition/Utility/Utility.h
        include/ORBSLAM3/FrameObjectProcess.cc
        include/ORBSLAM3/FrameObjectProcess.h
        include/ORBSLAM3/QuickHull.cc
        include/ORBSLAM3/QuickHull.h
        src/ORBSLAM3/TwoViewReconstruction.cc)


set(ORBSLAM3_CAMERAMODEL_FILES
        include/CameraModels/GeometricCamera.h
        include/CameraModels/Pinhole.h
        include/CameraModels/KannalaBrandt8.h)

set(OBJECTRECOGNITION_FILES
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
        include/ObjectRecognition/Detector/DetectorThread.h
        src/ObjectRecognition/Detector/DetectorThread.cc
        include/ObjectRecognition/Detector/PointCloudDetector.h
        src/ObjectRecognition/Detector/PointCloudDetector.cc
        include/ObjectRecognition/Utility/ThreadBase.h
        src/ObjectRecognition/Utility/ThreadBase.cc
        include/ObjectRecognition/Struct/Frame.h
        include/ObjectRecognition/Utility/RecognitionBase.h
        include/ObjectRecognition/Tracker/TrackerThread.h
        src/ObjectRecognition/Tracker/TrackerThread.cc
        src/ObjectRecognition/Struct/Object.cc
        include/ObjectRecognition/Struct/Object.h
        include/ObjectRecognition/Struct/PointCloudObject.h
        src/ObjectRecognition/Struct/PointCloudObject.cc
        include/ObjectRecognition/Struct/FrameBase.h
        src/ObjectRecognition/Struct/FrameBase.cc
        src/ObjectRecognition/Detector/DetectorFrame.cc
        include/ObjectRecognition/Detector/DetectorFrame.h
        src/ObjectRecognition/Utility/Parameters.cc
        include/ObjectRecognition/Utility/Parameters.h
        #include/ObjectRecognition/Optimizer/Factor.h
        #include/ObjectRecognition/Optimizer/LBAOptimizer.h
        #src/ObjectRecognition/Optimizer/LBAOptimizer.cc
        src/ObjectRecognition/Tracker/TrackerFrame.cc
        include/ObjectRecognition/Tracker/TrackerFrame.h
        src/ObjectRecognition/Tracker/PointCloudTracker.cc
        include/ObjectRecognition/Tracker/PointCloudTracker.h
        include/ObjectRecognition/Tracker/TrackerCommon.h
        src/ObjectRecognition/Tracker/TrackerCommon.cc
        src/ObjectRecognition/Detector/DetectorCommon.cc
        include/ObjectRecognition/Detector/DetectorCommon.h
        src/ObjectRecognition/Utility/FeatureExtractor/FeatureExtractor.cc
        src/ObjectRecognition/Utility/FeatureExtractor/ORBExtractor.cc
        src/ObjectRecognition/Utility/FeatureExtractor/ORBExtractorFactory.cc
        src/ObjectRecognition/Utility/FeatureExtractor/ORBExtractorHPC.cc
        include/ObjectRecognition/Utility/FeatureExtractor/FeatureExtractor.h
        include/ObjectRecognition/Utility/FeatureExtractor/ORBExtractor.h
        include/ObjectRecognition/Utility/FeatureExtractor/ORBExtractorHPC.h
        src/ObjectRecognition/Utility/FileIO.cc
        include/ObjectRecognition/Utility/FileIO.h
        include/ObjectRecognition/Utility/GlobalSummary.h
        src/ObjectRecognition/Utility/GlobalSummary.cc
        )
set(CERES_SOLVER_FILES
        include/ceres-solver/include/ceres/autodiff_cost_function.h
        include/ceres-solver/include/ceres/c_api.h
        include/ceres-solver/include/ceres/autodiff_local_parameterization.h
        include/ceres-solver/include/ceres/ceres.h
        include/ceres-solver/include/ceres/conditioned_cost_function.h
        include/ceres-solver/include/ceres/cost_function.h
        include/ceres-solver/include/ceres/cost_function_to_functor.h
        include/ceres-solver/include/ceres/covariance.h
        include/ceres-solver/include/ceres/crs_matrix.h
        include/ceres-solver/include/ceres/cubic_interpolation.h
        include/ceres-solver/include/ceres/dynamic_autodiff_cost_function.h
        include/ceres-solver/include/ceres/dynamic_cost_function.h
        include/ceres-solver/include/ceres/dynamic_cost_function_to_functor.h
        include/ceres-solver/include/ceres/dynamic_numeric_diff_cost_function.h
        include/ceres-solver/include/ceres/fpclassify.h
        include/ceres-solver/include/ceres/gradient_checker.h
        include/ceres-solver/include/ceres/gradient_problem.h
        include/ceres-solver/include/ceres/gradient_problem_solver.h
        include/ceres-solver/include/ceres/iteration_callback.h
        include/ceres-solver/include/ceres/jet.h
        include/ceres-solver/include/ceres/local_parameterization.h
        include/ceres-solver/include/ceres/loss_function.h
        include/ceres-solver/include/ceres/normal_prior.h
        include/ceres-solver/include/ceres/numeric_diff_cost_function.h
        include/ceres-solver/include/ceres/numeric_diff_options.h
        include/ceres-solver/include/ceres/ordered_groups.h
        include/ceres-solver/include/ceres/problem.h
        include/ceres-solver/include/ceres/rotation.h
        include/ceres-solver/include/ceres/sized_cost_function.h
        include/ceres-solver/include/ceres/solver.h
        include/ceres-solver/include/ceres/types.h
        include/ceres-solver/include/ceres/version.h
        include/ceres-solver/include/ceres/internal/autodiff.h
        include/ceres-solver/include/ceres/internal/config.h
        include/ceres-solver/include/ceres/internal/disable_warnings.h
        include/ceres-solver/include/ceres/internal/eigen.h
        include/ceres-solver/include/ceres/internal/fixed_array.h
        include/ceres-solver/include/ceres/internal/macros.h
        include/ceres-solver/include/ceres/internal/manual_constructor.h
        include/ceres-solver/include/ceres/internal/numeric_diff.h
        include/ceres-solver/include/ceres/internal/port.h
        include/ceres-solver/include/ceres/internal/reenable_warnings.h
        include/ceres-solver/include/ceres/internal/scoped_ptr.h
        include/ceres-solver/include/ceres/internal/variadic_evaluate.h
        )

set (POSE_SOLVER_FILES
        include/PoseSolver/essential3.h
        include/PoseSolver/Generator.h
        include/PoseSolver/geometry.h
        include/PoseSolver/p2p.h
        include/PoseSolver/PoseSolver.h
        src/PoseSolver/Generator.cc
        src/PoseSolver/p2p.cc
        src/PoseSolver/PoseSolver.cc
        src/PoseSolver/PoseSolverRansac.cc
        src/PoseSolver/PoseSolverRefine.cc
        src/PoseSolver/essential3.cc
        )

set (DBOW3_FILES
        Thirdparty/DBow3/src/BowVector.h
        Thirdparty/DBow3/src/BowVector.cpp
        Thirdparty/DBow3/src/Database.cpp
        Thirdparty/DBow3/src/Database.h
        Thirdparty/DBow3/src/DBoW3.h
        Thirdparty/DBow3/src/DescManip.cpp
        Thirdparty/DBow3/src/DescManip.h
        Thirdparty/DBow3/src/exports.h
        Thirdparty/DBow3/src/FeatureVector.cpp
        Thirdparty/DBow3/src/FeatureVector.h
        Thirdparty/DBow3/src/QueryResults.cpp
        Thirdparty/DBow3/src/QueryResults.h
        Thirdparty/DBow3/src/quicklz.c
        Thirdparty/DBow3/src/quicklz.h
        Thirdparty/DBow3/src/ScoringObject.cpp
        Thirdparty/DBow3/src/ScoringObject.h
        Thirdparty/DBow3/src/timers.h
        Thirdparty/DBow3/src/Vocabulary.cpp
        Thirdparty/DBow3/src/Vocabulary.h
        )

set (MODEL_FILE
        include/mode.h)

set (SOURCE_FILES
        ${ORBSLAM3_FILES}
        ${OBJECTRECOGNITION_FILES}
        ${POSE_SOLVER_FILES}
        ${CERES_SOLVER_FILES}
        ${DBOW3_FILES}
        ${ORBSLAM3_CAMERAMODEL_FILES}
        ${MODEL_FILE}
        )

add_library(${PROJECT_NAME} SHARED  ${SOURCE_FILES}

)

add_subdirectory(Thirdparty/g2o)

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
glog::glog
${GFLAGS_LIBRARIES}
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