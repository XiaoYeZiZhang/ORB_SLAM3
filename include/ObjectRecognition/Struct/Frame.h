//
// Created by zhangye on 2020/9/16.
//

#ifndef ORB_SLAM3_FRAME_H
#define ORB_SLAM3_FRAME_H
#include <Eigen/Core>
#include <opencv2/core/mat.hpp>
namespace ObjRecognition {

class FrameData {
public:
    long unsigned int mFrmIndex;
    double mTimeStamp;

    Eigen::Matrix3d mRcw;
    Eigen::Vector3d mTcw;

    cv::Mat img;
    std::vector<cv::KeyPoint> mKpts;
    cv::Mat mDesp;
};

typedef struct ObjRecogImageCallbackData {
    unsigned char *data; /// can be read during callback
    int type;            /// same as cv::Mat type(). e.g. CV_8UC1 or CV_16UC1
    int width;
    int height;
    int stride; /// same as cv::Mat step
} ObjRecogImageCallbackData;

typedef struct ObjRecogFrameCallbackData {
    long unsigned int id; /// frame index
    int flag;             /// frame state
    double timestamp;     /// frame timestamp

    bool has_image;                /// rgb is valid
    ObjRecogImageCallbackData img; /// img data

    /// camera pose, Rcw, tcw
    double R[3][3];
    double t[3];

    /// kpts and descriptor
    int feature_mem_size;
    char *feature_mem;
} ObjRecogFrameCallbackData;

} // namespace ObjRecognition

#endif // ORB_SLAM3_FRAME_H
