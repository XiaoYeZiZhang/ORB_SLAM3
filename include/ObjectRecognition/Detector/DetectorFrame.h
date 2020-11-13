//
// Created by zhangye on 2020/9/16.
//

#ifndef ORB_SLAM3_DETECTORFRAME_H
#define ORB_SLAM3_DETECTORFRAME_H
#include <map>
#include "Struct/FrameBase.h"

typedef long unsigned int MapPointIndex;
namespace ObjRecognition {

class DetectorFrame : public FrameBase {

public:
    std::vector<std::vector<cv::DMatch>> m_dmatches_2d;
    std::map<int, MapPointIndex> m_matches_3d;
    std::map<int, MapPointIndex> m_matches_3d_byconnection;
    std::map<int, MapPointIndex> m_matches2dto3d_inliers;

private:
}; // class DetectorFrame

} // namespace ObjRecognition
#endif // ORB_SLAM3_DETECTORFRAME_H
