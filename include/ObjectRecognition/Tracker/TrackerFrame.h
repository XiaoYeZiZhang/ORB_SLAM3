//
// Created by zhangye on 2020/9/16.
//

#ifndef ORB_SLAM3_TRACKERFRAME_H
#define ORB_SLAM3_TRACKERFRAME_H
#include <memory>
#include <map>
#include "Struct/FrameBase.h"
typedef long unsigned int MapPointIndex;
namespace ObjRecognition {

class TrackerFrame : public FrameBase {

public:
    typedef std::shared_ptr<TrackerFrame> Ptr;

public:
    std::map<int, MapPointIndex> m_opticalflow_matches2dto3d;
    std::vector<cv::Point2d> m_opticalflow_point2ds;

    std::vector<std::pair<int, MapPointIndex>> m_opticalflow_matches3d_vec;
    std::vector<std::pair<int, MapPointIndex>> m_projection_matches3d_vec;

private:
}; // TrackerFrame

} // namespace ObjRecognition
#endif // ORB_SLAM3_TRACKERFRAME_H
