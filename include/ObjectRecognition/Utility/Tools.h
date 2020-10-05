//
// Created by zhangye on 2020/9/15.
//

#ifndef ORB_SLAM3_TOOLS_H
#define ORB_SLAM3_TOOLS_H
#include <glog/logging.h>
namespace ObjRecognition {
template <class T1, class T2>
static void PutDataToMem(
    T1 *dst_mem, T2 *src_mem, const unsigned int mem_size, unsigned int &pos) {
    memcpy(dst_mem, src_mem, mem_size);
    pos += mem_size;
}

static void PackCamCWToMem(
    const Eigen::Vector3d &Tcw, const Eigen::Matrix3d &Rcw,
    unsigned int &mem_pos, char *mem) {
    PutDataToMem(mem + mem_pos, &Tcw(0), sizeof(double), mem_pos);
    PutDataToMem(mem + mem_pos, &Tcw(1), sizeof(double), mem_pos);
    PutDataToMem(mem + mem_pos, &Tcw(2), sizeof(double), mem_pos);

    Eigen::Quaterniond QR(Rcw);
    PutDataToMem(mem + mem_pos, &(QR.w()), sizeof(double), mem_pos);
    PutDataToMem(mem + mem_pos, &(QR.x()), sizeof(double), mem_pos);
    PutDataToMem(mem + mem_pos, &(QR.y()), sizeof(double), mem_pos);
    PutDataToMem(mem + mem_pos, &(QR.z()), sizeof(double), mem_pos);
    VLOG(10) << "pose: " << QR.z();
}

static void PackORBFeatures(
    const std::vector<cv::KeyPoint> &vKpts, const cv::Mat &desp,
    unsigned int &mem_cur, char *mem) {
    unsigned int nKpts = vKpts.size();
    VLOG(10) << "keyframe kpts: " << nKpts;
    PutDataToMem(mem + mem_cur, &(nKpts), sizeof(nKpts), mem_cur);
    if (nKpts == 0) {
        LOG(FATAL) << "error: PackORBFeatures: nKpts.size = 0, mem_cur = "
                   << mem_cur;
        return;
    }

    for (auto &kpt : vKpts) {
        PutDataToMem(mem + mem_cur, &(kpt.pt.x), sizeof(kpt.pt.x), mem_cur);
        PutDataToMem(mem + mem_cur, &(kpt.pt.y), sizeof(kpt.pt.y), mem_cur);
        PutDataToMem(mem + mem_cur, &(kpt.size), sizeof(kpt.size), mem_cur);
        PutDataToMem(mem + mem_cur, &(kpt.angle), sizeof(kpt.angle), mem_cur);
        PutDataToMem(
            mem + mem_cur, &(kpt.response), sizeof(kpt.response), mem_cur);
        PutDataToMem(mem + mem_cur, &(kpt.octave), sizeof(kpt.octave), mem_cur);
        PutDataToMem(
            mem + mem_cur, &(kpt.class_id), sizeof(kpt.class_id), mem_cur);
    }
    const uchar *src_mem = desp.data;
    PutDataToMem(
        (uchar *)(mem + mem_cur), src_mem,
        desp.rows * desp.cols * sizeof(uchar), mem_cur);
}
} // namespace ObjRecognition
#endif // ORB_SLAM3_TOOLS_H
