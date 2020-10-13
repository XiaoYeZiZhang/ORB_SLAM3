//
// Created by zhangye on 2020/9/15.
//

#ifndef ORB_SLAM3_TOOLS_H
#define ORB_SLAM3_TOOLS_H
#include <glog/logging.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pangolin/gl/gltext.h>
#include <pangolin/gl/glfont.h>
namespace ObjRecognition {
template <class T1, class T2>
static void PutDataToMem(
    T1 *dst_mem, T2 *src_mem, const unsigned int mem_size, unsigned int &pos) {
    memcpy(dst_mem, src_mem, mem_size);
    pos += mem_size;
}

static std::vector<std::string> split(std::string str, std::string pattern) {
    std::string::size_type pos;
    std::vector<std::string> result;
    str += pattern;
    int size = str.size();

    for (int i = 0; i < size; i++) {
        pos = str.find(pattern, i);
        if (pos < size) {
            std::string s = str.substr(i, pos - i);
            result.push_back(s);
            i = pos + pattern.size() - 1;
        }
    }
    return result;
}

static void DrawTxt(const std::string str, int x, int y) {
    if (str.empty()) {
        return;
    }

    std::vector<std::string> str_split_vector = split(str, "\n");
    glColor3f(0.8f, 0.8f, 0.8f);
    for (size_t index = 0; index < str_split_vector.size(); index++) {
        pangolin::GlText txt =
            pangolin::GlFont::I().Text(str_split_vector.at(index).c_str());
        txt.DrawWindow(x, y + 15 * index);
    }
}

static void ChangeCV44ToGLMatrixDouble(
    const cv::Mat &cv_mat, pangolin::OpenGlMatrix &opengl_matrix) {
    opengl_matrix.m[0] = cv_mat.at<double>(0, 0);
    opengl_matrix.m[1] = cv_mat.at<double>(1, 0);
    opengl_matrix.m[2] = cv_mat.at<double>(2, 0);
    opengl_matrix.m[3] = 0.0;

    opengl_matrix.m[4] = cv_mat.at<double>(0, 1);
    opengl_matrix.m[5] = cv_mat.at<double>(1, 1);
    opengl_matrix.m[6] = cv_mat.at<double>(2, 1);
    opengl_matrix.m[7] = 0.0;

    opengl_matrix.m[8] = cv_mat.at<double>(0, 2);
    opengl_matrix.m[9] = cv_mat.at<double>(1, 2);
    opengl_matrix.m[10] = cv_mat.at<double>(2, 2);
    opengl_matrix.m[11] = 0.0;

    opengl_matrix.m[12] = cv_mat.at<double>(0, 3);
    opengl_matrix.m[13] = cv_mat.at<double>(1, 3);
    opengl_matrix.m[14] = cv_mat.at<double>(2, 3);
    opengl_matrix.m[15] = 1.0;
}

static void ChangeCV44ToGLMatrixFloat(
    const cv::Mat &cv_mat, pangolin::OpenGlMatrix &opengl_matrix) {
    opengl_matrix.m[0] = cv_mat.at<float>(0, 0);
    opengl_matrix.m[1] = cv_mat.at<float>(1, 0);
    opengl_matrix.m[2] = cv_mat.at<float>(2, 0);
    opengl_matrix.m[3] = 0.0;

    opengl_matrix.m[4] = cv_mat.at<float>(0, 1);
    opengl_matrix.m[5] = cv_mat.at<float>(1, 1);
    opengl_matrix.m[6] = cv_mat.at<float>(2, 1);
    opengl_matrix.m[7] = 0.0;

    opengl_matrix.m[8] = cv_mat.at<float>(0, 2);
    opengl_matrix.m[9] = cv_mat.at<float>(1, 2);
    opengl_matrix.m[10] = cv_mat.at<float>(2, 2);
    opengl_matrix.m[11] = 0.0;

    opengl_matrix.m[12] = cv_mat.at<float>(0, 3);
    opengl_matrix.m[13] = cv_mat.at<float>(1, 3);
    opengl_matrix.m[14] = cv_mat.at<float>(2, 3);
    opengl_matrix.m[15] = 1.0;
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
