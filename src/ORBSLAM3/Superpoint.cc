//
// Created by root on 2020/10/21.
//
#include "ORBSLAM3/SuperPoint.h"
#include <glog/logging.h>
#include <Eigen/Core>
#include <chrono>
#include <utility>
using namespace std::chrono;

namespace ORB_SLAM3 {

const int c1 = 64;
const int c2 = 64;
const int c3 = 128;
const int c4 = 128;
const int c5 = 256;
const int d1 = 256;

SuperPoint::SuperPoint()
    : conv1a(torch::nn::Conv2dOptions(1, c1, 3).stride(1).padding(1)),
      conv1b(torch::nn::Conv2dOptions(c1, c1, 3).stride(1).padding(1)),

      conv2a(torch::nn::Conv2dOptions(c1, c2, 3).stride(1).padding(1)),
      conv2b(torch::nn::Conv2dOptions(c2, c2, 3).stride(1).padding(1)),

      conv3a(torch::nn::Conv2dOptions(c2, c3, 3).stride(1).padding(1)),
      conv3b(torch::nn::Conv2dOptions(c3, c3, 3).stride(1).padding(1)),

      conv4a(torch::nn::Conv2dOptions(c3, c4, 3).stride(1).padding(1)),
      conv4b(torch::nn::Conv2dOptions(c4, c4, 3).stride(1).padding(1)),

      convPa(torch::nn::Conv2dOptions(c4, c5, 3).stride(1).padding(1)),
      convPb(torch::nn::Conv2dOptions(c5, 65, 1).stride(1).padding(0)),

      convDa(torch::nn::Conv2dOptions(c4, c5, 3).stride(1).padding(1)),
      convDb(torch::nn::Conv2dOptions(c5, d1, 1).stride(1).padding(0))

{

    register_module("conv1a", conv1a);
    register_module("conv1b", conv1b);
    register_module("conv2a", conv2a);
    register_module("conv2b", conv2b);
    register_module("conv3a", conv3a);
    register_module("conv3b", conv3b);
    register_module("conv4a", conv4a);
    register_module("conv4b", conv4b);
    register_module("convPa", convPa);
    register_module("convPb", convPb);
    register_module("convDa", convDa);
    register_module("convDb", convDb);
}

std::vector<torch::Tensor> SuperPoint::forward(torch::Tensor x) {
    x = torch::relu(conv1a->forward(x));
    x = torch::relu(conv1b->forward(x));
    x = torch::max_pool2d(x, 2, 2);

    x = torch::relu(conv2a->forward(x));
    x = torch::relu(conv2b->forward(x));
    x = torch::max_pool2d(x, 2, 2);

    x = torch::relu(conv3a->forward(x));
    x = torch::relu(conv3b->forward(x));
    x = torch::max_pool2d(x, 2, 2);

    x = torch::relu(conv4a->forward(x));
    x = torch::relu(conv4b->forward(x));

    auto cPa = torch::relu(convPa->forward(x));
    auto semi = convPb->forward(cPa); // [B, 65, H/8, W/8]

    auto cDa = torch::relu(convDa->forward(x));
    auto desc = convDb->forward(cDa); // [B, d1, H/8, W/8]

    auto dn = torch::norm(desc, 2, 1);
    desc = desc.div(torch::unsqueeze(dn, 1));

    semi = torch::softmax(semi, 1);
    // semi = semi / (torch::sum(semi, 0) + 0.00001);

    semi = semi.slice(1, 0, 64);
    semi = semi.permute({0, 2, 3, 1}); // [B, H/8, W/8, 64]

    int Hc = semi.size(1);
    int Wc = semi.size(2);
    semi = semi.contiguous().view({-1, Hc, Wc, 8, 8});
    semi = semi.permute({0, 1, 3, 2, 4});
    semi = semi.contiguous().view({-1, Hc * 8, Wc * 8}); // [B, H, W]

    std::vector<torch::Tensor> ret;
    ret.push_back(semi);
    ret.push_back(desc);
    return ret;
}

void NMS(
    cv::Mat det, cv::Mat conf, cv::Mat desc, std::vector<cv::KeyPoint> &pts,
    cv::Mat &descriptors, int border, int dist_thresh, int img_width,
    int img_height);
void NMS2(
    std::vector<cv::KeyPoint> det, cv::Mat conf, std::vector<cv::KeyPoint> &pts,
    int border, int dist_thresh, int img_width, int img_height);

SPDetector::SPDetector(
    std::shared_ptr<SuperPoint> _model,
    std::shared_ptr<torch::jit::script::Module> _traced_module_480_640,
    std::shared_ptr<torch::jit::script::Module> _traced_module_400_533,
    std::shared_ptr<torch::jit::script::Module> _traced_module_333_444)
    : model(std::move(_model)),
      traced_module_480_640(std::move(_traced_module_480_640)),
      traced_module_400_533(std::move(_traced_module_400_533)),
      traced_module_333_444(std::move(_traced_module_333_444)) {
    traced_module_480_640->to(torch::Device(torch::kCUDA));
    traced_module_400_533->to(torch::Device(torch::kCUDA));
    traced_module_333_444->to(torch::Device(torch::kCUDA));
}

// get network output
void SPDetector::detect(cv::Mat &img, int level, bool cuda) {
    auto start = high_resolution_clock::now();
    cv::Mat img_float;
    img.convertTo(img_float, CV_32F);
    auto x = torch::zeros({1, 1, img_float.rows, img_float.cols});
    memcpy(x.data_ptr(), img_float.clone().data, x.numel() * sizeof(float));
    x = x / 255.0;
    bool use_cuda = cuda && torch::cuda::is_available();
    assert(use_cuda == true);
    x = x.set_requires_grad(false);

    std::vector<torch::jit::IValue> inputs;
    inputs.emplace_back(x.to(torch::Device(torch::kCUDA)));

    torch::Tensor semi;
    if (level == 0) {
        auto out_test = traced_module_480_640->forward(inputs).toGenericDict();
        semi = out_test.at("semi").toTensor();
        mDesc = out_test.at("desc").toTensor();
    } else if (level == 1) {
        auto out_test = traced_module_400_533->forward(inputs).toGenericDict();
        semi = out_test.at("semi").toTensor();
        mDesc = out_test.at("desc").toTensor();
    } else if (level == 2) {
        auto out_test = traced_module_333_444->forward(inputs).toGenericDict();
        semi = out_test.at("semi").toTensor();
        mDesc = out_test.at("desc").toTensor();
    }

    VLOG(5) << "Time taken by get network output: "
            << (duration_cast<microseconds>(
                    high_resolution_clock::now() - start))
                       .count() /
                   1000.0
            << " ms" << std::endl;
    // pose process
    semi = semi.squeeze(0).squeeze(0);
    start = high_resolution_clock::now();
    mProb_cpu = semi.to(torch::kCPU);
    VLOG(5) << "Time taken by move from GPU to CPU: "
            << (duration_cast<microseconds>(
                    high_resolution_clock::now() - start))
                       .count() /
                   1000.0
            << " ms" << std::endl;
}

static bool compare_response(cv::KeyPoint first, cv::KeyPoint second) {
    if (first.response > second.response)
        return true;
    else
        return false;
}

void SPDetector::getKeyPoints(
    float threshold, int iniX, int maxX, int iniY, int maxY,
    std::vector<cv::KeyPoint> &keypoints, bool nms) {
    auto start = high_resolution_clock::now();
    cv::Mat resultImg(mProb_cpu.size(0), mProb_cpu.size(1), CV_32F);
    std::memcpy(
        (void *)resultImg.data, mProb_cpu.data_ptr(),
        sizeof(float) * mProb_cpu.numel());
    VLOG(5) << "time of cv::Mat: "
            << (duration_cast<microseconds>(
                    high_resolution_clock::now() - start))
                       .count() /
                   1000.0
            << " ms" << std::endl;

    std::vector<cv::KeyPoint> keypoints_no_nms;
    keypoints_no_nms.reserve(resultImg.rows * resultImg.cols);

    start = high_resolution_clock::now();
    for (size_t i = 4; i < resultImg.rows - 10; i++) {
        for (size_t j = 4; j < resultImg.cols - 10; j++) {
            float value = resultImg.at<float>(i, j);
            if (value > threshold) {
                keypoints_no_nms.emplace_back(
                    (cv::KeyPoint(j, i, 8, -1, value)));
            }
        }
    }
    VLOG(5) << "time of push data to no nms: "
            << (duration_cast<microseconds>(
                    high_resolution_clock::now() - start))
                       .count() /
                   1000.0
            << "ms " << std::endl;

    start = high_resolution_clock::now();
    sort(keypoints_no_nms.begin(), keypoints_no_nms.end(), compare_response);
    if (nms) {
        cv::Mat conf(keypoints_no_nms.size(), 1, CV_32F);
        for (size_t i = 0; i < keypoints_no_nms.size(); i++) {
            int x = keypoints_no_nms[i].pt.x;
            int y = keypoints_no_nms[i].pt.y;
            conf.at<float>(i, 0) = resultImg.at<float>(y, x);
        }
        int border = 0;
        int dist_thresh = 4;
        int height = maxY - iniY;
        int width = maxX - iniX;
        NMS2(
            keypoints_no_nms, conf, keypoints, border, dist_thresh, width,
            height);
    } else {
        keypoints = keypoints_no_nms;
    }

    VLOG(5) << "time for nms: "
            << (duration_cast<microseconds>(
                    high_resolution_clock::now() - start))
                       .count() /
                   1000.0
            << "ms " << std::endl;
}

void SPDetector::computeDescriptors(
    const std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors,
    const bool cuda) {
    cv::Mat kpt_mat(keypoints.size(), 2, CV_32F); // [n_keypoints, 2]  (y, x)

    for (size_t i = 0; i < keypoints.size(); i++) {
        kpt_mat.at<float>(i, 0) = (float)keypoints[i].pt.y;
        kpt_mat.at<float>(i, 1) = (float)keypoints[i].pt.x;
    }

    auto fkpts = torch::from_blob(
        kpt_mat.data, {(long)keypoints.size(), 2}, torch::kFloat);

    auto grid =
        torch::zeros({1, 1, fkpts.size(0), 2}); // [1, 1, n_keypoints, 2]
    grid[0][0].slice(1, 0, 1) =
        2.0 * fkpts.slice(1, 1, 2) / mProb_cpu.size(1) - 1; // x
    grid[0][0].slice(1, 1, 2) =
        2.0 * fkpts.slice(1, 0, 1) / mProb_cpu.size(0) - 1; // y
    if (cuda) {
        grid = grid.to(torch::kCUDA);
    }

    auto desc = torch::grid_sampler(
        mDesc, grid, 0, 0, true);      // [1, 256, 1, n_keypoints]
    desc = desc.squeeze(0).squeeze(1); // [256, n_keypoints]

    // normalize to 1
    desc = desc.div(torch::unsqueeze(torch::norm(desc, 2, 0), 0));
    desc = desc.transpose(0, 1).contiguous(); // [n_keypoints, 256]
    desc = desc.to(torch::kCPU);
    cv::Mat desc_mat(
        cv::Size(desc.size(1), desc.size(0)), CV_32FC1, desc.data<float>());
    descriptors = desc_mat.clone();
}

void NMS2(
    std::vector<cv::KeyPoint> det, cv::Mat conf, std::vector<cv::KeyPoint> &pts,
    int border, int dist_thresh, int img_width, int img_height) {

    std::vector<cv::Point2f> pts_raw;
    for (int i = 0; i < det.size(); i++) {

        int u = (int)det[i].pt.x;
        int v = (int)det[i].pt.y;

        pts_raw.push_back(cv::Point2f(u, v));
    }

    cv::Mat grid = cv::Mat(cv::Size(img_width, img_height), CV_8UC1);
    cv::Mat inds = cv::Mat(cv::Size(img_width, img_height), CV_16UC1);

    cv::Mat confidence = cv::Mat(cv::Size(img_width, img_height), CV_32FC1);

    grid.setTo(0);
    inds.setTo(0);
    confidence.setTo(0);

    for (int i = 0; i < pts_raw.size(); i++) {
        int uu = (int)pts_raw[i].x;
        int vv = (int)pts_raw[i].y;

        grid.at<char>(vv, uu) = 1;
        inds.at<unsigned short>(vv, uu) = i;

        confidence.at<float>(vv, uu) = conf.at<float>(i, 0);
    }

    cv::copyMakeBorder(
        grid, grid, dist_thresh, dist_thresh, dist_thresh, dist_thresh,
        cv::BORDER_CONSTANT, 0);

    for (int i = 0; i < pts_raw.size(); i++) {
        int uu = (int)pts_raw[i].x + dist_thresh;
        int vv = (int)pts_raw[i].y + dist_thresh;

        if (grid.at<char>(vv, uu) != 1)
            continue;

        for (int k = -dist_thresh; k < (dist_thresh + 1); k++)
            for (int j = -dist_thresh; j < (dist_thresh + 1); j++) {
                if (j == 0 && k == 0)
                    continue;

                if (confidence.at<float>(vv + k, uu + j) <
                    confidence.at<float>(vv, uu))
                    grid.at<char>(vv + k, uu + j) = 0;
            }
        grid.at<char>(vv, uu) = 2;
    }

    size_t valid_cnt = 0;
    std::vector<int> select_indice;

    for (int v = 0; v < (img_height + dist_thresh); v++) {
        for (int u = 0; u < (img_width + dist_thresh); u++) {
            if (u - dist_thresh >= (img_width - border) ||
                u - dist_thresh < border ||
                v - dist_thresh >= (img_height - border) ||
                v - dist_thresh < border)
                continue;

            if (grid.at<char>(v, u) == 2) {
                int select_ind = (int)inds.at<unsigned short>(
                    v - dist_thresh, u - dist_thresh);
                cv::Point2f p = pts_raw[select_ind];
                float response = conf.at<float>(select_ind, 0);
                pts.push_back(cv::KeyPoint(p, 8.0f, -1, response));

                select_indice.push_back(select_ind);
                valid_cnt++;
            }
        }
    }
}

void NMS(
    cv::Mat det, cv::Mat conf, cv::Mat desc, std::vector<cv::KeyPoint> &pts,
    cv::Mat &descriptors, int border, int dist_thresh, int img_width,
    int img_height) {

    std::vector<cv::Point2f> pts_raw;

    for (int i = 0; i < det.rows; i++) {

        int u = (int)det.at<float>(i, 0);
        int v = (int)det.at<float>(i, 1);
        // float conf = det.at<float>(i, 2);

        pts_raw.push_back(cv::Point2f(u, v));
    }

    cv::Mat grid = cv::Mat(cv::Size(img_width, img_height), CV_8UC1);
    cv::Mat inds = cv::Mat(cv::Size(img_width, img_height), CV_16UC1);

    cv::Mat confidence = cv::Mat(cv::Size(img_width, img_height), CV_32FC1);

    grid.setTo(0);
    inds.setTo(0);
    confidence.setTo(0);

    for (int i = 0; i < pts_raw.size(); i++) {
        int uu = (int)pts_raw[i].x;
        int vv = (int)pts_raw[i].y;

        grid.at<char>(vv, uu) = 1;
        inds.at<unsigned short>(vv, uu) = i;

        confidence.at<float>(vv, uu) = conf.at<float>(i, 0);
    }

    cv::copyMakeBorder(
        grid, grid, dist_thresh, dist_thresh, dist_thresh, dist_thresh,
        cv::BORDER_CONSTANT, 0);

    for (int i = 0; i < pts_raw.size(); i++) {
        int uu = (int)pts_raw[i].x + dist_thresh;
        int vv = (int)pts_raw[i].y + dist_thresh;

        if (grid.at<char>(vv, uu) != 1)
            continue;

        for (int k = -dist_thresh; k < (dist_thresh + 1); k++)
            for (int j = -dist_thresh; j < (dist_thresh + 1); j++) {
                if (j == 0 && k == 0)
                    continue;

                if (conf.at<float>(vv + k, uu + j) < conf.at<float>(vv, uu))
                    grid.at<char>(vv + k, uu + j) = 0;
            }
        grid.at<char>(vv, uu) = 2;
    }

    size_t valid_cnt = 0;
    std::vector<int> select_indice;

    for (int v = 0; v < (img_height + dist_thresh); v++) {
        for (int u = 0; u < (img_width + dist_thresh); u++) {
            if (u - dist_thresh >= (img_width - border) ||
                u - dist_thresh < border ||
                v - dist_thresh >= (img_height - border) ||
                v - dist_thresh < border)
                continue;

            if (grid.at<char>(v, u) == 2) {
                int select_ind = (int)inds.at<unsigned short>(
                    v - dist_thresh, u - dist_thresh);
                pts.push_back(cv::KeyPoint(pts_raw[select_ind], 1.0f));

                select_indice.push_back(select_ind);
                valid_cnt++;
            }
        }
    }

    descriptors.create(select_indice.size(), 256, CV_32F);

    for (int i = 0; i < select_indice.size(); i++) {
        for (int j = 0; j < 256; j++) {
            descriptors.at<float>(i, j) = desc.at<float>(select_indice[i], j);
        }
    }
}

} // namespace ORB_SLAM3