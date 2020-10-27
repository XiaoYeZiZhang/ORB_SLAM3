//
// Created by root on 2020/10/9.
//

#include <pangolin/display/opengl_render_state.h>
#include <pangolin/gl/gl.h>
#include "ORBSLAM3/ViewerCommon.h"
#include "include/Tools.h"

void AddTextToImage(
    const std::string &s, cv::Mat &im, const int r, const int g, const int b) {
    int l = 10;
    // imText.rowRange(im.rows-imText.rows,imText.rows) =
    // cv::Mat::zeros(textSize.height+10,im.cols,im.type());
    cv::putText(
        im, s, cv::Point(l, im.rows - l), cv::FONT_HERSHEY_PLAIN, 1.5,
        cv::Scalar(255, 255, 255), 2, 8);
    cv::putText(
        im, s, cv::Point(l - 1, im.rows - l), cv::FONT_HERSHEY_PLAIN, 1.5,
        cv::Scalar(255, 255, 255), 2, 8);
    cv::putText(
        im, s, cv::Point(l + 1, im.rows - l), cv::FONT_HERSHEY_PLAIN, 1.5,
        cv::Scalar(255, 255, 255), 2, 8);
    cv::putText(
        im, s, cv::Point(l - 1, im.rows - (l - 1)), cv::FONT_HERSHEY_PLAIN, 1.5,
        cv::Scalar(255, 255, 255), 2, 8);
    cv::putText(
        im, s, cv::Point(l, im.rows - (l - 1)), cv::FONT_HERSHEY_PLAIN, 1.5,
        cv::Scalar(255, 255, 255), 2, 8);
    cv::putText(
        im, s, cv::Point(l + 1, im.rows - (l - 1)), cv::FONT_HERSHEY_PLAIN, 1.5,
        cv::Scalar(255, 255, 255), 2, 8);
    cv::putText(
        im, s, cv::Point(l - 1, im.rows - (l + 1)), cv::FONT_HERSHEY_PLAIN, 1.5,
        cv::Scalar(255, 255, 255), 2, 8);
    cv::putText(
        im, s, cv::Point(l, im.rows - (l + 1)), cv::FONT_HERSHEY_PLAIN, 1.5,
        cv::Scalar(255, 255, 255), 2, 8);
    cv::putText(
        im, s, cv::Point(l + 1, im.rows - (l + 1)), cv::FONT_HERSHEY_PLAIN, 1.5,
        cv::Scalar(255, 255, 255), 2, 8);

    cv::putText(
        im, s, cv::Point(l, im.rows - l), cv::FONT_HERSHEY_PLAIN, 1.5,
        cv::Scalar(r, g, b), 2, 8);
}

void PrintSLAMStatusForViewer(
    const int &status, const int &image_num, cv::Mat &img) {
    std::string img_str = "IMAGE: " + std::to_string(image_num);
    switch (status) {
    case 1: {
        AddTextToImage(img_str + " | SLAM NOT INITIALIZED", img, 255, 0, 0);
        break;
    }
    case 2: {
        AddTextToImage(img_str + " | SLAM ON", img, 0, 255, 0);
        break;
    }
    case 3: {
        AddTextToImage(img_str + " | SLAM LOST", img, 255, 0, 0);
        break;
    }
    }
}

void PrintStatus(
    const int &status, const bool &bLocMode, const int mappoint_num,
    cv::Mat &im) {
    if (!bLocMode) {
        switch (status) {
        case 1: {
            AddTextToImage(
                "SLAM NOT INITIALIZED| num: " + std::to_string(mappoint_num),
                im, 255, 0, 0);
            break;
        }
        case 2: {
            AddTextToImage(
                "SLAM ON| num: " + std::to_string(mappoint_num), im, 0, 255, 0);
            break;
        }
        case 3: {
            AddTextToImage(
                "SLAM LOST| num: " + std::to_string(mappoint_num), im, 255, 0,
                0);
            break;
        }
        }
    } else {
        switch (status) {
        case 1: {
            AddTextToImage(
                "SLAM NOT INITIALIZED| num: " + std::to_string(mappoint_num),
                im, 255, 0, 0);
            break;
        }
        case 2: {
            AddTextToImage(
                "LOCALIZATION ON| num: " + std::to_string(mappoint_num), im, 0,
                255, 0);
            break;
        }
        case 3: {
            AddTextToImage(
                "LOCALIZATION LOST| num: " + std::to_string(mappoint_num), im,
                255, 0, 0);
            break;
        }
        }
    }
}

void LoadCameraPose(const cv::Mat &Tcw) {
    if (!Tcw.empty()) {
        pangolin::OpenGlMatrix M;
        Tools::ChangeCV44ToGLMatrixFloat(Tcw, M);
        M.Load();
    }
}

void DrawImageTexture(pangolin::GlTexture &imageTexture, cv::Mat &im) {
    if (!im.empty()) {
        imageTexture.Upload(im.data, GL_RGB, GL_UNSIGNED_BYTE);
        imageTexture.RenderToViewportFlipY();
    }
}