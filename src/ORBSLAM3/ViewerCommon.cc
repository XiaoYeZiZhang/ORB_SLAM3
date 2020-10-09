//
// Created by root on 2020/10/9.
//

#include <pangolin/display/opengl_render_state.h>
#include <pangolin/gl/gl.h>
#include "ORBSLAM3/ViewerCommon.h"

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

void PrintStatusForViewer(const int &status, cv::Mat &img) {
    switch (status) {
    case 1: {
        AddTextToImage("SLAM NOT INITIALIZED", img, 255, 0, 0);
        break;
    }
    case 2: {
        AddTextToImage("SLAM ON", img, 0, 255, 0);
        break;
    }
    case 3: {
        AddTextToImage("SLAM LOST", img, 255, 0, 0);
        break;
    }
    }
}

void PrintStatus(const int &status, const bool &bLocMode, cv::Mat &im) {
    if (!bLocMode) {
        switch (status) {
        case 1: {
            AddTextToImage("SLAM NOT INITIALIZED", im, 255, 0, 0);
            break;
        }
        case 2: {
            AddTextToImage("SLAM ON", im, 0, 255, 0);
            break;
        }
        case 3: {
            AddTextToImage("SLAM LOST", im, 255, 0, 0);
            break;
        }
        }
    } else {
        switch (status) {
        case 1: {
            AddTextToImage("SLAM NOT INITIALIZED", im, 255, 0, 0);
            break;
        }
        case 2: {
            AddTextToImage("LOCALIZATION ON", im, 0, 255, 0);
            break;
        }
        case 3: {
            AddTextToImage("LOCALIZATION LOST", im, 255, 0, 0);
            break;
        }
        }
    }
}

void LoadCameraPose(const cv::Mat &Tcw) {
    if (!Tcw.empty()) {
        pangolin::OpenGlMatrix M;

        M.m[0] = Tcw.at<float>(0, 0);
        M.m[1] = Tcw.at<float>(1, 0);
        M.m[2] = Tcw.at<float>(2, 0);
        M.m[3] = 0.0;

        M.m[4] = Tcw.at<float>(0, 1);
        M.m[5] = Tcw.at<float>(1, 1);
        M.m[6] = Tcw.at<float>(2, 1);
        M.m[7] = 0.0;

        M.m[8] = Tcw.at<float>(0, 2);
        M.m[9] = Tcw.at<float>(1, 2);
        M.m[10] = Tcw.at<float>(2, 2);
        M.m[11] = 0.0;

        M.m[12] = Tcw.at<float>(0, 3);
        M.m[13] = Tcw.at<float>(1, 3);
        M.m[14] = Tcw.at<float>(2, 3);
        M.m[15] = 1.0;

        M.Load();
    }
}

void DrawImageTexture(pangolin::GlTexture &imageTexture, cv::Mat &im) {
    if (!im.empty()) {
        imageTexture.Upload(im.data, GL_RGB, GL_UNSIGNED_BYTE);
        imageTexture.RenderToViewportFlipY();
    }
}