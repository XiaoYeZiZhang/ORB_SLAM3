//
// Created by zhangye on 2020/9/17.
//

#include <fstream>
#include <glog/logging.h>
#include <sys/stat.h>
#include "Utility/FileIO.h"
void LoadPointCloudModel(
    const std::string &model_path,
    std::shared_ptr<ObjRecognition::Object> &object) {

    std::ifstream in;
    int length;
    in.open(model_path.c_str(), std::ios::in);
    if (!in.is_open()) {
        LOG(FATAL) << "Error opening the pointCloud file!";
    }
    in.seekg(0, std::ios::end);
    length = in.tellg();
    in.seekg(0, std::ios::beg);
    char *buffer = new char[length];
    in.read(buffer, length);
    in.close();

    if (length <= 0 || buffer == nullptr) {
        LOG(FATAL) << "Point cloud model file can't open " << model_path;
    }

    object->LoadPointCloud(length, buffer);
    delete[] buffer;
    VLOG(0) << "PointCloud load succ";
}

void ReadPointCloudModelToBuffer(
    const std::string &model_path, char **buffer, int &buffer_len) {

    std::ifstream in;
    in.open(model_path.c_str(), std::ios::in);
    if (!in.is_open()) {
        LOG(FATAL) << "Error opening the pointCloud file!";
    }
    in.seekg(0, std::ios::end);
    buffer_len = in.tellg();
    in.seekg(0, std::ios::beg);
    *buffer = new char[buffer_len];
    in.read(*buffer, buffer_len);
    in.close();

    if (buffer_len <= 0 || *buffer == nullptr) {
        LOG(FATAL) << "Point cloud model file can't open " << model_path;
    }
}

std::string GetTimeStampString() {
    std::time_t t = std::time(0);
    std::tm *now = std::localtime(&t);
    char str[150];
    snprintf(
        str, sizeof(str), "%4d-%02d-%02d-%02d-%02d-%02d", now->tm_year + 1900,
        now->tm_mon + 1, now->tm_mday, now->tm_hour, now->tm_min, now->tm_sec);
    return std::string(str);
}

bool CreateFolder(std::string file_path_name) {

    if (access(file_path_name.c_str(), 0) == -1) {
        LOG(INFO) << file_path_name << " is not existing and create it";

        int res = mkdir(file_path_name.c_str(), 0777);
        if (res != 0) {
            LOG(INFO) << "can't create folder" << std::endl;
            return false;
        }
    }
    return true;
}