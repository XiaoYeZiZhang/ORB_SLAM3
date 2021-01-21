#include <fstream>
#include <glog/logging.h>
#include "Utility/FileIO.h"
void LoadPointCloudModel(
    const std::string &model_path,
    std::shared_ptr<ObjRecognition::Object> &object) {

    std::ifstream in;
    long long length;
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
    const std::string &model_path, char **buffer, long long &buffer_len) {

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

void LoadVoc(
    std::string voc_path_compressed, char **voc_buffer,
    unsigned int &voc_buf_size) {
    VLOG(0) << "LoadVoc voc path = " << voc_path_compressed.c_str();
    auto fb = fopen(voc_path_compressed.c_str(), "rb");
    fseek(fb, 0, SEEK_END);
    auto nFileLen = ftell(fb);
    fseek(fb, 0, SEEK_SET);
    *voc_buffer = new char[nFileLen];
    fread(*voc_buffer, sizeof(char), nFileLen, fb);
    fclose(fb);
    voc_buf_size = nFileLen;

    if (voc_buf_size <= 0 || voc_buffer == nullptr) {
        LOG(FATAL) << "voc file can't open " << voc_path_compressed;
    }
}