//
// Created by zhangye on 2020/9/17.
//

#ifndef ORB_SLAM3_FILEIO_H
#define ORB_SLAM3_FILEIO_H
#include <iostream>
#include <memory>
#include "Struct/PointCloudObject.h"

void LoadPointCloudModel(
    const std::string &model_path,
    std::shared_ptr<ObjRecognition::Object> &object);

void ReadPointCloudModelToBuffer(
    const std::string &model_path, char **buffer, int &buffer_len);
std::string GetTimeStampString();
bool CreateFolder(std::string file_path_name);
void LoadVoc(
    std::string voc_path_compressed, char **voc_buffer,
    unsigned int &voc_buf_size);
#endif // ORB_SLAM3_FILEIO_H
