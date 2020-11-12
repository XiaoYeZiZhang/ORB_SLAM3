//
// Created by zhangye on 2020/9/16.
//

#ifndef ORB_SLAM3_RECOGNITIONBASE_H
#define ORB_SLAM3_RECOGNITIONBASE_H
#include <memory>
#include "Struct/Frame.h"
namespace ObjRecognition {
class RecognitionBase {
public:
    RecognitionBase() {
    }
    virtual ~RecognitionBase() {
    }

    virtual void Process(const std::shared_ptr<FrameData> &frm) = 0;
    virtual void Reset() = 0;
    virtual void Clear() = 0;
    virtual bool Load(const long long &mem_size, const char *mem) = 0;
    virtual bool Save(long long &mem_size, char **mem) = 0;
};
} // namespace ObjRecognition

#endif // ORB_SLAM3_RECOGNITIONBASE_H
