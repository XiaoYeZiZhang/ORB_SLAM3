//
// Created by zhangye on 2020/9/15.
//
#include <iostream>
#include <iomanip>
#include <include/ORBSLAM3/SPextractor.h>
#include "Utility/Thread/ThreadBase.h"
#include "Utility/Parameters.h"
#include "Visualizer/GlobalImageViewer.h"
#include "Utility/Statistics.h"
#include "ObjectRecognitionSystem/ObjectRecognitionSystem.h"
#include "Utility/FeatureExtractor/ORBExtractor.h"
#include "mode.h"

namespace ObjRecognition {

ObjRecogThread::ObjRecogThread() : ThreadBase(1, false) {
    SPextractor = new ORB_SLAM3::SPextractor(
        Parameters::GetInstance().KSPExtractor_nFeatures, 1.2,
        Parameters::GetInstance().KSPExtractor_nlevels, 0.015, 0.007, true);
}

int ObjRecogThread::Init() {
    pointcloudobj_detector_ =
        std::make_shared<ObjRecognition::PointCloudObjDetector>();
    detector_thread_.SetDetector(pointcloudobj_detector_);

    pointcloudobj_tracker_ =
        std::make_shared<ObjRecognition::PointCloudObjTracker>();
    tracker_thread_.SetTracker(pointcloudobj_tracker_);

    if (!detector_thread_.StartThread()) {
        VLOG(0) << "create detect thread failed";
        return -1;
    }

    if (!tracker_thread_.StartThread()) {
        VLOG(0) << "create tracker thread failed";
        return -1;
    }

    pointcloudobj_detector_->SetVoc(voc_);
    VLOG(10) << "Detector set voc success";

    return 0;
}

int ObjRecogThread::SetVocabulary(
    const std::shared_ptr<DBoW3::Vocabulary> &voc) {
    voc_ = voc;
    return 0;
}

int ObjRecogThread::SetModel(const std::shared_ptr<Object> &object) {
    object_ = object;

    std::shared_ptr<DBoW3::Database> database =
        std::make_shared<DBoW3::Database>(voc_, true, 4);
    object_->SetDatabase(database);

    VLOG(0) << "PointCloud detector database create success ";
    object_->GetDatabase()->size();

    auto allKFs = object_->GetKeyFrames();
    object_->AddKeyFrames2Database(allKFs);

    pointcloudobj_detector_->SetPointCloudObj(object_);
    pointcloudobj_tracker_->SetPointCloudObj(object_);
    VLOG(0) << "tracker and detector thread load object " << object_->GetId();
    return 0;
}

void ObjRecogThread::PushUnProcessedFrame(
    const std::shared_ptr<ObjRecogFrameCallbackData> &frame) {
    PushData(DATA_TYPE_UNPROCESSED_FRAME, frame);
}

void ObjRecogThread::GetResult(
    FrameIndex &frmIndex, double &timeStamp, ObjRecogState &state,
    Eigen::Matrix3d &R_cam, Eigen::Vector3d &t_cam, Eigen::Matrix3d &Rwo,
    Eigen::Vector3d &two) {

    if (object_ != nullptr) {
        object_->GetPose(frmIndex, timeStamp, state, R_cam, t_cam, Rwo, two);

    } else {
        frmIndex = 0;
        timeStamp = 0;
        state = ObjRecogState::TrackingBad;
    }
}

int ObjRecogThread::GetInfo(std::string &info) {
    {
        std::unique_lock<std::mutex> lock(mMutexInfoBuffer);
        info = info_;
    }
    return 0;
}

template <typename T>
std::string to_string_with_precision(const T a_value, const int n = 6) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(n) << a_value;
    return out.str();
}

void ObjRecogThread::SetInfo() {
    std::string info;

    info +=
        "frame processed num: " + std::to_string(frame_processed_num_) + '\n';

    pointcloudobj_detector_->GetInfo(info);
    pointcloudobj_tracker_->GetInfo(info);

    FrameIndex frmIndex;
    double timeStamp;
    ObjRecogState state;
    Eigen::Matrix3d Rcw, R_obj;
    Eigen::Vector3d Tcw, T_obj;
    int obj_num = 0;

    object_->GetPose(frmIndex, timeStamp, state, Rcw, Tcw, R_obj, T_obj);
    if (state == TrackingGood) {
        STATISTICS_UTILITY::StatsCollector pointCloudFinalStateNum(
            "finalState good num");
        pointCloudFinalStateNum.IncrementOne();
        obj_num = 1;
    }
    info += "obj num: " + std::to_string(obj_num) + '\n';

    info += "t_cam: ";
    {
        Eigen::Vector3f t = Tcw.cast<float>();
        info += "[" + to_string_with_precision(t[0], 3) + " " +
                to_string_with_precision(t[1], 3) + " " +
                to_string_with_precision(t[2], 3) + "]" + "\n";
    }

    info += "q_cam: ";
    {
        Eigen::Quaternionf q(Rcw.cast<float>());
        info += "[" + to_string_with_precision(q.x(), 3) + " " +
                to_string_with_precision(q.y(), 3) + " " +
                to_string_with_precision(q.z(), 3) + " " +
                to_string_with_precision(q.w(), 3) + "]" + "\n";
    }

    info += "t_obj: ";
    {
        Eigen::Vector3f t = T_obj.cast<float>();
        info += "[" + to_string_with_precision(t[0], 3) + " " +
                to_string_with_precision(t[1], 3) + " " +
                to_string_with_precision(t[2], 3) + "]" + "\n";
    }

    info += "q_obj: ";
    {
        Eigen::Quaternionf q(R_obj.cast<float>());
        info += "[" + to_string_with_precision(q.x(), 3) + " " +
                to_string_with_precision(q.y(), 3) + " " +
                to_string_with_precision(q.z(), 3) + " " +
                to_string_with_precision(q.w(), 3) + "]" + "\n";
    }

    info += "objrecog buffer size: " + std::to_string(Size()) + "\n";
    info += "detector buffer size: " + std::to_string(detector_thread_.Size()) +
            "\n";
    info +=
        "tracker buffer size: " + std::to_string(tracker_thread_.Size()) + "\n";

    {
        std::unique_lock<std::mutex> lock(mMutexInfoBuffer);
        info_.clear();
        info_ = info + "\0";
    }
}

int ObjRecogThread::Process() {
    int ret = -1;

    if (object_ == nullptr) {
        VLOG(10) << "the object model is empty";
        return ret;
    }

    std::shared_ptr<void> frame_tmp;
    int type = PopFront(frame_tmp);

    if (DATA_TYPE_UNPROCESSED_FRAME != type) {
        VLOG(10) << "data type is not right";
        return ret;
    }

    VLOG(10) << "ObjRecogThread Process";

    std::shared_ptr<ObjRecogFrameCallbackData> platformFrame =
        std::static_pointer_cast<ObjRecogFrameCallbackData>(frame_tmp);

    // STSLAMCommon::Timer timer("raw data process");

    std::shared_ptr<FrameData> cur_frame = std::make_shared<FrameData>();

    cur_frame->mTimeStamp = platformFrame->timestamp;
    cur_frame->mFrmIndex = platformFrame->id;
    if (platformFrame->img.height <= 0 || platformFrame->img.width <= 0)
        return ret;

    cur_frame->img = cv::Mat(
        platformFrame->img.height, platformFrame->img.width, CV_8UC1,
        platformFrame->img.data);

#ifdef ORBPOINT
    // STSLAMCommon::Timer ORBExtractorTimer("ORBExtractor process");
    SLAMCommon::ORBExtractor orb_extractor(
        Parameters::GetInstance().KObjRecognitionORB_nFeatures, 1.2f, 8, 20, 7);
    orb_extractor.DetectKeyPoints(cur_frame->img, cur_frame->mKpts);
    orb_extractor.ComputeDescriptors(
        cur_frame->img, cur_frame->mKpts, cur_frame->mDesp);
#endif

#ifdef SUPERPOINT

    (*SPextractor)(
        cur_frame->img, cv::Mat(), cur_frame->mKpts, cur_frame->mDesp);
    VLOG(0) << "Superpoint per frame: " << cur_frame->mKpts.size();
#endif

    // VLOG(10) << "ORBExtractor process time: " << ORBExtractorTimer.Stop();
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            cur_frame->mRcw(i, j) = platformFrame->R[i][j];
        }
        cur_frame->mTcw(i) = platformFrame->t[i];
    }

    //    GlobalOcvViewer::UpdateView(
    //        "Frame for Detector and Tracker", cur_frame->img);
    detector_thread_.PushData(cur_frame);
    tracker_thread_.PushData(cur_frame);
    ret = 0;

    frame_processed_num_++;

    SetInfo();

    // timer.Stop();
    return ret;
}

int StatisticsPrint() {
    return 0;
}

int ObjRecogThread::Reset() {
    //    StatisticsPrint();
    frame_processed_num_ = 0;
    {
        std::unique_lock<std::mutex> lock(mMutexInfoBuffer);
        info_.clear();
    }

    if (object_) {
        object_->Reset();
    }

    detector_thread_.RequestReset();
    detector_thread_.WaitEndReset();
    tracker_thread_.RequestReset();
    tracker_thread_.WaitEndReset();

    return 0;
}

int ObjRecogThread::Stop() {
    detector_thread_.RequestStop();
    tracker_thread_.RequestStop();
    detector_thread_.WaitEndStop();
    tracker_thread_.WaitEndStop();
    return 0;
}

} // namespace ObjRecognition