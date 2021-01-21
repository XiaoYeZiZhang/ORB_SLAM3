#include <iostream>
#include <iomanip>
#include "ORBSLAM3/SPextractor.h"
#include "Parameters.h"
#include "GlobalImageViewer.h"
#include "Statistics.h"
#include "Timer.h"
#include "ObjectRecognitionSystem.h"
#include "ORBExtractor.h"
#include "mode.h"

namespace ObjRecognition {
ObjRecogThread::ObjRecogThread() {
    m_SPextractor = new ORB_SLAM3::SPextractor(
        64, Parameters::GetInstance().KSPExtractor_nFeatures, 1.2,
        Parameters::GetInstance().KSPExtractor_nlevels, 0.015, 0.007, true);
}

int ObjRecogThread::Init() {
    m_pointcloudobj_detector =
        std::make_shared<ObjRecognition::PointCloudObjDetector>();
    m_detector_thread.SetDetector(m_pointcloudobj_detector);

    m_pointcloudobj_tracker =
        std::make_shared<ObjRecognition::PointCloudObjTracker>();
    m_tracker_thread.SetTracker(m_pointcloudobj_tracker);

    if (!m_detector_thread.StartThread()) {
        VLOG(0) << "create detect thread failed";
        return -1;
    }

    if (!m_tracker_thread.StartThread()) {
        VLOG(0) << "create tracker thread failed";
        return -1;
    }

    m_pointcloudobj_detector->SetVoc(m_voc);
    return 0;
}

int ObjRecogThread::SetVocabulary(
    const std::shared_ptr<DBoW3::Vocabulary> &voc) {
    m_voc = voc;
    return 0;
}

int ObjRecogThread::SetModel(const std::shared_ptr<Object> &object) {
    m_object = object;

#ifdef USE_NO_VOC_FOR_OBJRECOGNITION_SUPERPOINT
#else
    std::shared_ptr<DBoW3::Database> database =
        std::make_shared<DBoW3::Database>(m_voc, true, 4);
    m_object->SetDatabase(database);

    VLOG(0) << "PointCloud detector database create success ";
    m_object->GetDatabase()->size();
#endif

    auto allKFs = m_object->GetKeyFrames();

    TIMER_UTILITY::Timer timer;
    m_object->AddKeyFrames2Database(allKFs);
    STATISTICS_UTILITY::StatsCollector createDatabase(
        "Time: create keyframes database");
    createDatabase.AddSample(timer.Stop());

    m_pointcloudobj_detector->SetPointCloudObj(m_object);
    m_pointcloudobj_tracker->SetPointCloudObj(m_object);
    VLOG(0) << "tracker and detector thread load object " << m_object->GetId();
    return 0;
}

// TODO(zhangye) check the bug
void ObjRecogThread::GetNewestData() {
}

void ObjRecogThread::GetResult(
    FrameIndex &frmIndex, double &timeStamp, ObjRecogState &state,
    Eigen::Matrix3d &R_cam, Eigen::Vector3d &t_cam, Eigen::Matrix3d &Rwo,
    Eigen::Vector3d &two) {

    if (m_object != nullptr) {
        m_object->GetPose(frmIndex, timeStamp, state, R_cam, t_cam, Rwo, two);

    } else {
        frmIndex = 0;
        timeStamp = 0;
        state = ObjRecogState::TrackingBad;
    }
}

template <typename T>
std::string to_string_with_precision(const T a_value, const int n = 6) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(n) << a_value;
    return out.str();
}

void ObjRecogThread::Process() {
    if (m_object == nullptr) {
        return;
    }
    {
        std::lock_guard<std::mutex> lck(m_input_mutex);
        {
            if (m_curData) {
                last_processed_frame = (long int)m_curData->id;
            }

            if (!m_input_queue.empty()) {
                m_curData = m_input_queue[0];
                m_input_queue.clear();
            }

            if (m_curData) {
                if ((long int)m_curData->id == last_processed_frame) {
                    return;
                }
            }
        }
    }

    std::shared_ptr<void> frame_tmp = m_curData;
    if (!frame_tmp) {
        return;
    }

    std::shared_ptr<CallbackFrame> callback_frame =
        std::static_pointer_cast<CallbackFrame>(frame_tmp);

    std::shared_ptr<FrameForObjRecognition> cur_frame =
        std::make_shared<FrameForObjRecognition>();

    cur_frame->m_frmIndex = callback_frame->id;
    cur_frame->m_img = cv::Mat(
        callback_frame->height, callback_frame->width, CV_8UC1,
        callback_frame->data);

#ifdef ORBPOINT
    TIMER_UTILITY::Timer timer_orb;
    SLAMCommon::ORBExtractor orb_extractor(
        Parameters::GetInstance().KObjRecognitionORB_nFeatures, 1.2f, 8, 20, 7);
    orb_extractor.DetectKeyPoints(cur_frame->m_img, cur_frame->m_kpts);
    orb_extractor.ComputeDescriptors(
        cur_frame->m_img, cur_frame->m_kpts, cur_frame->m_desp);
    STATISTICS_UTILITY::StatsCollector ORB_objRecognition(
        "Time: ORB extractor for objRecognition");
    ORB_objRecognition.AddSample(timer_orb.Stop());
#endif

#ifdef SUPERPOINT
    TIMER_UTILITY::Timer timer_superpoint;
    (*m_SPextractor)(
        cur_frame->m_img, cv::Mat(), cur_frame->m_kpts, cur_frame->m_desp);
    STATISTICS_UTILITY::StatsCollector SUPERPOINT_objRecognition(
        "Time: SUPERPOINT extractor for objRecognition");
    SUPERPOINT_objRecognition.AddSample(timer_superpoint.Stop());
#endif

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            cur_frame->m_Rcw(i, j) = callback_frame->R[i][j];
        }
        cur_frame->m_tcw(i) = callback_frame->t[i];
    }

    m_detector_thread.PushData(cur_frame);
    m_tracker_thread.PushData(cur_frame);
}

void ObjRecogThread::Reset() {
    if (m_object) {
        m_object->Reset();
    }

    m_detector_thread.RequestReset();
    m_detector_thread.WaitEndReset();
    m_tracker_thread.RequestReset();
    m_tracker_thread.WaitEndReset();
}

void ObjRecogThread::Stop() {
    m_detector_thread.RequestStop();
    m_tracker_thread.RequestStop();
    m_detector_thread.WaitEndStop();
    m_tracker_thread.WaitEndStop();
}
} // namespace ObjRecognition