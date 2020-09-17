//
// Created by zhangye on 2020/9/16.
//

#ifndef ORB_SLAM3_PARAMETERS_H
#define ORB_SLAM3_PARAMETERS_H
class Parameters {
public:
    Parameters();

    static Parameters &GetInstance() {
        static Parameters instance;
        return instance;
    }

public:
    int kObjectModelVersion;
    int kObjectModelKFWidth;
    int kObjectModelKFHeight;

    int kTrackerProjectSuccessNumTh;
    int kTrackerMatchPointsNumTh;
    int kTrackerPnPInliersGoodNumTh;
    int kTrackerPnPInliersUnreliableNumTh;

    int kDetectorKNNMatchNumTh;
    int kDetectorPnPInliersGoodNumTh;
    int kDetectorPnPInliersUnreliableNumTh;
    int kDetectorPnPInliersGoodWithKFNumTh;
    int kDetectorPnPInliersUnreliableWithKFNumTh;
};

#endif // ORB_SLAM3_PARAMETERS_H
