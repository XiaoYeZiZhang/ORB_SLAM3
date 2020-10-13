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

    void SetScaleFactor(double factor) {
        KORBExtractor_scaleFactor = factor;
    }
    void SetLevels(int level) {
        KORBExtractor_nlevels = level;
    }
    int SetFastInit(double init) {
        KORBExtractor_fastInit = init;
    }
    int SetFastThreathold(double threashold) {
        KORBExtractor_fastThreathold = threashold;
    }

public:
    int kObjectModelVersion;
    int kObjectModelKFWidth;
    int kObjectModelKFHeight;

    int kTrackerProjectSuccessNumTh;
    int kTrackerMatchPointsNumTh;
    int kTrackerPnPInliersGoodNumTh;
    int kTrackerPnP3DInliersGoodNumTh;
    int kTrackerPnPInliersUnreliableNumTh;

    int kDetectorKNNMatchNumTh;
    int kDetectorPnPInliersGoodNumTh;
    int kDetectorPnPInliersUnreliableNumTh;
    int kDetectorPnPInliersGoodWithKFNumTh;
    int kDetectorPnP3DInliersGoodWithKFNumTh;
    int kDetectorPnPInliersUnreliableWithKFNumTh;

    double KORBExtractor_scaleFactor;
    int KORBExtractor_nlevels;
    int KORBExtractor_fastInit;
    int KORBExtractor_fastThreathold;
};

#endif // ORB_SLAM3_PARAMETERS_H
