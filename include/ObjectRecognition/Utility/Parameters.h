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
    int SetFastInitThreshold(double initThre) {
        KORBExtractor_fastInitThreshold = initThre;
    }
    int SetFastMinThreshold(double minThre) {
        KORBExtractor_fastMinThrethold = minThre;
    }

    void SetSPScaleFactor(double factor) {
        KSPExtractor_scaleFactor = factor;
    }
    void SetSPLevels(int level) {
        KSPExtractor_nlevels = level;
    }
    void SetSPFeatures(int features) {
        KSPExtractor_nFeatures = features;
    }
    void SetObjRecognitionORBFeatures(int features) {
        KObjRecognitionORB_nFeatures = features;
    }

public:
    int kTrackerProjectSuccessNumTh;
    int kTrackerMatchPointsNumTh;
    int kTrackerPnPInliersGoodNumTh;
    int kTrackerPnPInliersGoodNumTh_PoseSolver;
    int kTrackerPnPInliersUnreliableNumTh;

    int kDetectorKNNMatchNumTh;
    int kDetectorPnPInliersGoodNumTh;
    int kDetectorPnPInliersUnreliableNumTh;
    int kDetectorPnPInliersGoodWithKFNumTh;
    int kDetectorPnPInliersGoodWithKFNumTh_PoseSolver;

    int kDetectorPnPInliersUnreliableWithKFNumTh;

    double KORBExtractor_scaleFactor;
    int KORBExtractor_nlevels;
    int KORBExtractor_fastInitThreshold;
    int KORBExtractor_fastMinThrethold;

    double KSPExtractor_scaleFactor;
    int KSPExtractor_nlevels;
    int KSPExtractor_nFeatures;

    int KObjRecognitionORB_nFeatures;
};

#endif // ORB_SLAM3_PARAMETERS_H
