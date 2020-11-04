//
// Created by zhangye on 2020/9/16.
//

#include "Utility/Parameters.h"

Parameters::Parameters() {
    kObjectModelVersion = 0;
    kObjectModelKFHeight = 0;
    kObjectModelKFWidth = 0;

    // tracker need to be more strict
    kTrackerProjectSuccessNumTh = 50;
    kTrackerMatchPointsNumTh = 40;
    kTrackerPnPInliersGoodNumTh = 50;
    kTrackerPnP3DInliersGoodNumTh = 30;
    kTrackerPnPInliersUnreliableNumTh = 30;

    kDetectorKNNMatchNumTh = 8;
    kDetectorPnPInliersGoodNumTh = 20;
    kDetectorPnPInliersUnreliableNumTh = 8;

    kDetectorPnPInliersGoodWithKFNumTh = 30;
    kDetectorPnP3DInliersGoodWithKFNumTh = 20;
    kDetectorPnPInliersUnreliableWithKFNumTh = 20;

    KORBExtractor_scaleFactor = 0.0;
    KORBExtractor_nlevels = 0;
    KORBExtractor_fastInitThreshold = 0;
    KORBExtractor_fastMinThrethold = 0;
}
