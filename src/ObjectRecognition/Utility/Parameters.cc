//
// Created by zhangye on 2020/9/16.
//

#include "Utility/Parameters.h"

Parameters::Parameters() {
    kObjectModelVersion = 0;
    kObjectModelKFHeight = 0;
    kObjectModelKFWidth = 0;

    kTrackerProjectSuccessNumTh = 50;
    kTrackerMatchPointsNumTh = 40;
    kTrackerPnPInliersGoodNumTh = 35;
    kTrackerPnPInliersUnreliableNumTh = 20;
    kDetectorKNNMatchNumTh = 8;
    kDetectorPnPInliersGoodNumTh = 20;
    kDetectorPnPInliersUnreliableNumTh = 8;
    kDetectorPnPInliersGoodWithKFNumTh = 30;
    kDetectorPnPInliersUnreliableWithKFNumTh = 20;
}
