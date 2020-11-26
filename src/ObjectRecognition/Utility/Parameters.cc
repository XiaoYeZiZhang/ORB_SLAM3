//
// Created by zhangye on 2020/9/16.
//

#include "Utility/Parameters.h"
#include "mode.h"

Parameters::Parameters() {
    kObjectModelVersion = 0;
    kObjectModelKFHeight = 0;
    kObjectModelKFWidth = 0;

    // tracker need to be more strict
    kTrackerProjectSuccessNumTh = 50;
    kTrackerMatchPointsNumTh = 40;

    kTrackerPnPInliersGoodNumTh = 60;
    kTrackerPnPInliersUnreliableNumTh = 30;

#ifdef OBJECT_BOX
#ifdef SUPERPOINT
    kTrackerPnPInliersGoodNumTh = 150;
    kTrackerPnPInliersGoodNumTh_PoseSolver = 150;
#else
    kTrackerPnPInliersGoodNumTh = 40;
    kTrackerPnPInliersGoodNumTh_PoseSolver = 40;
#endif
#endif

#ifdef OBJECT_BAG
#ifdef SUPERPOINT
    kTrackerPnPInliersGoodNumTh = 120;
    kTrackerPnPInliersGoodNumTh_PoseSolver = 120;
#else
    kTrackerPnPInliersGoodNumTh = 60;
    kTrackerPnPInliersGoodNumTh_PoseSolver = 60;
#endif
#endif

#ifdef OBJECT_TOY
#ifdef SUPERPOINT
    kTrackerPnPInliersGoodNumTh = 80;
    kTrackerPnPInliersGoodNumTh_PoseSolver = 80;
#else
    kTrackerPnPInliersGoodNumTh = 80;
    kTrackerPnPInliersGoodNumTh_PoseSolver = 80;
#endif
#endif

    kDetectorKNNMatchNumTh = 8;
    kDetectorPnPInliersGoodNumTh = 20;
    kDetectorPnPInliersUnreliableNumTh = 8;

    kDetectorPnPInliersGoodWithKFNumTh = 50;
    kDetectorPnPInliersGoodWithKFNumTh_PoseSolver = 50;
    kDetectorPnPInliersUnreliableWithKFNumTh = 20;

#ifdef OBJECT_BOX
#ifdef SUPERPOINT
    kDetectorPnPInliersGoodWithKFNumTh = 130;
    kDetectorPnPInliersGoodWithKFNumTh_PoseSolver = 130;
    kDetectorPnPInliersUnreliableWithKFNumTh = 40;
#else
    kDetectorPnPInliersGoodWithKFNumTh = 90;
    kDetectorPnPInliersGoodWithKFNumTh_PoseSolver = 90;
    kDetectorPnPInliersUnreliableWithKFNumTh = 30;
#endif
#endif

#ifdef OBJECT_TOY
#ifdef SUPERPOINT
    kDetectorPnPInliersGoodWithKFNumTh = 80;
    kDetectorPnPInliersGoodWithKFNumTh_PoseSolver = 80;
    kDetectorPnPInliersUnreliableWithKFNumTh = 30;
#else
    kDetectorPnPInliersGoodWithKFNumTh = 80;
    kDetectorPnPInliersGoodWithKFNumTh_PoseSolver = 80;
    kDetectorPnPInliersUnreliableWithKFNumTh = 30;
#endif
#endif

#ifdef OBJECT_BAG
#ifdef SUPERPOINT
    kDetectorPnPInliersGoodWithKFNumTh = 100;
    kDetectorPnPInliersGoodWithKFNumTh_PoseSolver = 100;
    kDetectorPnPInliersUnreliableWithKFNumTh = 40;
#else
    kDetectorPnPInliersGoodWithKFNumTh = 55;
    kDetectorPnPInliersGoodWithKFNumTh_PoseSolver = 55;
    kDetectorPnPInliersUnreliableWithKFNumTh = 30;
#endif
#endif

    KORBExtractor_scaleFactor = 0.0;
    KORBExtractor_nlevels = 0;
    KORBExtractor_fastInitThreshold = 0;
    KORBExtractor_fastMinThrethold = 0;

    KSPExtractor_scaleFactor = 0.0;
    KSPExtractor_nlevels = 0;
    KSPExtractor_nFeatures = 0;

    KObjRecognitionORB_nFeatures = 0;
}
