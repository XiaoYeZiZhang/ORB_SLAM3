#include "Utility/Parameters.h"
#include "mode.h"

Parameters::Parameters() {
    // tracker need to be more strict
    kTrackerProjectSuccessNumTh = 50;
    kTrackerMatchPointsNumTh = 40;

    kTrackerPnPInliersGoodNumTh = 60;
    kTrackerPnPInliersUnreliableNumTh = 30;

#ifdef OBJECT_BOX
#ifdef SUPERPOINT
#ifdef MONO
    kTrackerPnPInliersGoodNumTh = 100;
    kTrackerPnPInliersGoodNumTh_PoseSolver = 100;
#else
    kTrackerPnPInliersGoodNumTh = 150;
    kTrackerPnPInliersGoodNumTh_PoseSolver = 150;
#endif
#else
    kTrackerPnPInliersGoodNumTh = 40;
    kTrackerPnPInliersGoodNumTh_PoseSolver = 40;
#endif
#endif

#ifdef OBJECT_BAG
#ifdef SUPERPOINT
#ifdef MONO
    kTrackerPnPInliersGoodNumTh = 60;
    kTrackerPnPInliersGoodNumTh_PoseSolver = 60;
#else
    kTrackerPnPInliersGoodNumTh = 120;
    kTrackerPnPInliersGoodNumTh_PoseSolver = 120;
#endif
#else
    kTrackerPnPInliersGoodNumTh = 60;
    kTrackerPnPInliersGoodNumTh_PoseSolver = 60;
#endif
#endif

#ifdef OBJECT_TOY
#ifdef SUPERPOINT
#ifdef MONO
    kTrackerPnPInliersGoodNumTh = 60;
    kTrackerPnPInliersGoodNumTh_PoseSolver = 60;
#else
    kTrackerPnPInliersGoodNumTh = 80;
    kTrackerPnPInliersGoodNumTh_PoseSolver = 80;
#endif
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
#ifdef MONO
    kDetectorPnPInliersGoodWithKFNumTh = 100;
    kDetectorPnPInliersGoodWithKFNumTh_PoseSolver = 100;
    kDetectorPnPInliersUnreliableWithKFNumTh = 30;
#else
    kDetectorPnPInliersGoodWithKFNumTh = 130;
    kDetectorPnPInliersGoodWithKFNumTh_PoseSolver = 130;
    kDetectorPnPInliersUnreliableWithKFNumTh = 40;
#endif
#else
    kDetectorPnPInliersGoodWithKFNumTh = 90;
    kDetectorPnPInliersGoodWithKFNumTh_PoseSolver = 90;
    kDetectorPnPInliersUnreliableWithKFNumTh = 30;
#endif
#endif

#ifdef OBJECT_TOY
#ifdef SUPERPOINT
#ifdef MONO
    kDetectorPnPInliersGoodWithKFNumTh = 60;
    kDetectorPnPInliersGoodWithKFNumTh_PoseSolver = 60;
    kDetectorPnPInliersUnreliableWithKFNumTh = 30;
#else
    kDetectorPnPInliersGoodWithKFNumTh = 80;
    kDetectorPnPInliersGoodWithKFNumTh_PoseSolver = 80;
    kDetectorPnPInliersUnreliableWithKFNumTh = 30;
#endif
#else
    kDetectorPnPInliersGoodWithKFNumTh = 80;
    kDetectorPnPInliersGoodWithKFNumTh_PoseSolver = 80;
    kDetectorPnPInliersUnreliableWithKFNumTh = 30;
#endif
#endif

#ifdef OBJECT_BAG
#ifdef SUPERPOINT
#ifdef MONO
    kDetectorPnPInliersGoodWithKFNumTh = 80;
    kDetectorPnPInliersGoodWithKFNumTh_PoseSolver = 80;
    kDetectorPnPInliersUnreliableWithKFNumTh = 40;
#else
    kDetectorPnPInliersGoodWithKFNumTh = 100;
    kDetectorPnPInliersGoodWithKFNumTh_PoseSolver = 100;
    kDetectorPnPInliersUnreliableWithKFNumTh = 40;
#endif
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
