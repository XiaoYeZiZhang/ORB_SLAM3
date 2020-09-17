#include "ORBExtractor.h"
#include "ORBExtractorHPC.h"

namespace SLAMCommon {

static orb_optimization_mode_t GetActualMode(orb_optimization_mode_t srcMode) {
    orb_optimization_mode_t result = srcMode;
    if (AUTO_OPTIMIZATION == srcMode) {
#if defined(ANDROID) || defined(__ANDOIRD__)
        result = HPC_MOBILE_OPTIMIZATION;
#else
        result = NORMAL_OPTIMIZATION;
#endif
    }

    return result;
}

std::shared_ptr<ORBExtractor> MakeORBExtractor(
    orb_optimization_mode_t mode, int nfeatures, float scaleFactor, int nlevels,
    int iniThFAST, int minThFAST) {
    std::shared_ptr<ORBExtractor> result;

    switch (GetActualMode(mode)) {
    case NORMAL_OPTIMIZATION:
        result = std::make_shared<ORBExtractor>(
            nfeatures, scaleFactor, nlevels, iniThFAST, minThFAST);
        break;
#if defined(ANDROID) || defined(__ANDOIRD__)
    case HPC_MOBILE_OPTIMIZATION:
        result = std::make_shared<ORBExtractorHPC>(
            nfeatures, scaleFactor, nlevels, iniThFAST, minThFAST);
        break;
    case HEXAGON_DSP_OPTIMIZATION:
        // result = std::make_shared<ORBExtractorHexagon>(nfeatures,
        // scaleFactor, nlevels, iniThFAST, minThFAST);
        break;
#endif
    default:
        break;
    }

    return result;
}
} // namespace SLAMCommon