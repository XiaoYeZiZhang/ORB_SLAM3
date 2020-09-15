#include "include/ObjectRecognition/Utility/Thread/Mutex.h"

#ifdef RECORD_LOCK
DEFINE_RUN_MODE(Common::deterministic_multi_thread::kRecord)
#else
DEFINE_RUN_MODE(Common::deterministic_multi_thread::kReplay)
#endif
