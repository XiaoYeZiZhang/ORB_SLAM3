//
// Created by zhangye on 2020/9/21.
//

#ifndef ORB_SLAM3_MODE_H
#define ORB_SLAM3_MODE_H

// mode
#define SCANNER
//#define OBJECTRECOGNITION

// feature point mode
#define SUPERPOINT
#define USE_NO_VOC_FOR_OBJRECOGNITION_SUPERPOINT
#define USE_NO_VOC_FOR_SCAN_SFM

//#define ORBPOINT

// for expirements
#define SAVE_CONNECT_FOR_DETECTOR // need more info for scanner
#define USE_CONNECT_FOR_DETECTOR
//#define USE_OLNY_SCAN_MAPPOINT
//#define USE_NO_OPTICALFLOW_FOR_TRACKER
//#define USE_NO_EXTRA_ORB_EXTRACT
//#define USE_NO_METHOD_FOR_FUSE

//#define OBJECT_TOY

#endif // ORB_SLAM3_MODE_H
