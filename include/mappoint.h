#ifndef MAP_POINT_H
#define MAP_POINT_H

#include "common_include.h"

namespace myslam
{
    class MapPoint
    {
    public:
        typedef std::shared_ptr<MapPoint> Ptr;
        unsigned long id_;
        Eigen::Vector3d pos_;
        Eigen::Vector3d norm_;
        cv::Mat descriptor_;
        int observed_times_;
        int correct_times_;

        MapPoint();
        MapPoint(unsigned long id, Eigen::Vector3d pos, Eigen::Vector3d norm);

        static MapPoint::Ptr createMapPoint();
    };
}

#endif