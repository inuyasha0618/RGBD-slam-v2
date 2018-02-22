#include "myslam/mappoint.h"

namespace myslam
{
    MapPoint::MapPoint() {}

    MapPoint::MapPoint(unsigned long id, Eigen::Vector3d pos, Eigen::Vector3d norm):
    id_(id), pos_(pos), norm_(norm) {}

    MapPoint::Ptr MapPoint::createMapPoint() {
        static unsigned long id = 0;
        return MapPoint::Ptr(new MapPoint(id++, Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0, 0, 0)));
    }
}