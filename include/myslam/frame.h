#ifndef FRAME_H
#define FRAME_H

#include "myslam/common_include.h"
#include "myslam/camera.h"

namespace myslam
{
    class Frame
    {
    public:
        typedef std::shared_ptr<Frame> Ptr;
        unsigned long id_; // 帧的id
        double time_stamp_;
        Sophus::SE3 T_c_w_;
        Camera::Ptr camera_;
        cv::Mat color_, depth_;

    public:
        Frame();

        Frame(unsigned long id, double time_stamp = 0, Sophus::SE3 T_c_w = Sophus::SE3(), Camera::Ptr camera = nullptr, cv::Mat color = cv::Mat(), cv::Mat depth = cv::Mat());

        ~Frame();

        static Frame::Ptr createFrame();

        double findDepth(const cv::KeyPoint& kp);

        Eigen::Vector3d getCameraCenter() const;

        bool isInFrame(const Eigen::Vector3d& p_w);
    };
}

#endif