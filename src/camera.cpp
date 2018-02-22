#include "myslam/camera.h"
#include "myslam/config.h"

namespace myslam
{
    Camera::Camera() {
        fx_ = Config::getParam<float>("camera.fx");
        fy_ = Config::getParam<float>("camera.fy");
        cx_ = Config::getParam<float>("camera.cx");
        cy_ = Config::getParam<float>("camera.cy");

        depth_sacle_ = Config::getParam<float>("camera.depth_scale");

    }

    Eigen::Vector3d Camera::world2camera(const Eigen::Vector3d &p_w, const Sophus::SE3 &T_c_w) {
        return T_c_w * p_w;
    }

    Eigen::Vector3d Camera::camera2world(const Eigen::Vector3d &p_c, const Sophus::SE3 &T_c_w) {
        return T_c_w.inverse() * p_c;
    }

    Eigen::Vector2d Camera::camera2pixel(const Eigen::Vector3d &p_c) {
        Eigen::Vector2d p_p;
        p_p(0) = fx_ * p_c(0) / p_c(2) + cx_;
        p_p(1) = fy_ * p_c(1) / p_c(2) + cy_;

        return p_p;
    }

    Eigen::Vector3d Camera::pixel2camera(const Eigen::Vector2d &p_p, double depth) {
        Eigen::Vector3d p_c;
        p_c(0) = depth * (p_p(0) - cx_) / fx_;
        p_c(1) = depth * (p_p(1) - cy_) / fy_;
        p_c(2) = depth;

        return p_c;

    }

    Eigen::Vector3d Camera::pixel2world(const Eigen::Vector2d &p_p, const Sophus::SE3 &T_c_w, double depth) {
        return camera2world(pixel2camera(p_p, depth), T_c_w);
    }

    Eigen::Vector2d Camera::world2pixel(const Eigen::Vector3d &p_w, const Sophus::SE3 T_c_w) {
        return camera2pixel(world2camera(p_w, T_c_w));
    }

}