#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <boost/timer.hpp>
#include <g2o/core/block_solver.h>

#include "myslam/config.h"
#include "myslam/visual_odometry.h"
#include "myslam/g2o_types.h"

namespace myslam
{
    VisualOdometry::VisualOdometry():
    state_(INITIALIZING), map_(new Map), ref_(nullptr), curr_(nullptr), num_lost_(0), num_inliers_(0)
    {
        num_of_features_ = Config::getParam<int>("number_of_features");
        scale_factor_ = Config::getParam<double>("scale_factor");
        level_pyramid_ = Config::getParam<int>("level_pyramid");
        match_ratio_ = Config::getParam<float>("match_ratio");
        max_num_lost_ = Config::getParam<int>("max_num_lost");
        min_inliers_ = Config::getParam<int>("min_inliers");
        key_frame_min_rot_ = Config::getParam<double>("key_frame_min_rot");
        key_frame_min_trans_ = Config::getParam<double>("key_frame_min_trans");

        orb_ = cv::ORB::create(num_of_features_, scale_factor_, level_pyramid_);
    }

    VisualOdometry::~VisualOdometry() {}

    bool VisualOdometry::addFrame(Frame::Ptr frame) {
        switch (state_) {
            case INITIALIZING:
            {
                state_ = OK;
                curr_ = ref_ = frame;
                map_->insertKeyFrame(frame);
                extractKeyPoints();
                computeDescriptors();
                setRef3DPoints();
                break;
            }
            case OK:
            {
                curr_ = frame;
                extractKeyPoints();
                computeDescriptors();
                featrureMatching();
                poseEstimationPnP();
                cout << "num inliers: " << num_inliers_ << endl;

                if (checkEstimatedPose()) {
                    curr_->T_c_w_ = T_c_r_esti_ * ref_->T_c_w_;
                    //　本帧就算弄完了，把它变成参考帧，供下个帧使用
                    ref_ = curr_;
                    setRef3DPoints();
                    num_lost_ = 0;

                    if (checkKeyFrame()) {
                        addKeyFrame();
                    }
                } else {
                    num_lost_++;
                    if (num_lost_ > max_num_lost_) {
                        state_ = LOST;
                    }
                    return false;
                }
                break;
            }
            case LOST:
            {
                break;
            }
        }
        return true;
    }

    void VisualOdometry::extractKeyPoints() {
        orb_->detect(curr_->color_, keypoints_curr_);
    }

    void VisualOdometry::computeDescriptors() {
        orb_->compute(curr_->color_, keypoints_curr_, descriptors_curr_);
    }

    void VisualOdometry::featrureMatching() {
        vector<cv::DMatch> matches;
        cv::BFMatcher matcher(cv::NORM_HAMMING);
        matcher.match(descriptors_ref_, descriptors_curr_, matches);

        double min_dis = 999999999.0;

        for (cv::DMatch& match : matches) {
            if (match.distance < min_dis) {
                min_dis = match.distance;
            }
        }

        features_matches_.clear();

        for (cv::DMatch& match: matches) {
            if (match.distance < max<float>(min_dis * match_ratio_, 30.0)) {
                features_matches_.push_back(match);
            }
        }

        cout << "good matches: " << features_matches_.size() << endl;

    }

    void VisualOdometry::poseEstimationPnP() {
        vector<cv::Point3f> pts3d;
        vector<cv::Point2f> pts2d;

        for (cv::DMatch match : features_matches_) {
            pts3d.push_back(pts_3d_ref_[match.queryIdx]);
            pts2d.push_back(keypoints_curr_[match.trainIdx].pt);
        }
        cv::Mat K = (cv::Mat_<double>(3, 3)
                << ref_->camera_->fx_, 0, ref_->camera_->cx_,
                0, ref_->camera_->fy_, ref_->camera_->cy_,
                0, 0, 1.0);
        cv::Mat rvec, tvec, inliers;
        cv::solvePnPRansac(pts3d, pts2d, K, cv::Mat(), rvec, tvec, false, 100, 4.0, 0.99, inliers);
        num_inliers_ = inliers.rows;
        T_c_r_esti_ = Sophus::SE3(
                Sophus::SO3(rvec.at<double>(0, 0), rvec.at<double>(1, 0), rvec.at<double>(2, 0)),
                Eigen::Vector3d(tvec.at<double>(0, 0), tvec.at<double>(1, 0), tvec.at<double>(2, 0))
        );

        //　定义块求解器类型
        typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,2>> Block;
        // 选择块求解器所使用的求解方式，稠密还是稀疏
        Block::LinearSolverType* linearSolver = new g2o::LinearSolverDense<Block::PoseMatrixType>();
        // 实例化一个块求解器指针
        Block* blockSolverPtr = new Block(linearSolver);

        //　设置所用优化算法
        g2o::OptimizationAlgorithmLevenberg* optiAlgorithm = new g2o::OptimizationAlgorithmLevenberg(blockSolverPtr);

        //　创建优化问题
        g2o::SparseOptimizer optimizer;

        //　给优化问题设置上刚刚选好的优化算法
        optimizer.setAlgorithm(optiAlgorithm);

        // 下面就是给优化问题添加顶点和边

        // 创建顶点,　该问题就一个顶点，即相机的相对参考帧的位姿
        g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();
        pose->setId(0);
        // 设置优化初始值
        pose->setEstimate(g2o::SE3Quat(T_c_r_esti_.rotation_matrix(), T_c_r_esti_.translation()));

        // 将该点加入优化问题中
        optimizer.addVertex(pose);

        // 添加边
        for (int i = 0; i < inliers.rows; i++) {
            // 创建边
            EdgeProjXYZ2UVPoseOnly* edge = new EdgeProjXYZ2UVPoseOnly();
            edge->setId(i);
            edge->setVertex(0, pose);

            int index = inliers.at<int>(i, 0);

            edge->setMeasurement(Eigen::Vector2d(pts2d[index].x, pts2d[index].y));
            edge->camera_ = curr_->camera_.get();
            edge->point_ = Eigen::Vector3d(pts3d[index].x, pts3d[index].y, pts3d[index].z);
            edge->setInformation(Eigen::Matrix2d::Identity());

            optimizer.addEdge(edge);
        }

        //　开始优化
        optimizer.initializeOptimization();
        optimizer.optimize(10);

        T_c_r_esti_ = Sophus::SE3(pose->estimate().rotation(), pose->estimate().translation());
    }

    void VisualOdometry::setRef3DPoints() {
        pts_3d_ref_.clear();
        descriptors_ref_ = cv::Mat();

        for (size_t i = 0; i < keypoints_curr_.size(); i++) {
            //　查询深度
            double d = ref_->findDepth(keypoints_curr_[i]);
            if (d > 0) {
                Eigen::Vector3d p_cam = ref_->camera_->pixel2camera(Eigen::Vector2d(keypoints_curr_[i].pt.x, keypoints_curr_[i].pt.y), d);
                pts_3d_ref_.push_back(cv::Point3f(p_cam(0, 0), p_cam(1, 0), p_cam(2, 0)));
                descriptors_ref_.push_back(descriptors_curr_.row(i));
            }

        }
    }

    bool VisualOdometry::checkEstimatedPose() {
        if (num_inliers_ < min_inliers_) {
            return false;
        }

        Sophus::Vector6d tcr_vec = T_c_r_esti_.log();
        if (tcr_vec.norm() > 5.0) {
            return false;
        }

        return true;
    }

    bool VisualOdometry::checkKeyFrame() {
        Sophus::Vector6d tcr_vec = T_c_r_esti_.log();
        Eigen::Vector3d trans = tcr_vec.head<3>();
        Eigen::Vector3d rot = tcr_vec.tail<3>();

        if (trans.norm() < key_frame_min_trans_ && rot.norm() < key_frame_min_rot_) {
            return false;
        }

        return true;
    }

    void VisualOdometry::addKeyFrame() {
        map_->insertKeyFrame(curr_);
    }
}