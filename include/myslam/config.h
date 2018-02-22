#ifndef CONFIG_H
#define CONFIG_H

#include "myslam/common_include.h"

namespace myslam
{
    class Config
    {
    private:
        static shared_ptr<Config> config_;
        cv::FileStorage file_;

        Config() {}

    public:
        ~Config();
        static void setParamFile(const string& filename);

        template <typename T>
        static T getParam(const string& paramKey) {
            return T(Config::config_->file_[paramKey]);
        }
    };
}

#endif