#pragma once
#include <gtsam/base/Vector.h>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>

namespace vjo {
class Frontend {
public:
  cv::Ptr<cv::Feature2D> m_featureDetector;
  std::vector<cv::KeyPoint> m_previousKeypoints;
  cv::Mat m_previousDescriptors;

  Frontend();

  void registerFirstImage(const cv::Mat &image);
  void registerImage(const cv::Mat &image);
};

} // namespace vjo
