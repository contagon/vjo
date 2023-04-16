#include "vjo/vjo.hpp"

namespace vjo {

Frontend::Frontend() : m_featureDetector(cv::ORB::create()) {}

void Frontend::registerFirstImage(const cv::Mat &image) {
  m_featureDetector->detectAndCompute(image, cv::noArray(), m_previousKeypoints,
                                      m_previousDescriptors);
}

void Frontend::registerImage(const cv::Mat &image) {
  std::vector<cv::KeyPoint> currentKeypoints;
  cv::Mat currentDescriptors;
  m_featureDetector->detectAndCompute(image, cv::noArray(), currentKeypoints,
                                      currentDescriptors);
}
} // namespace vjo
