#include "vjo/vjo.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>

int main(int argc, char *argv[]) {
  std::filesystem::path dataDirectory(argv[1]);

  std::ifstream stateFile(dataDirectory / "states.csv");

  std::string firstLine;
  std::getline(stateFile, firstLine);
  std::cout << firstLine << std::endl;

  const int numberOfJoints = 7;

  int imageIndex = 0;

  while (stateFile.good()) {

    gtsam::Vector joints(numberOfJoints);
    for (int jointIndex = 0; jointIndex < numberOfJoints; ++jointIndex) {
      stateFile >> joints(jointIndex);
    }

    const cv::Mat image = cv::imread(
        dataDirectory / ("image" + std::to_string(imageIndex) + ".png"),
        cv::IMREAD_GRAYSCALE);
  }

  return 0;
}
