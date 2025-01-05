#include <opencv2/opencv.hpp>

#include <dirent.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "feature_extraction.h"
#include "morphologicalOperations.h"
#include "thresholding.h"

bool isBinaryImage(const cv::Mat &src) {
  for (int i = 0; i < src.rows; i++) {
    for (int j = 0; j < src.cols; j++) {
      uchar pixel = src.at<uchar>(i, j);
      if (pixel != 0 && pixel != 255) {
        return false;
      }
    }
  }
  return true;
}

cv::Mat labelConnectedComponents(const cv::Mat &src, int minSize,
                                 cv::Mat &stats) {
  cv::Mat labels, centroids, invertedSrc;

  // Check if src is empty
  if (src.empty()) {
    std::cerr << "Error: The source image is empty!" << std::endl;
    return labels;
  }

  if (src.channels() != 1) {
    std::cerr << "Error: The source image is not single-channel." << std::endl;
    return labels;
  }

  if (!isBinaryImage(src)) {
    std::cerr << "Error: The source image is not binary." << std::endl;
    return labels;
  }

  // Invert the binary image
  cv::bitwise_not(src, invertedSrc);

  // Connected components analysis
  int nLabels =
      cv::connectedComponentsWithStats(invertedSrc, labels, stats, centroids);

  // Check boundaries
  if (nLabels > stats.rows || nLabels > centroids.rows) {
    std::cerr << "Error: Mismatch between nLabels and stats/centroids rows."
              << std::endl;
    return labels;
  }

  for (int i = 1; i < nLabels; i++) {
    if (stats.at<int>(i, cv::CC_STAT_AREA) < minSize) {
      for (int j = 0; j < src.rows; j++) {
        for (int k = 0; k < src.cols; k++) {
          if (labels.at<int>(j, k) == i) {
            labels.at<int>(j, k) =
                0; // set the label to 0 (background) for regions below minSize
          }
        }
      }
    }
  }
  return labels;
}

cv::Mat colorConnectedComponents(const cv::Mat &labels) {
  // A vector to hold the colors for each label
  std::vector<cv::Vec3b> colors;

  // Find the maximum label value in the labels image.
  double minVal, maxVal;
  cv::minMaxLoc(labels, &minVal, &maxVal);

  // Generate a random color for each label
  for (int i = 0; i <= maxVal; i++) {
    colors.push_back(cv::Vec3b(rand() & 255, rand() & 255, rand() & 255));
  }

  // Set the color for the background (using top-left pixel as reference for
  // background) to black
  int backgroundLabel = labels.at<int>(0, 0);
  colors[backgroundLabel] = cv::Vec3b(0, 0, 0);

  // Create a new image to store the colored version of the labels image
  cv::Mat colored = cv::Mat(labels.size(), CV_8UC3);

  // Populate the colored image using the colors vector
  for (int i = 0; i < labels.rows; i++) {
    for (int j = 0; j < labels.cols; j++) {
      int label = labels.at<int>(i, j);
      colored.at<cv::Vec3b>(i, j) = colors[label];
    }
  }

  return colored;
}

// Function to hold training database
// Data structure to hold feature vectors and labels
struct ObjectData {
  std::string label;
  RegionFeatures features;
  float additionalFeatures[16];
};

void saveObjectDB(const std::string &filename,
                  const std::vector<ObjectData> &objectDB) {
  // Get the directory path
  size_t last_slash = filename.find_last_of("/\\");
  if (last_slash != std::string::npos) {
    std::string dir = filename.substr(0, last_slash);
    // Create directory if it doesn't exist
    if (system(("mkdir -p " + dir).c_str()) != 0) {
      std::cerr << "Warning: Could not create directory: " << dir << std::endl;
    }
  }

  std::ofstream outFile(filename, std::ios::out);
  if (!outFile) {
    std::cerr << "Error: Couldn't open file for writing: " << filename
              << std::endl;
    return;
  }

  // Write each object's data
  for (const ObjectData &data : objectDB) {
    // Write label with comma separator
    outFile << data.label << ",";

    // Write features with space separators
    outFile << data.features.orientedBoundingBox.center.x << " "
            << data.features.orientedBoundingBox.center.y << " "
            << data.features.orientedBoundingBox.size.width << " "
            << data.features.orientedBoundingBox.size.height << " "
            << data.features.orientedBoundingBox.angle << " "
            << data.features.axisOfLeastMoment[0] << " "
            << data.features.axisOfLeastMoment[1] << " "
            << data.features.orthogonalVector[0] << " "
            << data.features.orthogonalVector[1] << " "
            << data.features.percentFilled << " "
            << data.features.bboxAspectRatio;

    // Write additional features
    for (int i = 0; i < 16; i++) {
      outFile << " " << data.features.huMoments[i];
    }

    outFile << std::endl;
  }

  outFile.close();
  std::cout << "Saved object data to: " << filename << std::endl;
}

// Function to define the scaled Euclidean distance metric
float scaledEuclideanDistance(const RegionFeatures &f1,
                              const RegionFeatures &f2) {
  float stdev_orientedBoundingBox_center_x = 1.0;
  float stdev_orientedBoundingBox_center_y = 1.0;
  float stdev_orientedBoundingBox_size_width = 1.0;
  float stdev_orientedBoundingBox_size_height = 1.0;
  float stdev_orientedBoundingBox_angle = 1.0;
  float stdev_axisOfLeastMoment_0 = 1.0;
  float stdev_axisOfLeastMoment_1 = 1.0;
  float stdev_percentFilled = 1.0;
  float stdev_bboxAspectRatio = 1.0;
  float distance = 0.0;
  float distance_center_x = std::pow(
      (f1.orientedBoundingBox.center.x - f2.orientedBoundingBox.center.x) /
          stdev_orientedBoundingBox_center_x,
      2);
  float distance_center_y = std::pow(
      (f1.orientedBoundingBox.center.y - f2.orientedBoundingBox.center.y) /
          stdev_orientedBoundingBox_center_y,
      2);
  float distance_width = std::pow(
      (f1.orientedBoundingBox.size.width - f2.orientedBoundingBox.size.width) /
          stdev_orientedBoundingBox_size_width,
      2);
  float distance_height = std::pow((f1.orientedBoundingBox.size.height -
                                    f2.orientedBoundingBox.size.height) /
                                       stdev_orientedBoundingBox_size_height,
                                   2);
  float distance_angle =
      std::pow((f1.orientedBoundingBox.angle - f2.orientedBoundingBox.angle) /
                   stdev_orientedBoundingBox_angle,
               2);
  float distance_axis_0 =
      std::pow((f1.axisOfLeastMoment[0] - f2.axisOfLeastMoment[0]) /
                   stdev_axisOfLeastMoment_0,
               2);
  float distance_axis_1 =
      std::pow((f1.axisOfLeastMoment[1] - f2.axisOfLeastMoment[1]) /
                   stdev_axisOfLeastMoment_1,
               2);
  float distance_filled =
      std::pow((f1.percentFilled - f2.percentFilled) / stdev_percentFilled, 2);
  float distance_aspect = std::pow(
      (f1.bboxAspectRatio - f2.bboxAspectRatio) / stdev_bboxAspectRatio, 2);

  float hausdorffDist = hausdorffDistance(f1.contour, f2.contour);
  float weight_hausdorff = 1.0;
  distance += weight_hausdorff * hausdorffDist;

  distance += distance_center_x + distance_center_y + distance_width +
              distance_height + distance_angle + distance_axis_0 +
              distance_axis_1 + distance_filled + distance_aspect;

  // Debug prints
  std::cout << "Calculating distance..." << std::endl;
  std::cout << "center_x distance: " << distance_center_x << std::endl;
  std::cout << "center_y distance: " << distance_center_y << std::endl;
  std::cout << "width distance: " << distance_width << std::endl;
  std::cout << "height distance: " << distance_height << std::endl;
  std::cout << "angle distance: " << distance_angle << std::endl;
  std::cout << "axis_0 distance: " << distance_axis_0 << std::endl;
  std::cout << "axis_1 distance: " << distance_axis_1 << std::endl;
  std::cout << "percent filled distance: " << distance_filled << std::endl;
  std::cout << "aspect ratio distance: " << distance_aspect << std::endl;
  return std::sqrt(distance);
}

// Add this new function to extract label from filename
std::string extractLabelFromFileName(const std::string &filename) {
  // Get just the filename without path
  size_t last_slash = filename.find_last_of("/\\");
  std::string basename = (last_slash != std::string::npos)
                             ? filename.substr(last_slash + 1)
                             : filename;

  // Find the last dot (extension)
  size_t last_dot = basename.find_last_of('.');
  if (last_dot == std::string::npos)
    return "unknown";

  // Find the last underscore
  size_t last_underscore = basename.find_last_of('_');
  if (last_underscore == std::string::npos)
    return "unknown";

  // Extract the label (everything between last underscore and extension)
  return basename.substr(last_underscore + 1, last_dot - last_underscore - 1);
}

int main() {
  std::vector<ObjectData> objectDB;

  // Create database directory if it doesn't exist
  std::string dbPath = "../database";
  if (system(("mkdir -p " + dbPath).c_str()) != 0) {
    std::cerr << "Warning: Could not create directory: " << dbPath << std::endl;
  }

  // Debug output to check the actual path
  char resolved_path[PATH_MAX];
  if (realpath("../train_data", resolved_path) != nullptr) {
    std::cout << "Full path to train_data: " << resolved_path << std::endl;

    // List all files in the directory
    std::cout << "Listing all files in directory:" << std::endl;
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir(resolved_path)) != nullptr) {
      while ((ent = readdir(dir)) != nullptr) {
        std::cout << "  " << ent->d_name << std::endl;
      }
      closedir(dir);
    }
  }

  // Get all image files from train_data directory
  std::vector<cv::String> filenames;

  // Try with absolute path
  std::string path_str(resolved_path);
  cv::glob(path_str + "/*_*.jpg",
           filenames); // Match any filename containing an underscore

  std::cout << "Found " << filenames.size() << " images in " << path_str
            << std::endl;

  if (filenames.empty()) {
    std::cout << "No images found in ../train_data" << std::endl;
    std::cout
        << "Please place your training images in the ../train_data directory"
        << std::endl;
    std::cout << "Image names should be in format: image_1_label.jpg"
              << std::endl;
    return 1;
  }

  // Process each image
  for (const auto &filename : filenames) {
    std::cout << "Processing: " << filename << std::endl;

    // Read image
    cv::Mat frame = cv::imread(filename);
    if (frame.empty()) {
      std::cerr << "Error: Could not read image: " << filename << std::endl;
      continue;
    }

    // Process image
    cv::Mat thresholded;
    dynamicThresholding(frame, thresholded);

    cv::Mat cleaned;
    applyMorphologicalOperations(thresholded, cleaned);

    // Ensure the cleaned image is strictly binary
    cv::threshold(cleaned, cleaned, 127, 255, cv::THRESH_BINARY);

    cv::Mat stats;
    cv::Mat labels = labelConnectedComponents(cleaned, 500, stats);
    cv::Mat coloredLabels = colorConnectedComponents(labels);

    // Find the largest connected component
    int largestArea = 0;
    int largestLabel = 0;
    for (int i = 1; i < stats.rows; i++) {
      int area = stats.at<int>(i, cv::CC_STAT_AREA);
      if (area > largestArea) {
        largestArea = area;
        largestLabel = i;
      }
    }

    if (largestLabel > 0) {
      // Extract features from the largest component
      RegionFeatures features = computeRegionFeatures(labels, largestLabel);

      // Extract label from filename
      std::string label = extractLabelFromFileName(filename);

      // Add to database
      objectDB.push_back({label, features});
      std::cout << "Added " << label << " to the database with features."
                << std::endl;
    } else {
      std::cerr << "No significant objects found in: " << filename << std::endl;
    }
  }

  // Print database contents
  std::cout << "\nDatabase contents:" << std::endl;
  for (const ObjectData &data : objectDB) {
    std::cout << "Label: " << data.label << std::endl;
    std::cout << "Features: " << std::endl;
    std::cout << "Center: (" << data.features.orientedBoundingBox.center.x
              << ", " << data.features.orientedBoundingBox.center.y << ")"
              << std::endl;
    std::cout << "Width: " << data.features.orientedBoundingBox.size.width
              << std::endl;
    std::cout << "Height: " << data.features.orientedBoundingBox.size.height
              << std::endl;
    std::cout << "Angle: " << data.features.orientedBoundingBox.angle
              << std::endl;
    std::cout << "--------------------------------" << std::endl;
  }

  // Save the updated database with relative path
  std::cout << "Total objects in database: " << objectDB.size() << std::endl;
  saveObjectDB("../database/images.csv", objectDB);
  std::cout << "Database saved successfully." << std::endl;

  return 0;
}
