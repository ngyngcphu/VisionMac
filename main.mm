#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <Cocoa/Cocoa.h>
#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "display.h"
#include "feature_extraction.h"
#include "morphologicalOperations.h"
#include "thresholding.h"

// function to select image
std::vector<std::string> openFileDialog() {
  std::vector<std::string> fileNames;
  @autoreleasepool {
    NSOpenPanel *panel = [NSOpenPanel openPanel];
    [panel setCanChooseFiles:YES];
    [panel setCanChooseDirectories:NO];
    [panel setAllowsMultipleSelection:YES];

    [panel setAllowedContentTypes:@[ UTTypeJPEG, UTTypePNG ]];

    if ([panel runModal] == NSModalResponseOK) {
      for (NSURL *url in [panel URLs]) {
        fileNames.push_back(std::string([[url path] UTF8String]));
      }
    }
  }

  return fileNames;
}
bool saveFileDialog(std::string &outFilePath) {
  bool result = false;

  @autoreleasepool {
    NSSavePanel *panel = [NSSavePanel savePanel];
    [panel setAllowedContentTypes:@[ UTTypeJPEG, UTTypePNG ]];
    [panel setExtensionHidden:NO];

    if ([panel runModal] == NSModalResponseOK) {
      NSURL *url = [panel URL];
      outFilePath = std::string([[url path] UTF8String]);
      result = true;
    }
  }

  return result;
}

void displayAndOptionallySave(const cv::Mat &original,
                              const cv::Mat &thresholded,
                              const cv::Mat &cleaned,
                              const cv::Mat &segmented) {
  displayImages(original, thresholded, cleaned, segmented);

  @autoreleasepool {
    // Prompt for the original image
    NSAlert *originalAlert = [[NSAlert alloc] init];
    [originalAlert setMessageText:@"Save Original Image"];
    [originalAlert
        setInformativeText:@"Would you like to save the original image?"];
    [originalAlert addButtonWithTitle:@"Save"];
    [originalAlert addButtonWithTitle:@"Cancel"];

    if ([originalAlert runModal] == NSAlertFirstButtonReturn) {
      std::string savePath;
      if (saveFileDialog(savePath)) {
        if (!cv::imwrite(savePath, original)) {
          NSAlert *errorAlert = [[NSAlert alloc] init];
          [errorAlert setMessageText:@"Error"];
          [errorAlert
              setInformativeText:@"Could not save the original image to "
                                 @"the specified path."];
          [errorAlert runModal];
        } else {
          NSAlert *successAlert = [[NSAlert alloc] init];
          [successAlert setMessageText:@"Success"];
          [successAlert
              setInformativeText:@"Original image saved successfully!"];
          [successAlert runModal];
        }
      }
    }
    // Prompt for the thresholded image
    NSAlert *thresholdAlert = [[NSAlert alloc] init];
    [thresholdAlert setMessageText:@"Save Thresholded Image"];
    [thresholdAlert
        setInformativeText:@"Would you like to save the thresholded image?"];
    [thresholdAlert addButtonWithTitle:@"Save"];
    [thresholdAlert addButtonWithTitle:@"Cancel"];

    if ([thresholdAlert runModal] == NSAlertFirstButtonReturn) {
      std::string savePath;
      if (saveFileDialog(savePath)) {
        if (!cv::imwrite(savePath, thresholded)) {
          NSAlert *errorAlert = [[NSAlert alloc] init];
          [errorAlert setMessageText:@"Error"];
          [errorAlert setInformativeText:@"Could not save the thresholded "
                                         @"image to the specified path."];
          [errorAlert runModal];
        } else {
          NSAlert *successAlert = [[NSAlert alloc] init];
          [successAlert setMessageText:@"Success"];
          [successAlert
              setInformativeText:@"Thresholded image saved successfully!"];
          [successAlert runModal];
        }
      }
    }

    // Prompt for the morphological image
    NSAlert *morphAlert = [[NSAlert alloc] init];
    [morphAlert setMessageText:@"Save Morphological Image"];
    [morphAlert
        setInformativeText:@"Would you like to save the morphological image?"];
    [morphAlert addButtonWithTitle:@"Save"];
    [morphAlert addButtonWithTitle:@"Cancel"];

    if ([morphAlert runModal] == NSAlertFirstButtonReturn) {
      std::string savePath;
      if (saveFileDialog(savePath)) {
        if (!cv::imwrite(savePath, cleaned)) {
          NSAlert *errorAlert = [[NSAlert alloc] init];
          [errorAlert setMessageText:@"Error"];
          [errorAlert setInformativeText:@"Could not save the morphological "
                                         @"image to the specified path."];
          [errorAlert runModal];
        } else {
          NSAlert *successAlert = [[NSAlert alloc] init];
          [successAlert setMessageText:@"Success"];
          [successAlert
              setInformativeText:@"Morphological image saved successfully!"];
          [successAlert runModal];
        }
      }
    }

    // Prompt for the segmented image
    NSAlert *segmentAlert = [[NSAlert alloc] init];
    [segmentAlert setMessageText:@"Save Segmented Image"];
    [segmentAlert
        setInformativeText:@"Would you like to save the segmented image?"];
    [segmentAlert addButtonWithTitle:@"Save"];
    [segmentAlert addButtonWithTitle:@"Cancel"];

    if ([segmentAlert runModal] == NSAlertFirstButtonReturn) {
      std::string savePath;
      if (saveFileDialog(savePath)) {
        if (!cv::imwrite(savePath, segmented)) {
          NSAlert *errorAlert = [[NSAlert alloc] init];
          [errorAlert setMessageText:@"Error"];
          [errorAlert
              setInformativeText:
                  @"Could not save the segmented image to the specified path."];
          [errorAlert runModal];
        } else {
          NSAlert *successAlert = [[NSAlert alloc] init];
          [successAlert setMessageText:@"Success"];
          [successAlert
              setInformativeText:@"Segmented image saved successfully!"];
          [successAlert runModal];
        }
      }
    }
  }
}

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
  std::ofstream outFile(filename, std::ios::out);
  if (!outFile) {
    std::cerr << "Error: Couldn't open file for writing: " << filename
              << std::endl;
    return;
  }
  // Write the number of objects
  outFile << objectDB.size() << std::endl;
  for (const ObjectData &data : objectDB) {
    outFile << data.label << " ";
    // Serialize the features:
    outFile << data.features.orientedBoundingBox.center.x << " "
            << data.features.orientedBoundingBox.center.y << " "
            << data.features.orientedBoundingBox.size.width << " "
            << data.features.orientedBoundingBox.size.height << " "
            << data.features.orientedBoundingBox.angle << " "
            << data.features.axisOfLeastMoment[0] << " "
            << data.features.axisOfLeastMoment[1] << " "
            << data.features.percentFilled << " "
            << data.features.bboxAspectRatio << std::endl;

    // Serialize the Hu Moments
    for (int i = 0; i < 7; i++) {
      outFile << data.features.huMoments[i] << " ";
    }
    outFile << std::endl;
  }
  outFile.close();
  std::cout << "Saved object data to: " << filename << std::endl;
}

// function to load object database from a csv file
bool loadObjectDBFromCSV(const std::string &filename,
                         std::vector<ObjectData> &objectDB) {
  std::ifstream inFile(filename, std::ios::in);
  if (!inFile) {
    std::cerr << "Error: Couldn't open file for reading: " << filename
              << std::endl;
    return false;
  }
  std::string line;
  while (std::getline(inFile, line)) {
    std::stringstream ss(line);
    ObjectData data;
    std::getline(ss, data.label, ','); // Get the label

    // Deserialize the features:
    ss >> data.features.orientedBoundingBox.center.x >>
        data.features.orientedBoundingBox.center.y >>
        data.features.orientedBoundingBox.size.width >>
        data.features.orientedBoundingBox.size.height >>
        data.features.orientedBoundingBox.angle >>
        data.features.axisOfLeastMoment[0] >>
        data.features.axisOfLeastMoment[1] >>
        data.features.orthogonalVector[0] >>
        data.features.orthogonalVector[1] >> data.features.percentFilled >>
        data.features.bboxAspectRatio;

    // Deserialize the additional features
    for (int i = 0; i < 16; i++) {
      ss >> data.additionalFeatures[i];
    }

    objectDB.push_back(data);
  }
  inFile.close();
  std::cout << "Loaded object data from: " << filename << std::endl;
  return true;
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

// Function to compare new object's feature vector to existing database using
// the distance metric
std::string classifyObject(const RegionFeatures &features,
                           const std::vector<ObjectData> &objectDB) {
  std::cout << "Classifying object..." << std::endl;
  std::cout << "Number of objects in objectDB: " << objectDB.size()
            << std::endl;
  float minDistance = std::numeric_limits<float>::max();
  std::string bestMatchLabel = "Unknown";

  for (const ObjectData &data : objectDB) {
    std::cout << "Comparing with object labeled: " << data.label << std::endl;
    float distance = scaledEuclideanDistance(features, data.features);
    std::cout << "Distance to " << data.label << ": " << distance << std::endl;
    if (distance < minDistance) {
      minDistance = distance;
      bestMatchLabel = data.label;
    }
  }
  return bestMatchLabel;
}

int main() {
  [NSApplication sharedApplication]; // Initialize NSApplication
  std::vector<ObjectData> objectDB;  // Initialize the objectDB vector

  // load the existing database from a local csv:
  if (loadObjectDBFromCSV("/database/images.csv", objectDB)) {
    std::cout << "Number of objects in database: " << objectDB.size()
              << std::endl;
    for (const ObjectData &data : objectDB) {
      std::cout << "Label: " << data.label << std::endl;
    }
  } else {
    std::cerr << "Failed to load object database." << std::endl;
    return -1; // Exit if we can't load the database
  }

  // Open the default camera
  cv::VideoCapture cap(0);
  cv::waitKey(2000);

  if (!cap.isOpened()) {
    std::cerr << "Error: Could not open camera." << std::endl;
    return -1;
  }

  // Adjust the exposure and brightness of the external camera
  cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 0);
  cap.set(cv::CAP_PROP_EXPOSURE, 0);
  cap.set(cv::CAP_PROP_BRIGHTNESS, 20);

  // Create windows first
  cv::namedWindow("Original", cv::WINDOW_NORMAL);
  cv::namedWindow("Thresholded", cv::WINDOW_NORMAL);
  cv::namedWindow("Cleaned", cv::WINDOW_NORMAL);
  cv::namedWindow("Segmented", cv::WINDOW_NORMAL);

  while (true) {
    @autoreleasepool {
      cv::Mat frame;
      cap >> frame;

      if (frame.empty()) {
        std::cerr << "Error: Could not read frame from camera." << std::endl;
        break;
      }

      cv::Mat thresholded;
      dynamicThresholding(frame, thresholded);

      cv::Mat cleaned;
      applyMorphologicalOperations(thresholded, cleaned);

      // Ensure the cleaned image is strictly binary
      cv::threshold(cleaned, cleaned, 127, 255, cv::THRESH_BINARY);

      cv::Mat stats;
      cv::Mat labels = labelConnectedComponents(cleaned, 500, stats);
      cv::Mat coloredLabels = colorConnectedComponents(labels);

      // Compute features for each major region and visualize
      cv::Mat visualization = coloredLabels.clone();
      for (int i = 1; i < stats.rows;
           i++) { // Starting from 1 to skip the background
        std::cout << "Processing region: " << i << std::endl;
        RegionFeatures features = computeRegionFeatures(labels, i);

        // add label on top of the image
        int x = static_cast<int>(features.orientedBoundingBox.center.x);
        int y = static_cast<int>(features.orientedBoundingBox.center.y);
        int textOffsetY = 10;
        int textOffsetX = 40; // Increase this if necessary, depending on the
                              // average width of the text
        if (y - textOffsetY < 0) {
          y = textOffsetY; // Reset y to the offset if it's too close to the top
        }
        if (x - textOffsetX < 0) {
          x = textOffsetX; // Reset x to the offset if it's too close to the
                           // left edge
        }
        std::string label = classifyObject(features, objectDB);
        cv::putText(
            visualization, label, cv::Point(x - textOffsetX, y - textOffsetY),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

        // Draw oriented bounding box
        cv::Point2f vertices[4];
        features.orientedBoundingBox.points(vertices);
        for (int j = 0; j < 4; j++) {
          cv::line(visualization, vertices[j], vertices[(j + 1) % 4],
                   cv::Scalar(0, 255, 0), 2);
        }

        // Draw axis of least moment (as a line centered on the bounding box's
        // center)
        cv::Point2f center = features.orientedBoundingBox.center;

        // Draw in both directions from the center
        cv::Point2f endpoint =
            center + 100 * cv::Point2f(features.axisOfLeastMoment[0],
                                       features.axisOfLeastMoment[1]);
        cv::Point2f startpoint =
            center - 100 * cv::Point2f(features.axisOfLeastMoment[0],
                                       features.axisOfLeastMoment[1]);
        cv::line(visualization, startpoint, endpoint, cv::Scalar(0, 0, 255), 2);
      }

      // Displaying the image
      displayImages(frame, thresholded, cleaned, visualization);

      // Inform the user about possible actions
      std::cout << "Press 'n' to label the object, 's' to save the images, 'w' "
                   "to save the database, or 'q' to quit."
                << std::endl;

      // Wait for a user keypress
      int key = cv::waitKey(0);

      if (key == 27 || key == 113) { // ESC || 'q' key
        break;
      } else if (key == 'n' || key == 'N') {
        std::string label;
        std::cout << "Enter the label for the current object: ";
        std::cin >> label;
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(),
                        '\n'); // Clear the newline from the buffer

        // storing features from the lasrgest region
        int largestArea = 0;
        int largestLabel = 0;
        for (int i = 1; i < stats.rows; i++) {
          int area = stats.at<int>(i, cv::CC_STAT_AREA);
          if (area > largestArea) {
            largestArea = area;
            largestLabel = i;
          }
        }

        RegionFeatures features = computeRegionFeatures(labels, largestLabel);
        objectDB.push_back({label, features});
        std::cout << "Stored features for object labeled: " << label
                  << std::endl;
      } else if (key == 115) { // 's' key
        displayAndOptionallySave(frame, thresholded, cleaned, visualization);
      } else if (key == 119) { // 'w' key
        saveObjectDB("/database/images.csv", objectDB);
      } else if (cv::getWindowProperty("Original", cv::WND_PROP_VISIBLE) < 1 ||
                 cv::getWindowProperty("Thresholded", cv::WND_PROP_VISIBLE) <
                     1 ||
                 cv::getWindowProperty("Cleaned", cv::WND_PROP_VISIBLE) < 1 ||
                 cv::getWindowProperty("Segmented", cv::WND_PROP_VISIBLE) < 1) {

        NSAlert *alert = [[NSAlert alloc] init];
        [alert setMessageText:@"Save Frames"];
        [alert setInformativeText:@"Do you want to save the frames?"];
        [alert addButtonWithTitle:@"Yes"];
        [alert addButtonWithTitle:@"No"];

        if ([alert runModal] == NSAlertFirstButtonReturn) {
          displayAndOptionallySave(frame, thresholded, cleaned, visualization);
        }

        break;
      }
    }
  }

  cap.release();
  cv::destroyAllWindows();
  return 0;
}
