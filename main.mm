#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <Cocoa/Cocoa.h>
#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>
#include <iostream>
#include <string>
#include <vector>

#include "display.h"
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

// function to compute features
// Structure to hold region features
struct RegionFeatures {
  cv::RotatedRect orientedBoundingBox;
  cv::Vec2f axisOfLeastMoment;
  float percentFilled;
  float bboxAspectRatio;
};

RegionFeatures computeRegionFeatures(const cv::Mat &labels, int regionID) {
  RegionFeatures features;
  std::vector<cv::Point2f> regionPoints;
  // Extract points belonging to the region
  for (int i = 0; i < labels.rows; i++) {
    for (int j = 0; j < labels.cols; j++) {
      if (labels.at<int>(i, j) == regionID) {
        regionPoints.push_back(cv::Point2f(
            j, i)); // Note: (j, i) because j is x (column) and i is y (row)
      }
    }
  }
  // Skip if there are not enough points
  if (regionPoints.size() < 2) {
    std::cerr << "Error: Not enough distinct points in region " << regionID
              << " to compute features." << std::endl;
    return features;
  }
  try {
    // Compute the oriented bounding box
    features.orientedBoundingBox = cv::minAreaRect(regionPoints);
    // Compute the aspect ratio of the oriented bounding box
    features.bboxAspectRatio =
        std::max(features.orientedBoundingBox.size.width,
                 features.orientedBoundingBox.size.height) /
        std::min(features.orientedBoundingBox.size.width,
                 features.orientedBoundingBox.size.height);
    // Compute percent filled within the oriented bounding box
    cv::Mat mask = cv::Mat::zeros(labels.size(), CV_8U);
    for (const auto &point : regionPoints) {
      mask.at<uchar>(point.y, point.x) = 255;
    }
    cv::RotatedRect rect = features.orientedBoundingBox;
    cv::Rect boundingBox = rect.boundingRect();
    cv::Mat croppedMask = mask(boundingBox);
    double areaFilled = cv::countNonZero(croppedMask);
    double totalArea = boundingBox.width * boundingBox.height;
    features.percentFilled = areaFilled / totalArea;
  } catch (const cv::Exception &e) {
    std::cerr << "Error computing features for region " << regionID << ": "
              << e.what() << std::endl;
  }
  return features;
}

int main() {
  [NSApplication sharedApplication]; // Initialize NSApplication

  // Open the default camera
  cv::VideoCapture cap(0);
  cv::waitKey(2000);

  if (!cap.isOpened()) {
    std::cerr << "Error: Could not open camera." << std::endl;
    return -1;
  }

  // adjust the exposure and brightness of the external camera
  cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 0);
  cap.set(cv::CAP_PROP_EXPOSURE, 3);
  cap.set(cv::CAP_PROP_BRIGHTNESS, 70);

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

      // Debug output
      std::cout << "Number of channels in 'original' image: "
                << frame.channels() << std::endl;

      cv::Mat thresholded;
      dynamicThresholding(frame, thresholded);

      // Debug output
      std::cout << "Number of channels in 'thresholded' image: "
                << thresholded.channels() << std::endl;

      cv::Mat cleaned;
      applyMorphologicalOperations(thresholded, cleaned);

      // Ensure the cleaned image is strictly binary
      cv::threshold(cleaned, cleaned, 127, 255, cv::THRESH_BINARY);

      // Debug output
      std::cout << "Number of channels in 'cleaned' image: "
                << cleaned.channels() << std::endl;

      cv::Mat stats;
      cv::Mat labels = labelConnectedComponents(
          cleaned, 500, stats); // 500 is a sample minimum size threshold
      cv::Mat coloredLabels = colorConnectedComponents(labels);

      std::cout << "Number of channels in 'coloredLabels' image: "
                << coloredLabels.channels() << std::endl;

      // Compute features for each major region and visualize
      cv::Mat visualization = coloredLabels.clone();
      for (int i = 1; i < stats.rows;
           i++) { // Starting from 1 to skip the background
        RegionFeatures features = computeRegionFeatures(labels, i);
        // Draw oriented bounding box
        cv::Point2f vertices[4];
        features.orientedBoundingBox.points(vertices);
        for (int j = 0; j < 4; j++) {
          cv::line(visualization, vertices[j], vertices[(j + 1) % 4],
                   cv::Scalar(0, 255, 0), 2);
        }
        // Draw axis of least moment
        cv::Point2f center = features.orientedBoundingBox.center;
        cv::Point2f endpoint =
            center + 100 * cv::Point2f(features.axisOfLeastMoment[0],
                                       features.axisOfLeastMoment[1]);
        cv::line(visualization, center, endpoint, cv::Scalar(0, 0, 255), 2);
      }

      // Displaying the original, thresholded, morphological, and segmented
      // frames with features
      displayImages(frame, thresholded, cleaned, visualization);

      int key = cv::waitKey(100);
      if (key == 27) { // ESC key
        break;
      } else if (key == 115) { // 's' key
        displayAndOptionallySave(frame, thresholded, cleaned, visualization);
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
