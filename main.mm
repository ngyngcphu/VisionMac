#include <opencv2/opencv.hpp>

#include <Cocoa/Cocoa.h>
#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>
#include <iostream>
#include <string>
#include <vector>

#include "display.h"
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
                              const cv::Mat &thresholded) {
  displayImages(original, thresholded);

  @autoreleasepool {
    NSAlert *alert = [[NSAlert alloc] init];
    [alert setMessageText:@"Save Image"];
    [alert setInformativeText:@"Would you like to save the thresholded image?"];
    [alert addButtonWithTitle:@"Save"];
    [alert addButtonWithTitle:@"Cancel"];

    if ([alert runModal] == NSAlertFirstButtonReturn) {
      std::string savePath;
      if (saveFileDialog(savePath)) {
        if (!cv::imwrite(savePath, thresholded)) {
          NSAlert *errorAlert = [[NSAlert alloc] init];
          [errorAlert setMessageText:@"Error"];
          [errorAlert setInformativeText:
                          @"Could not save the image to the specified path."];
          [errorAlert runModal];
        } else {
          NSAlert *successAlert = [[NSAlert alloc] init];
          [successAlert setMessageText:@"Success"];
          [successAlert setInformativeText:@"Image saved successfully!"];
          [successAlert runModal];
        }
      }
    }
  }
}

void saveImages(const cv::Mat &original, const cv::Mat &thresholded) {
  @autoreleasepool {
    NSAlert *alert = [[NSAlert alloc] init];
    [alert setMessageText:@"Save Images"];
    [alert setInformativeText:@"Which image would you like to save?"];
    [alert addButtonWithTitle:@"Both"];
    [alert addButtonWithTitle:@"Original"];
    [alert addButtonWithTitle:@"Thresholded"];
    [alert addButtonWithTitle:@"Cancel"];

    NSModalResponse response = [alert runModal];

    if (response == NSAlertFirstButtonReturn) { // Both
      std::string originalPath, thresholdedPath;

      NSAlert *saveOriginal = [[NSAlert alloc] init];
      [saveOriginal setMessageText:@"Save Original Image"];
      [saveOriginal
          setInformativeText:@"Choose where to save the original image"];
      [saveOriginal runModal];

      if (saveFileDialog(originalPath)) {
        cv::imwrite(originalPath, original);
      }

      NSAlert *saveThresholded = [[NSAlert alloc] init];
      [saveThresholded setMessageText:@"Save Thresholded Image"];
      [saveThresholded
          setInformativeText:@"Choose where to save the thresholded image"];
      [saveThresholded runModal];

      if (saveFileDialog(thresholdedPath)) {
        cv::imwrite(thresholdedPath, thresholded);
      }
    } else if (response == NSAlertSecondButtonReturn) { // Original
      std::string savePath;
      if (saveFileDialog(savePath)) {
        cv::imwrite(savePath, original);
      }
    } else if (response == NSAlertThirdButtonReturn) { // Thresholded
      std::string savePath;
      if (saveFileDialog(savePath)) {
        cv::imwrite(savePath, thresholded);
      }
    }
  }
}

int main() {
  [NSApplication sharedApplication];

  // Open the default camera
  cv::VideoCapture cap(0);

  if (!cap.isOpened()) {
    std::cerr << "Error: Could not open camera." << std::endl;
    return -1;
  }

  // Adjust camera settings if needed
  cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 0);
  cap.set(cv::CAP_PROP_EXPOSURE, 0);
  cap.set(cv::CAP_PROP_BRIGHTNESS, 70);

  // Create windows first
  cv::namedWindow("Original", cv::WINDOW_NORMAL);
  cv::namedWindow("Processed", cv::WINDOW_NORMAL);

  bool running = true;
  while (running) {
    @autoreleasepool {
      cv::Mat frame;
      cap >> frame;

      if (frame.empty()) {
        std::cerr << "Error: Could not read frame from camera." << std::endl;
        break;
      }

      cv::Mat thresholded;
      dynamicThresholding(frame, thresholded);

      // Show the images
      if (!frame.empty()) {
        cv::imshow("Original", frame);
      }
      if (!thresholded.empty()) {
        cv::imshow("Processed", thresholded);
      }

      // Handle key press with a shorter wait time
      char key = cv::waitKey(30);

      if (key == 27) { // ESC key
        running = false;
      } else if (key == 's' || key == 'S') {
        saveImages(frame, thresholded);
      }

      // Check if windows are still open
      if (cv::getWindowProperty("Original", cv::WND_PROP_VISIBLE) < 1 ||
          cv::getWindowProperty("Processed", cv::WND_PROP_VISIBLE) < 1) {
        running = false;
      }
    }
  }

  cap.release();
  cv::destroyAllWindows();

  return 0;
}
