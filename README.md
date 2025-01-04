# VisionMac
This project develops an Object Recognition (OR) system on macOS designed to identify objects within video footage. The system employs computer vision methods and algorithms to analyze video frames. This involves image preprocessing steps such as thresholding and cleaning, followed by image segmentation to isolate object regions. Feature extraction is then performed on these regions, which are compared against a database of known objects to enable real-time classification. This report provides a comprehensive overview of the technical aspects, visual examples, and insights gained throughout the development process.

## Requirements
- OpenCV 4.10.0
- macOS Sequoia 15.2

## Project Report
### I. Thresholded Objects
The thresholding process is a crucial step in the VisionMac project, which aims to identify objects within video footage. This section of the report will delve into the technical details of the thresholding algorithm implemented in the project, highlighting its role in the overall object recognition system.

1. **Thresholding Process Overview**
The thresholding process in VisionMac is designed to convert a color image into a binary image, where the objects of interest are separated from the background. This is achieved through a series of image processing steps, as outlined in the [thresholding.mm](thresholding.mm) file.

2. **Dynamic Thresholding Algorithm**
The [dynamicThresholding](thresholding.mm) function is the core of the thresholding process. It involves the following steps:

    - **Pre-processing with Gaussian Filter**: The input image is first smoothed using a Gaussian blur to reduce noise and detail, which helps in achieving a more accurate thresholding result.
    - **Color Space Conversion**: The blurred image is converted from the BGR color space to the LAB color space. This conversion is beneficial as the LAB color space separates the luminance from the color information, making it easier to perform thresholding based on intensity.
    - **K-Means Clustering**: The image is reshaped into a 1D array and converted to a floating-point format. K-means clustering is then applied to segment the image into two clusters. This segmentation helps in distinguishing the object from the background based on color intensity.
    - **Threshold Calculation**: The average intensity of the two clusters is calculated, and the threshold value is set as the average of these two means. This dynamic calculation allows the threshold to adapt to different lighting conditions and object colors.
    - **Image Thresholding**: The [thresholdImage](thresholding.mm) function applies the calculated threshold to the original image, setting pixels below the threshold to black and those above to white, effectively isolating the objects.

3. **Implementation Details**
The implementation of the thresholding process is encapsulated in the [thresholding.mm](thresholding.mm) file. The key functions involved are:
- **dynamicThresholding**: Handles the entire thresholding process, from pre-processing to threshold application.
- **thresholdImage**: Applies the calculated threshold to produce a binary image.

4. **Results**
The following images show the thresholding process in action:

![Original Image](images/original.jpeg)
![Thresholded Image](images/thresholded.jpeg)

5. **Conclusion**
The thresholding algorithm implemented in VisionMac is a vital component of the object recognition pipeline. By dynamically adjusting the threshold based on image content, it provides a reliable method for object isolation, paving the way for subsequent feature extraction and classification steps.
