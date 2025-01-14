# VisionMac
This project develops an Object Recognition (OR) system on macOS designed to identify objects within video footage. The system employs computer vision methods and algorithms to analyze video frames. This involves image preprocessing steps such as thresholding and cleaning, followed by image segmentation to isolate object regions. Feature extraction is then performed on these regions, which are compared against a database of known objects to enable real-time classification. This report provides a comprehensive overview of the technical aspects, visual examples, and insights gained throughout the development process.

## Requirements
- OpenCV 4.10.0
- macOS Sequoia 15.2

## Project Report
### I. Thresholded Objects

The thresholding process is a crucial step in the VisionMac project, which aims to identify objects within video footage. This section of the report will delve into the technical details of the thresholding algorithm implemented in the project, highlighting its role in the overall object recognition system.

1. #### Thresholding Process Overview
    The thresholding process in VisionMac is designed to convert a color image into a binary image, where the objects of interest are separated from the background. This is achieved through a series of image processing steps, as outlined in the [thresholding.mm](thresholding.mm) file.

2. #### Dynamic Thresholding Algorithm
    The [dynamicThresholding](thresholding.mm) function is the core of the thresholding process. It involves the following steps:

    - ##### Pre-processing with Gaussian Filter 
      The input image is first smoothed using a Gaussian filter to reduce noise and detail, which helps in achieving a more accurate thresholding result.
    - ##### Color Space Conversion
      The blurred image is converted from the BGR color space to the LAB color space. This conversion is beneficial as the LAB color space separates the luminance from the color information, making it easier to perform thresholding based on intensity.
    - ##### K-Means Clustering
      The image is reshaped into a 1D array and converted to a floating-point format. K-means clustering is then applied to segment the image into two clusters. This segmentation helps in distinguishing the object from the background based on color intensity.
    - ##### Threshold Calculation
      The average intensity of the two clusters is calculated, and the threshold value is set as the average of these two means. This dynamic calculation allows the threshold to adapt to different lighting conditions and object colors.
    - ##### Image Thresholding
      The [thresholdImage](thresholding.mm) function applies the calculated threshold to the original image, setting pixels below the threshold to black and those above to white, effectively isolating the objects.

3. #### Implementation Details
    The implementation of the thresholding process is encapsulated in the [thresholding.mm](thresholding.mm) file. The key functions involved are:
    ```cpp
    // Handles the entire thresholding process, from pre-processing to threshold application.
    void dynamicThresholding(const cv::Mat &src, cv::Mat &dst);
    ```
   ```cpp
   // Applies the calculated threshold to produce a binary image.
   void thresholdImage(const cv::Mat &src, cv::Mat &dst, int thresholdValue);
   ```

4. #### Results
    ![Original Image](images/original.jpeg)
       *Original video frame image*
   
    ![Thresholded Image](images/thresholded.jpeg)
       *Thresholded video frame image*

5. #### Conclusion
    The thresholding algorithm implemented in VisionMac is a vital component of the object recognition pipeline. By dynamically adjusting the threshold based on image content, it provides a reliable method for object isolation, paving the way for subsequent feature extraction and classification steps.

### II. Cleaned Objects and Segmentation

The cleaning and segmentation processes are essential steps following thresholding in the VisionMac project. These steps further refine the binary images to enhance object isolation and prepare them for feature extraction and classification.

1. #### Cleaning Process Overview
    The cleaning process involves morphological operations that help in removing noise and small artifacts from the thresholded images. This is achieved using erosion and dilation techniques, which are implemented in the [morphologicalOperations.mm](morphologicalOperations.mm) file.

2. #### Morphological Operations
    The [applyMorphologicalOperations](morphologicalOperations.mm) function is responsible for cleaning the thresholded images. It involves the following steps:

    - ##### Erosion
      Erosion is applied to remove small white noise and detach connected objects. This operation uses a kernel to erode away the boundaries of the foreground object.
    - ##### Dilation
      Following erosion, dilation is applied to restore the eroded parts of the object. This operation uses a kernel to dilate the boundaries of the foreground object, ensuring that the object remains intact while noise is minimized.

3. #### Segmentation Process Overview
    After cleaning, the segmentation process labels connected components in the binary image. This step is crucial for identifying distinct objects within the frame.

4. #### Connected Components Labeling
    The [labelConnectedComponents](main.mm) function labels each connected component in the cleaned image. It involves:

    - ##### Inversion and Labeling
      The binary image is inverted, and connected components analysis is performed to label each distinct object. The function also filters out components smaller than a specified minimum size to reduce false positives.
    - ##### Coloring
      The [colorConnectedComponents](main.mm) function assigns a unique color to each labeled component, making it easier to visualize and distinguish between different objects.

5. #### Implementation Details
    The implementation of the cleaning and segmentation processes is encapsulated in the [morphologicalOperations.mm](morphologicalOperations.mm) and [main.mm](main.mm) files. The key functions involved are:
    ```cpp
    // Applies erosion and dilation to clean the thresholded image.
    void applyMorphologicalOperations(const cv::Mat &src, cv::Mat &dst);
    ```
    ```cpp
    // Labels connected components in the cleaned image.
    cv::Mat labelConnectedComponents(const cv::Mat &src, int minSize, cv::Mat &stats);
    ```

6. #### Results
    ![Original Image](images/segment_original.jpeg)
       *Original video frame image*
   
    ![Thresholded Image](images/segment_thresholded.jpeg)
       *Thresholded video frame image*
   
    ![Morphological Image](images/segment_morphological.jpeg)
       *Morphological video frame image*
   
    ![Segmented Image](images/segment_segmented.jpeg)
       *Segmented video frame image*

8. #### Conclusion
    The cleaning and segmentation processes are integral to the VisionMac object recognition pipeline. By effectively removing noise and labeling distinct objects, these processes enhance the accuracy of subsequent feature extraction and classification steps.

### III. Feature Computation

The feature computation process is a critical step in the VisionMac project, following the segmentation of objects. This step involves extracting meaningful features from each identified object, which are then used for classification and recognition.

1. #### Feature Extraction Overview
    Feature computation involves analyzing the segmented objects to extract various attributes that can uniquely identify them. These features are crucial for distinguishing between different objects and are used in the classification phase.

2. #### Region Features
    The computeRegionFeatures function is responsible for calculating features for each labeled region. It involves the following steps:

    - ##### Oriented Bounding Box
      The oriented bounding box is computed for each region, providing a minimal enclosing rectangle that can rotate to fit the object. This helps in understanding the object's orientation and size.
    - ##### Axis of Least Moment
      The axis of least moment is calculated, representing the direction in which the object has the least rotational inertia. This feature is useful for understanding the object's shape and orientation.
    - ##### Percent Filled
      This feature measures the proportion of the bounding box area that is filled by the object. It provides insight into the object's density and compactness.
    - ##### Bounding Box Aspect Ratio
      The aspect ratio of the bounding box is computed, which helps in identifying the object's shape characteristics.

3. #### Implementation Details
    The implementation of the feature computation is encapsulated in the [main.mm](main.mm) file. The key functions involved are:
    ```cpp
    // Computes features for a given region identified by its label.
    RegionFeatures computeRegionFeatures(const cv::Mat &labels, int regionID);
    ```

4. #### Results
    ![Original Image](images/bounding_original.jpeg)
       *Original video frame image*
   
    ![Thresholded Image](images/bounding_thresholded.jpeg)
       *Thresholded video frame image*
   
    ![Morphological Image](images/bounding_morphological.jpeg)
       *Morphological video frame image*
   
    ![Segmented Image](images/bounding_segmented.jpeg)
       *Segmented video frame image with features*

5. #### Conclusion
    The feature computation process in VisionMac is a vital component of the object recognition pipeline. By extracting detailed features from each object, the system can accurately classify and recognize objects in real-time, enhancing the overall effectiveness of the object recognition system.

### IV. Training Data Collection

The training data collection process is a crucial component of the VisionMac project, enabling the system to build a database of known objects for classification. This section details the implementation of training data collection and management.

1. #### Training Process Overview
    The training process involves capturing and processing images of known objects, extracting their features, and storing them in a database for later comparison. This process is implemented in the [process_train_data.mm](process_train_data.mm) file.

2. #### Data Collection Implementation
    The training data collection process involves several key steps:

    - ##### Image Acquisition
      ```cpp
          // Get all image files from train_data directory
      std::vector<cv::String> filenames;
      cv::glob(path_str + "/*_*.jpg", filenames);
      ```
      Training images are stored in the `train_data` directory with a specific naming convention: `image_1_label.jpg`, where "label" identifies the object.

    - ##### Image Processing Pipeline
      Each training image undergoes the same processing pipeline as live video frames:
      ```cpp
      // Process image
      cv::Mat thresholded;
      dynamicThresholding(frame, thresholded);
      
      cv::Mat cleaned;
      applyMorphologicalOperations(thresholded, cleaned);
      
      cv::Mat stats;
      cv::Mat labels = labelConnectedComponents(cleaned, 500, stats);
      ```
3. #### Feature Extraction and Storage
    The system extracts features from the largest connected component in each training image:

    - ##### Feature Computation
      ```cpp
      RegionFeatures features = computeRegionFeatures(labels, largestLabel);
      ```
      Features include oriented bounding box parameters, axis of least moment, percent filled, and aspect ratio.

    - ##### Database Storage
      ```cpp
      struct ObjectData {
        std::string label;
        RegionFeatures features;
        float additionalFeatures[16];
      };
      ```
      Features are stored in a database along with their corresponding labels.

4. #### Database Management
    The system includes functions for managing the object database:

    - ##### Saving Database
      ```cpp
      void saveObjectDB(const std::string &filename, const std::vector<ObjectData> &objectDB) {
      // Writes object data to CSV file
      // Format: label, features...
      }
      ```
      The database is saved in CSV format for persistence between sessions.

    - ##### Loading Database
      ```cpp
      bool loadObjectDBFromCSV(const std::string &filename, std::vector<ObjectData> &objectDB) {
      // Reads object data from CSV file
      // Populates objectDB vector
      }
      ```
      The database can be loaded from a CSV file when the system starts.

5. #### Interactive Training
    The system supports interactive training during runtime:
    ```cpp
    if (key == 'n' || key == 'N') {
    std::string label;
    std::cout << "Enter the label for the current object: ";
    std::cin >> label;
    
    RegionFeatures features = computeRegionFeatures(labels, largestLabel);
    objectDB.push_back({label, features});
    }
    ```
    Users can add new objects to the database by:
    - Capturing an image of the object
    - Pressing 'n' to enter a label
    - The system automatically extracts features and adds them to the database

6. #### Results
    The training process creates a comprehensive database of object features:
    ```
    Database contents:
    Label: object1
    Features:
    Center: (x, y)
    Width: w
    Height: h
    Angle: θ
    --------------------------------
    ```

7. #### Conclusion
    The training data collection system provides a robust and flexible way to build and maintain a database of known objects. The combination of automated feature extraction and interactive training allows for easy expansion of the system's object recognition capabilities.