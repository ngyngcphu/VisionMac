cmake_minimum_required(VERSION 3.22)
project(cv-object-recognition)

# Require C++17
set(CMAKE_CXX_STANDARD          17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS        NO)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Include OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

# Add your source file
add_executable(main main.mm display.mm thresholding.mm)

# Link OpenCV libraries to your executable
target_link_libraries(main 
    ${OpenCV_LIBS}
    "-framework Cocoa"
    "-framework UniformTypeIdentifiers"
)
