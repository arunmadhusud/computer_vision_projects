cmake_minimum_required( VERSION 2.8 )
project(feature_extraction)

find_package(OpenCV 3)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 3 not found.")
   endif()

include_directories(${OpenCV_INCLUDE_DIRS})
add_executable( feature_extraction feature.cpp csv_util.cpp matching.cpp)

target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS} )
