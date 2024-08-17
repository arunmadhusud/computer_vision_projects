cmake_minimum_required( VERSION 2.8 )
project(calib)

find_package(OpenCV 4)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 4 not found.")
   endif()

include_directories(${OpenCV_INCLUDE_DIRS})
# add_executable( calib calibration.cpp definitions.cpp)
add_executable( calib arsystem.cpp definitions.cpp)
# add_executable( calib harriscorners.cpp definitions.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS} )

