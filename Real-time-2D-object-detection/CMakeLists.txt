cmake_minimum_required( VERSION 2.8 )
project(objectrec)

find_package(OpenCV 3)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 3 not found.")
   endif()

include_directories(${OpenCV_INCLUDE_DIRS})
add_executable( objectrec check.cpp recognition.cpp csv_util.cpp)
# add_executable( objectrec feature.cpp recognition.cpp csv_util.cpp)
# add_executable( objectrec backup.cpp recognition.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS} )

