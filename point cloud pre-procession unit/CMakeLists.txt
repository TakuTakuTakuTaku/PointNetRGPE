cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(test_pcd)

find_package(PCL 1.9 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (test_pcd test_pcd.cpp)
target_link_libraries (test_pcd ${PCL_LIBRARIES})
