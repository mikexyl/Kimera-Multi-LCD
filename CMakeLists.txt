cmake_minimum_required(VERSION 3.8)
project(kimera_multi_lcd)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(pose_graph_tools_msgs REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(rosbag2_cpp REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

find_package(Boost REQUIRED)

find_package(GTSAM REQUIRED)
find_package(opengv REQUIRED)
find_package(OpenCV REQUIRED)
find_package(DBoW2 REQUIRED)
find_package(PCL REQUIRED)

configure_file(cmake/json.CMakeLists.txt.in json-download/CMakeLists.txt)
execute_process(
  COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/json-download" OUTPUT_QUIET
)
execute_process(
  COMMAND "${CMAKE_COMMAND}" --build .
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/json-download" OUTPUT_QUIET
)

set(JSON_BuildTests OFF CACHE INTERNAL "")
set(JSON_Install OFF CACHE INTERNAL "")
add_subdirectory(${CMAKE_BINARY_DIR}/json-src ${CMAKE_BINARY_DIR}/json-build)

# Collect all ROS 2 dependencies
set(dependencies
  rclcpp
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  image_transport
  pose_graph_tools_msgs
  pcl_ros
  rosbag2_cpp
  cv_bridge
  tf2
  tf2_ros
  tf2_geometry_msgs
  DBoW2
)


# Create library
add_library(${PROJECT_NAME} 
  src/loop_closure_detector.cpp
  src/lcd_third_party.cpp
  src/types.cpp
  src/utils.cpp
  src/serializer.cpp
  src/io.cpp
)

target_link_libraries(${PROJECT_NAME}
  ${Boost_LIBRARIES}
  ${OpenCV_LIBRARIES}
  gtsam
  opengv
  ${DBoW2_LIBRARY}
  tbb
  glog
)

target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${Boost_INCLUDE_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_BINARY_DIR}/json-src/include
    ${DBoW2_INCLUDE_DIR}/../
)

message(STATUS "DBoW2_INCLUDE_DIR: ${DBoW2_INCLUDE_DIR}"  )

# Add ROS 2 dependencies to the target
ament_target_dependencies(${PROJECT_NAME} ${dependencies})

include(cmake/vocab_download.cmake)

# Unit tests 
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  # set(TEST_DATA_PATH "${CMAKE_CURRENT_SOURCE_DIR}/test/data")
  set(TEST_VOCAB_PATH "${CMAKE_CURRENT_SOURCE_DIR}/vocab")
  configure_file(test/test_config.h.in test/test_config.h)
  
  find_package(ament_cmake_gtest REQUIRED)
  
  ament_add_gtest(${PROJECT_NAME}-test_lcd test/test_lcd.cpp)
  target_link_libraries(${PROJECT_NAME}-test_lcd ${PROJECT_NAME})
  target_include_directories(${PROJECT_NAME}-test_lcd PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/test)
  ament_target_dependencies(${PROJECT_NAME}-test_lcd ${dependencies})

  ament_add_gtest(${PROJECT_NAME}-test_utils test/test_utils.cpp)
  target_link_libraries(${PROJECT_NAME}-test_utils ${PROJECT_NAME})
  target_include_directories(${PROJECT_NAME}-test_utils PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/test)
  ament_target_dependencies(${PROJECT_NAME}-test_utils ${dependencies})
endif()

# Install targets
install(TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include/
)

# Export dependencies
ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(${dependencies})

ament_package()
