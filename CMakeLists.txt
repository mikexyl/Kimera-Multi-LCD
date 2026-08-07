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
find_package(pcl_conversions REQUIRED)
find_package(rosbag2_cpp REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

find_package(Boost REQUIRED)

find_package(GTSAM REQUIRED)
find_package(opengv REQUIRED)
find_package(teaserpp REQUIRED)
find_package(OpenCV REQUIRED)
find_package(xfeat-cpp REQUIRED)
find_package(DBoW2 REQUIRED)
find_package(PCL REQUIRED)
find_package(spdlog REQUIRED)
find_package(gflags REQUIRED)
find_package(nlohmann_json REQUIRED)

if(NOT TARGET DBoW2::DBoW2)
  add_library(DBoW2::DBoW2 INTERFACE IMPORTED)
  set_target_properties(DBoW2::DBoW2 PROPERTIES
    INTERFACE_LINK_LIBRARIES "${DBoW2_LIBRARIES}"
    INTERFACE_INCLUDE_DIRECTORIES "${DBoW2_INCLUDE_DIRS}")
endif()

# Collect all ROS 2 dependencies
set(dependencies
  rclcpp
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  image_transport
  pose_graph_tools_msgs
  pcl_conversions
  rosbag2_cpp
  cv_bridge
  tf2
  tf2_ros
  tf2_geometry_msgs
)


# Create library
add_library(${PROJECT_NAME}
  src/loop_closure_detector.cpp
  src/lcd_third_party.cpp
  src/types.cpp
  src/utils.cpp
  src/serializer.cpp
  src/io.cpp
  src/VLADLoopClosureDetector.cpp
  src/teaser_sim3_estimator.cpp
  src/orbslam3_sim3_estimator.cpp
  third_party/pyslam/solvers/Random.cpp
  third_party/pyslam/solvers/Sim3Solver.cpp
)

target_link_libraries(${PROJECT_NAME}
  ${Boost_LIBRARIES}
  ${OpenCV_LIBRARIES}
  gtsam
  opengv
  teaserpp::teaser_registration
  DBoW2::DBoW2
  tbb
  glog
  ${GFLAGS_LIBRARIES}
  spdlog::spdlog
  xfeat-cpp::xfeat-cpp
  nlohmann_json::nlohmann_json
)

target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${Boost_INCLUDE_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${DBoW2_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/pyslam/solvers
)

message(STATUS "DBoW2_INCLUDE_DIR: ${DBoW2_INCLUDE_DIR}")

# Add ROS 2 dependencies to the target
ament_target_dependencies(${PROJECT_NAME} ${dependencies})

# Unit tests
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # This upstream fork follows its original Google-style formatting and does
  # not carry ament copyright headers. Keep the compatible static/configuration
  # checks without rewriting unrelated source as part of the ROS 2 port.
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  set(ament_cmake_uncrustify_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(${PROJECT_NAME}-test_lcd test/test_lcd.cpp)
  target_link_libraries(${PROJECT_NAME}-test_lcd ${PROJECT_NAME})
  ament_target_dependencies(${PROJECT_NAME}-test_lcd ${dependencies})

  ament_add_gtest(${PROJECT_NAME}-test_utils test/test_utils.cpp)
  target_link_libraries(${PROJECT_NAME}-test_utils ${PROJECT_NAME})
  ament_target_dependencies(${PROJECT_NAME}-test_utils ${dependencies})

  ament_add_gtest(${PROJECT_NAME}-test_teaser_sim3 test/test_teaser_sim3.cpp)
  target_link_libraries(${PROJECT_NAME}-test_teaser_sim3 ${PROJECT_NAME})
  ament_target_dependencies(${PROJECT_NAME}-test_teaser_sim3 ${dependencies})

  ament_add_gtest(${PROJECT_NAME}-test_orbslam3_sim3
    test/test_orbslam3_sim3.cpp)
  target_link_libraries(${PROJECT_NAME}-test_orbslam3_sim3 ${PROJECT_NAME})
  ament_target_dependencies(${PROJECT_NAME}-test_orbslam3_sim3 ${dependencies})
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
ament_export_dependencies(${dependencies} gflags nlohmann_json teaserpp)

ament_package()
