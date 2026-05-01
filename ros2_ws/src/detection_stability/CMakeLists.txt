cmake_minimum_required(VERSION 3.8)
project(detection_stability)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(inha_interfaces REQUIRED)
find_package(message_filters REQUIRED)
find_package(OpenCV QUIET COMPONENTS imgcodecs)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
fing_package(tf2_geomestry_msgs REQUIRED)
find_package(vision_msgs REQUIRED)

add_executable(detection_stability_node src/detection_stability_node.cpp)
target_compile_features(detection_stability_node PUBLIC cxx_std_17)
target_include_directories(detection_stability_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(
  detection_stability_node
  geometry_msgs
  inha_interfaces
  message_filters
  rclcpp
  rclcpp_action
  sensor_msgs
  tf2
  tf2_ros
  vision_msgs
)

if(OpenCV_FOUND)
  find_package(CURL REQUIRED)
  target_compile_definitions(detection_stability_node PRIVATE DETECTION_STABILITY_HAVE_OPENCV=1)
  target_link_libraries(detection_stability_node ${OpenCV_LIBS} ${CURL_LIBRARIES})
endif()

install(
  TARGETS detection_stability_node
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY include/
  DESTINATION include
)

ament_package()
