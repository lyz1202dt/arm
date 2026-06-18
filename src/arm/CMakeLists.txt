# 机械臂视觉包的 CMake 构建入口，负责组织推理库、视觉算法库和 ROS2 节点的编译与安装。
cmake_minimum_required(VERSION 3.8)
project(arm)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(ament_cmake REQUIRED)
find_package(OpenCV REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

add_library(arm_inference
  src/inference.cpp
)
target_include_directories(arm_inference PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(arm_inference ${OpenCV_LIBS})
target_compile_features(arm_inference PUBLIC cxx_std_17)

add_library(arm_vision
  src/box_grid_detector.cpp
)
target_include_directories(arm_vision PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(arm_vision arm_inference ${OpenCV_LIBS})
target_compile_features(arm_vision PUBLIC cxx_std_17)

add_executable(arm_node src/arm_node.cpp)
target_include_directories(arm_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(arm_node arm_vision arm_inference ${OpenCV_LIBS})
target_compile_features(arm_node PUBLIC cxx_std_17)
ament_target_dependencies(arm_node rcl_interfaces rclcpp std_msgs)

install(TARGETS arm_node arm_inference arm_vision
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY include/
  DESTINATION include)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
