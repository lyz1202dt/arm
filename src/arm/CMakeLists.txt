# 机械臂视觉包的 CMake 构建入口，负责组织推理库、视觉算法库和 ROS2 节点的编译与安装。
cmake_minimum_required(VERSION 3.8)
project(arm)

# 未显式指定构建类型时默认 Release，确保 CPU 推理等热点代码开启编译器优化。
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "构建类型" FORCE)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
  # 针对当前主机指令集做高强度优化，显著加速 OpenCV DNN 的 CPU 推理与图像处理。
  add_compile_options($<$<CONFIG:Release>:-O3>)
  add_compile_options($<$<CONFIG:Release>:-march=native>)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(ament_cmake REQUIRED)
find_package(OpenCV REQUIRED)
find_package(geometry_msgs REQUIRED)
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
  src/pnp_detector.cpp
  src/color_pnp_detector.cpp
  src/coordinate_filter.cpp
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
ament_target_dependencies(arm_node geometry_msgs rcl_interfaces rclcpp std_msgs)

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
