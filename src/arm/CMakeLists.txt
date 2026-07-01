# 机械臂视觉包的 CMake 构建入口，负责组织推理库、视觉算法库和 ROS2 节点的编译与安装。
cmake_minimum_required(VERSION 3.8)
project(arm)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 未显式指定构建类型时默认走 Release（-O3 -DNDEBUG），保证视觉算法以优化代码运行。
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
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
target_compile_definitions(arm_node PRIVATE ARM_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(arm_node arm_vision arm_inference ${OpenCV_LIBS})
target_compile_features(arm_node PUBLIC cxx_std_17)
ament_target_dependencies(arm_node ament_index_cpp geometry_msgs rcl_interfaces rclcpp std_msgs)

install(TARGETS arm_node arm_inference arm_vision
  DESTINATION lib/${PROJECT_NAME})

install(FILES best.onnx
  DESTINATION share/${PROJECT_NAME}/models)

install(DIRECTORY include/
  DESTINATION include)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
