cmake_minimum_required(VERSION 3.8)
project(arithmetic_problem)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ============ 依赖 ============
find_package(ament_cmake REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(OpenCV REQUIRED)

# 子项目：海康相机 + ONNX 推理封装
# include/camera_driver 是一个独立 CMake 子项目，编译为 camera_driver 静态库
add_subdirectory(include/camera_driver)

# ============ calculate 库 ============
# 把"取帧 + 推理 + 筛选 + 算式 + 众数统计"封装到一个共享库
add_library(calculate
  src/calculate.cpp
)

target_include_directories(calculate PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/camera_driver/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(calculate PUBLIC
  camera_driver
  ${OpenCV_LIBS}
)

target_compile_features(calculate PUBLIC cxx_std_17)

# ============ 主可执行：ROS2 节点 ============
add_executable(arithmetic_problem
  src/arithmetic_problem_node.cpp
  src/main.cpp
)

target_include_directories(arithmetic_problem PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_compile_features(arithmetic_problem PUBLIC c_std_99 cxx_std_17)

target_compile_definitions(arithmetic_problem PRIVATE
  ARITHMETIC_PROBLEM_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(arithmetic_problem
  calculate
)

ament_target_dependencies(arithmetic_problem
  rcl_interfaces
  rclcpp
  std_msgs
)

# ============ 安装 ============
install(TARGETS arithmetic_problem
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS calculate
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/arithmetic_problem
  DESTINATION include
)

# 海康 SDK 动态库一并装上，运行期 dlopen 才能找到
install(DIRECTORY include/camera_driver/lib/
  DESTINATION lib
  FILES_MATCHING PATTERN "*.so*"
)

# ============ 测试 ============
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
