cmake_minimum_required(VERSION 3.10)

project(camera_driver)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)

add_library(${PROJECT_NAME} 
    ${PROJECT_SOURCE_DIR}/src/camera_driver.cpp
    ${PROJECT_SOURCE_DIR}/src/inference.cpp
    ${PROJECT_SOURCE_DIR}/src/pnp_kalman.cpp
)

target_link_directories(${PROJECT_NAME} PRIVATE
	${CMAKE_CURRENT_BINARY_DIR}/lib
)

target_link_libraries(${PROJECT_NAME} PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR}/lib/libMvCameraControl.so
	${OpenCV_LIBS}
)
