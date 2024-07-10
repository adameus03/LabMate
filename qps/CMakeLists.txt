cmake_minimum_required(VERSION 3.10)
project(qps)
set(CMAKE_C_STANDARD 11)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/plibsys)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/plibsys/src)
add_executable(${CMAKE_PROJECT_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/main.c)

target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE plibsys)
#target_link_directories(${CMAKE_PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/plibsys/build)
