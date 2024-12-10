cmake_minimum_required(VERSION 3.10)
project(antc)
set(CMAKE_C_STANDARD 11)
set(BUILD_SHARED_LIBS OFF)

set(CMAKE_BUILD_TYPE Debug) # For debugging
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#option(LITTLE_ENDIAN "Little endian" ON)
include (TestBigEndian)
TEST_BIG_ENDIAN(IS_BIG_ENDIAN)
if(IS_BIG_ENDIAN)
 message(STATUS "Target is BIG ENDIAN")
else()
 message(STATUS "Target is LITTLE ENDIAN")
endif()
if(IS_BIG_ENDIAN)
    add_definitions(-DTARGET_BIG_ENDIAN) # For big endian
else()
    add_definitions(-DTARGET_LITTLE_ENDIAN) # For little endian
endif()

# Find FUSE
find_package(PkgConfig REQUIRED)
pkg_check_modules(FUSE REQUIRED fuse3)

message(STATUS "FUSE version is: ${FUSE_VERSION}")

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/plibsys)
#add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/libgpiod)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/plibsys/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/libgpiod/include)
# Add FUSE include directories
include_directories(${FUSE_INCLUDE_DIRS})
add_executable(${CMAKE_PROJECT_NAME} 
    ${CMAKE_CURRENT_SOURCE_DIR}/main.c
)

target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE plibsys libgpiod)
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE ${FUSE_LIBRARIES})
