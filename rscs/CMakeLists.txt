cmake_minimum_required(VERSION 3.10)
project(rscs)
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
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/libusb-cmake)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/plibsys/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/libusb-cmake/libusb/libusb)
# Add FUSE include directories
include_directories(${FUSE_INCLUDE_DIRS})
add_executable(${CMAKE_PROJECT_NAME} 
    ${CMAKE_CURRENT_SOURCE_DIR}/old_main.c
    ${CMAKE_CURRENT_SOURCE_DIR}/uhfman.c
    ${CMAKE_CURRENT_SOURCE_DIR}/ypdr200.c
    ${CMAKE_CURRENT_SOURCE_DIR}/ch340.c
    ${CMAKE_CURRENT_SOURCE_DIR}/utils.c
    ${CMAKE_CURRENT_SOURCE_DIR}/mtwister.c
    ${CMAKE_CURRENT_SOURCE_DIR}/uhfd.c
)

target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE plibsys usb-1.0)
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE ${FUSE_LIBRARIES})

# include(ExternalProject)
# ExternalProject_Add(
#     libusb
#     SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/libusb
#     CONFIGURE_COMMAND cd ${CMAKE_CURRENT_SOURCE_DIR}/libusb && ./autogen.sh && ./configure --disable-shared --enable-static && cd -
#     BUILD_COMMAND cd ${CMAKE_CURRENT_SOURCE_DIR}/libusb && make && cd -
#     INSTALL_COMMAND ""
# )
# add_dependencies(${CMAKE_PROJECT_NAME} libusb)
# ExternalProject_Get_Property(libusb source_dir)
# message(STATUS "libusb source_dir: ${source_dir}")
# include_directories(${source_dir}/libusb)
# set (LIBUSB_LIBRARY ${CMAKE_CURRENT_SOURCE_DIR}/libusb/libusb/.libs/libusb-1.0.a)
# message(STATUS "libusb binary_dir: ${binary_dir}")

# target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE plibsys ${LIBUSB_LIBRARY})
# target_link_directories(${CMAKE_PROJECT_NAME} PRIVATE ${LIBUSB_LIBRARY})
