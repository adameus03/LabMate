cmake_minimum_required(VERSION 3.10)
project(qps)
set(CMAKE_C_STANDARD 11)
set(BUILD_SHARED_LIBS OFF)

set(CMAKE_BUILD_TYPE Debug) # For debugging

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/plibsys)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/libusb-cmake)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/libqrencode)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/plibsys/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/libusb-cmake/libusb/libusb)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/libqrencode)
add_executable(${CMAKE_PROJECT_NAME} 
    ${CMAKE_CURRENT_SOURCE_DIR}/main.c
    ${CMAKE_CURRENT_SOURCE_DIR}/printer.c
    ${CMAKE_CURRENT_SOURCE_DIR}/lw400.c
    ${CMAKE_CURRENT_SOURCE_DIR}/qr_data_adapter.c
)

target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE plibsys usb-1.0 qrencode)

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
