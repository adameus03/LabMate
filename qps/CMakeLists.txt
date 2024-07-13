cmake_minimum_required(VERSION 3.10)
project(qps)
set(CMAKE_C_STANDARD 11)
set(BUILD_SHARED_LIBS OFF)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/plibsys)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/plibsys/src)
add_executable(${CMAKE_PROJECT_NAME} 
    ${CMAKE_CURRENT_SOURCE_DIR}/main.c
    ${CMAKE_CURRENT_SOURCE_DIR}/printer.c
)


include(ExternalProject)
# Add the external make-based project
ExternalProject_Add(
    libusb
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/libusb
    CONFIGURE_COMMAND cd ${CMAKE_CURRENT_SOURCE_DIR}/libusb && ./autogen.sh && ./configure --disable-shared --enable-static && cd -
    BUILD_COMMAND cd ${CMAKE_CURRENT_SOURCE_DIR}/libusb && make && cd - #CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER}
    #BUILD_COMMAND ${CMAKE_MAKE_PROGRAM} -C ${CMAKE_CURRENT_SOURCE_DIR}/libusb
    #BUILD_COMMAND ${CMAKE_MAKE_PROGRAM}
    INSTALL_COMMAND ""
)
# Specify dependencies for your main project
add_dependencies(${CMAKE_PROJECT_NAME} libusb)
# # Include any necessary directories or link against libraries produced by the make-based project
# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/libusb/libusb)
# #???link_directories(${CMAKE_BINARY_DIR}/external/make_project/lib)
# Specify the include directory for libusb
ExternalProject_Get_Property(libusb source_dir)
#print out the source_dir
message(STATUS "libusb source_dir: ${source_dir}")
include_directories(${source_dir}/libusb)
# Specify the library for libusb
#?ExternalProject_Get_Property(libusb binary_dir)
#?set(LIBUSB_LIBRARY ${binary_dir}/libusb/.libs/libusb-1.0.a)
set (LIBUSB_LIBRARY ${CMAKE_CURRENT_SOURCE_DIR}/libusb/libusb/.libs/libusb-1.0.a)
# print out the binary_dir
message(STATUS "libusb binary_dir: ${binary_dir}")
#set(LIBUSB_LIBRARY ${})


target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE plibsys ${LIBUSB_LIBRARY})
target_link_directories(${CMAKE_PROJECT_NAME} PRIVATE ${LIBUSB_LIBRARY})
#target_link_directories(${CMAKE_PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/plibsys/build)

#target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE libusb)
