cmake_minimum_required(VERSION 3.10)
project(labserv)
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

find_package(PkgConfig REQUIRED)
pkg_check_modules(UV REQUIRED libuv)
pkg_check_modules(OPENSSL REQUIRED openssl)
pkg_check_modules(H2O REQUIRED libh2o)
pkg_check_modules(YYJSON REQUIRED yyjson)
pkg_check_modules(PQ REQUIRED libpq)

message(STATUS "libuv version is: ${UV_VERSION}")
message(STATUS "openssl version is: ${OPENSSL_VERSION}")
message(STATUS "libh2o version is: ${H2O_VERSION}")
message(STATUS "yyjson version is: ${YYJSON_VERSION}")
message(STATUS "libpq version is: ${PQ_VERSION}")

#add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/plibsys)
#include_directories(${CMAKE_CURRENT_SOURCE_DIR}/plibsys/src)

include_directories(${UV_INCLUDE_DIRS} ${OPENSSL_INCLUDE_DIRS} ${H2O_INCLUDE_DIRS} ${YYJSON_INCLUDE_DIRS} ${PQ_INCLUDE_DIRS})

add_executable(${CMAKE_PROJECT_NAME} 
    ${CMAKE_CURRENT_SOURCE_DIR}/main.c
    ${CMAKE_CURRENT_SOURCE_DIR}/lsapi.c
)

#target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE plibsys)
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE ${UV_LIBRARIES} ${OPENSSL_LIBRARIES} ${H2O_LIBRARIES} ${YYJSON_LIBRARIES} ${PQ_LIBRARIES})
