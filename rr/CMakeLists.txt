cmake_minimum_required(VERSION 3.10)
project(rr)
set(CMAKE_C_STANDARD 11)
set(BUILD_SHARED_LIBS OFF)

set(CMAKE_BUILD_TYPE Debug) # For debugging
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#optional leak detection
option(FSANITIZE_LA  "Enable leak detection and address sanitizer" OFF)
if(FSANITIZE_LA)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address -fsanitize=leak")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address -fsanitize=leak")
    #add_compile_options(-fsanitize=address)
    #add_link_options(-fsanitize=address)
    message(STATUS "Address sanitizer and leak detection enabled")
else()
    message(STATUS "Address sanitizer and leak detection disabled")
endif()

#optional static build
option(STATIC_BUILD "Build static binary" OFF)
if(STATIC_BUILD)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static")
    message(STATUS "Static build enabled")
else()
    message(STATUS "Static build disabled")
endif()

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
pkg_check_modules(HIREDIS REQUIRED hiredis)
pkg_check_modules(PLIBSYS REQUIRED plibsys)
pkg_check_modules(CURL REQUIRED libcurl)

message(STATUS "hiredis version is: ${HIREDIS_VERSION}")
message(STATUS "plibsys version is: ${PLIBSYS_VERSION}")
message(STATUS "libcurl version is ${CURL_VERSION}")

#add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/plibsys)
#include_directories(${CMAKE_CURRENT_SOURCE_DIR}/plibsys/src)

include_directories(${HIREDIS_INCLUDE_DIRS})
include_directories(${PLIBSYS_INCLUDE_DIRS})
include_directories(${CURL_INCLUDE_DIRS})

add_executable(${CMAKE_PROJECT_NAME} 
    ${CMAKE_CURRENT_SOURCE_DIR}/main.c
    ${CMAKE_CURRENT_SOURCE_DIR}/mailer.c
    ${CMAKE_CURRENT_SOURCE_DIR}/log.c
)

#target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE plibsys)
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE ${HIREDIS_LIBRARIES} ${PLIBSYS_LIBRARIES} ${CURL_LIBRARIES})
