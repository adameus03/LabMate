cmake_minimum_required(VERSION 3.10)
project(antc)
set(CMAKE_C_STANDARD 11)
set(BUILD_SHARED_LIBS OFF)

set(CMAKE_BUILD_TYPE Debug) # For debugging
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#########################################################################################
### Configurable options ################################################################
#########################################################################################

#make sure FSANITIZE_LA and FSANITIZE_T are mutually exclusive
if(FSANITIZE_LA AND FSANITIZE_T)
    message(FATAL_ERROR "FSANITIZE_LA and FSANITIZE_T are mutually exclusive")
endif()

#optional leak detection at runtime
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

#optional fuzzing
option(ENABLE_FUZZING "Enable fuzzing" OFF)
if(ENABLE_FUZZING)
    message(FATAL_ERROR "This is experimental, I don't know what I'm doing") #TODO: learn how to fuzz
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=fuzzer")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=fuzzer")
    #add_compile_options(-fsanitize=fuzzer)
    #add_link_options(-fsanitize=fuzzer)
    message(STATUS "Fuzzing enabled")
else()
    message(STATUS "Fuzzing disabled")
endif()

#optional thread sanitizer
option(FSANITIZE_T "Enable thread sanitizer" OFF)
if(FSANITIZE_T)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=thread")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=thread")
    #add_compile_options(-fsanitize=thread)
    #add_link_options(-fsanitize=thread)
    message(STATUS "Thread sanitizer enabled")
else()
    message(STATUS "Thread sanitizer disabled")
endif()

#optional cppcheck target
option(SUPPORT_CPPCHECK "Enable cppcheck target" OFF)
if (SUPPORT_CPPCHECK)
    find_program(CPPCHECK cppcheck REQUIRED)
    message("cppcheck found: ${CPPCHECK}")
    add_custom_target(cppcheck COMMAND ${CPPCHECK} --enable=all --inconclusive --std=c11 --language=c --inline-suppr --template=gcc --quiet --error-exitcode=1 ${CMAKE_CURRENT_SOURCE_DIR})
    message(STATUS "cppcheck target supported")
else()
    message(STATUS "cppcheck target not supported, because SUPPORT_CPPCHECK=OFF")
endif()

#optional static build
option(STATIC_BUILD "Build static binary" OFF)
if(STATIC_BUILD)
    message(FATAL_ERROR "We don't support static builds yet because of dependencies... If you want to try and fix this, go ahead")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static")
    message(STATUS "Static build enabled")
else()
    message(STATUS "Static build disabled")
endif()

#########################################################################################
### End of configurable options #########################################################
#########################################################################################

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
pkg_check_modules(GPIOD REQUIRED libgpiod)

message(STATUS "FUSE version is: ${FUSE_VERSION}")
message(STATUS "GPIOD version is: ${GPIOD_VERSION}")

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/plibsys)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/plibsys/src)
# Add FUSE include directories
include_directories(${FUSE_INCLUDE_DIRS})
# Add GPIOD include directories
include_directories(${GPIOD_INCLUDE_DIRS})

add_executable(${CMAKE_PROJECT_NAME} 
    ${CMAKE_CURRENT_SOURCE_DIR}/main.c
    ${CMAKE_CURRENT_SOURCE_DIR}/hmc349.c
    ${CMAKE_CURRENT_SOURCE_DIR}/antennactl.c
)

target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE plibsys)
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE ${FUSE_LIBRARIES} ${GPIOD_LIBRARIES})
