cmake_minimum_required(VERSION 3.0.0)
project(fastmon-server-sink VERSION 0.1.0 LANGUAGES C)
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

add_executable(${CMAKE_PROJECT_NAME} 
            ${CMAKE_CURRENT_SOURCE_DIR}/visualizer.c 
            ${CMAKE_CURRENT_SOURCE_DIR}/../telemetry.c 
            ${CMAKE_CURRENT_SOURCE_DIR}/main.c)

INCLUDE(FindPkgConfig)

PKG_SEARCH_MODULE(SDL2 REQUIRED sdl2)
#https://stackoverflow.com/questions/58107854/cmake-is-unable-to-find-sdl2-ttf-im-trying-to-link-it-the-same-way-i-would-wit
PKG_SEARCH_MODULE(SDL2IMAGE REQUIRED SDL2_image>=2.0.0)
PKG_SEARCH_MODULE(SDL2TTF REQUIRED SDL2_ttf>=2.0.0)

INCLUDE_DIRECTORIES(${SDL2_INCLUDE_DIRS} ${SDL2IMAGE_INCLUDE_DIRS} ${SDL2TTF_INCLUDE_DIRS})
TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${SDL2_LIBRARIES} ${SDL2IMAGE_LIBRARIES} ${SDL2TTF_LIBRARIES} m)
