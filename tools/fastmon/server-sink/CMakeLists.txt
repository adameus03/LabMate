cmake_minimum_required(VERSION 3.0.0)
project(fastmon-server-sink VERSION 0.1.0 LANGUAGES C)

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