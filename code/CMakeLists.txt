cmake_minimum_required(VERSION 3.18)

project(neuron_segmentation_demo)

if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

include(FetchContent)

FetchContent_Declare(uikit
  URL "https://github.com/tay10r/uikit/archive/refs/tags/v0.2.2.zip"
  URL_HASH "SHA256=8e64d82bc922a0a27b124ac1466cba30a09f8213dca67061b5cab31782c0ec36"
)
FetchContent_MakeAvailable(uikit)

add_executable(neuron_segmentation_demo
  app.cpp
)

target_link_libraries(neuron_segmentation_demo
  PUBLIC
    uikit::uikit
    uikit::main
)

set_target_properties(neuron_segmentation_demo
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../demo"
)
