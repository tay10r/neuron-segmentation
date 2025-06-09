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

set(sources
  app.cpp
  request.h
  deps/json/json.hpp
  deps/stb/stb.c
  deps/stb/stb_image.h
  deps/stb/stb_image_write.h
  deps/base64/base64.hpp
)

if(EMSCRIPTEN)
  list(APPEND sources request_browser.cpp)
else()
  list(APPEND sources request_desktop.cpp)
endif()

add_executable(neuron_segmentation_demo ${sources})

target_compile_features(neuron_segmentation_demo PUBLIC cxx_std_17)

target_include_directories(neuron_segmentation_demo
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/deps/base64"
    "${CMAKE_CURRENT_SOURCE_DIR}/deps/json"
    "${CMAKE_CURRENT_SOURCE_DIR}/deps/stb"
)

target_link_libraries(neuron_segmentation_demo
  PUBLIC
    uikit::uikit
    uikit::main
)

if(EMSCRIPTEN)
  set_target_properties(neuron_segmentation_demo
    PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../demo"
  )
  target_link_options(neuron_segmentation_demo PUBLIC "-sFETCH")
endif()
