set(CMAKE_CXX_STANDARD 11)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}../../lib/ipc/)

set(target_name "demo_model")
add_library(${target_name} SHARED ${CMAKE_CURRENT_SOURCE_DIR})
