set(CMAKE_CXX_STANDARD 11)

PROJECT(jpg_decoder C CXX)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

file(GLOB_RECURSE sources "${CMAKE_CURRENT_SOURCE_DIR}/[a-zA-Z]*.c")

add_library(jpg_decoder STATIC ${sources})
set_property(TARGET jpg_decoder PROPERTY POSITION_INDEPENDENT_CODE ON)
set_target_properties(jpg_decoder PROPERTIES PUBLIC_HEADER "jfif.h")

INSTALL(TARGETS jpg_decoder DESTINATION ${CMAKE_INSTALL_LIBDIR})

