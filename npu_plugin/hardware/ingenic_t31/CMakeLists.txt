# ---[ jzdl
set(JZDL_COMPILER_VERSION "5.4.0")
set(CONAN_SETTINGS_COMPILER_LIBC "uclibc")
set(JZDL_LIB_PATH ${CMAKE_CURRENT_SOURCE_DIR}/jzdl/lib/${JZDL_COMPILER_VERSION}/${CONAN_SETTINGS_COMPILER_LIBC}/)
set(JZDL_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/jzdl/include/ ${CMAKE_CURRENT_SOURCE_DIR}/jzdl/include/mips-linux-gnu_5.4.0/)
include(${CMAKE_CURRENT_SOURCE_DIR}/jzdl.cmake)

################################################################################
## ---[ jzsdk
#set(JZSDK_LIB_PATH ${CMAKE_CURRENT_SOURCE_DIR}/jzsdk/${JZDL_COMPILER_VERSION}/${CONAN_SETTINGS_COMPILER_LIBC}/)
#include(${CMAKE_CURRENT_SOURCE_DIR}/jzsdk.cmake)
#
#
#list(APPEND 3RDPARTY_PREBUILD_SO_LIST "turbojpeg")
#add_library("turbojpeg" SHARED IMPORTED GLOBAL)
#set_target_properties(
#        "turbojpeg" PROPERTIES
#        IMPORTED_LOCATION ${CMAKE_HOME_DIRECTORY}/hardware/ingenic_t31/libjpeg-turbo/lib/libturbojpeg.so
#        INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_HOME_DIRECTORY}/hardware/ingenic_t31/libjpeg-turbo/include/
#        IMPORTED_NO_SONAME ON  # turn this on, otherwise will not run on board.
#)
#
#list(APPEND 3RDPARTY_PREBUILD_SO_LIST "jpeg")
#message("algosdk add lib: jpeg <--- ${CMAKE_HOME_DIRECTORY}/hardware/ingenic_t31/libjpeg-turbo/lib/libjpeg.so")
#add_library("jpeg" SHARED IMPORTED GLOBAL)
#set_target_properties(
#        "jpeg" PROPERTIES
#        IMPORTED_LOCATION ${CMAKE_HOME_DIRECTORY}/hardware/ingenic_t31/libjpeg-turbo/lib/libjpeg.a
#        INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_HOME_DIRECTORY}/hardware/ingenic_t31/libjpeg-turbo/include/
#        IMPORTED_NO_SONAME ON  # turn this on, otherwise will not run on board.
#)
##install(DIRECTORY ${CMAKE_HOME_DIRECTORY}/hardware/ingenic_t31/libjpeg-turbo/lib/ DESTINATION lib FILES_MATCHING PATTERN "lib*.so*" )
#
