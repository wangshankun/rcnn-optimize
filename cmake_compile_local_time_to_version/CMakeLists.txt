cmake_minimum_required(VERSION 3.0)

string(TIMESTAMP LOCAL_COMPILE_TIME "%Y_%m_%d_%H_%M_%S")
MESSAGE(STATUS "LOCAL_COMPILE_TIME is time " ${LOCAL_COMPILE_TIME})
add_definitions(-DVCS_VERSION_INFO=\"\(${LOCAL_COMPILE_TIME}\)\")

#add_definitions(-DVCS_VERSION_INFO=${LOCAL_TIME})
#add_definitions(-DVCS_VERSION_INFO="afasdfsdflsjl")


add_executable(hello hello.cpp)
