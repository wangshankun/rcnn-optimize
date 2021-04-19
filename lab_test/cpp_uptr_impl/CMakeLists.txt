cmake_minimum_required(VERSION 3.18)
set(CMAKE_CXX_STANDARD 14)

#获取build目录的名字
string(REGEX REPLACE ".*/\(.*\)" "\\1" BUILDDIR ${CMAKE_BINARY_DIR})

#递归加入所有子目录为include搜索路径
function(include_sub_directories_recursively root_dir)
    if (IS_DIRECTORY ${root_dir})               # 当前路径是一个目录吗，是的话就加入到包含目录
        #        if (${root_dir} MATCHES "include")
        message("include dir: " ${root_dir})
        include_directories(${root_dir})
        #        endif()
    endif()

    file(GLOB ALL_SUB RELATIVE ${root_dir} ${root_dir}/*) # 获得当前目录下的所有文件，让如ALL_SUB列表中
    foreach(sub ${ALL_SUB})
        if (IS_DIRECTORY ${root_dir}/${sub} AND (NOT ("${sub}" STREQUAL ${BUILDDIR})))#除去build文件夹
            include_sub_directories_recursively(${root_dir}/${sub}) # 对子目录递归调用，包含
        endif()
    endforeach()
endfunction()

include_sub_directories_recursively(${CMAKE_HOME_DIRECTORY})

project(demo)

file(GLOB_RECURSE sources "*.cpp" "*.c") #递归查到所有c文件
file(GLOB_RECURSE headers "*.h" "*.hpp") #递归查到所有头文件
#查到build目录下的编译相关文件
file(GLOB_RECURSE builds "${BUILDDIR}/*.cpp" "${BUILDDIR}/*.c" "${BUILDDIR}/*.h" "${BUILDDIR}/*.hpp")

list(REMOVE_ITEM sources ${builds})#剔除build下收录的文件
list(REMOVE_ITEM headers ${builds})

add_executable(demo ${sources} ${headers})

