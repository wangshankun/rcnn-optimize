set(CMAKE_CXX_STANDARD 11)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../lib/ipc/)
file(GLOB_RECURSE sources "./[a-zA-Z]*.cpp")

set(target_name "t31_ssd_pad_pfmh_0316")
add_library(${target_name} SHARED ${sources})
set_property(TARGET ${target_name} PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(${target_name} PRIVATE 
                            ${CMAKE_CURRENT_SOURCE_DIR}
                            ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/model_crypt/)
target_link_libraries(${target_name} PRIVATE jzdl tracker_core tracker_common model_crypt)

get_filename_component(this_model_dir ${CMAKE_CURRENT_SOURCE_DIR} NAME)

install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
    DESTINATION ${CMAKE_INSTALL_PREFIX}/model/${this_model_dir}/
    FILES_MATCHING 
    PATTERN "*.so"
    PATTERN "*.bin"
    PATTERN "*.yaml"
)

install(TARGETS ${target_name}  
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/model/${this_model_dir}/
)
