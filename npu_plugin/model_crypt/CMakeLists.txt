set(CMAKE_CXX_STANDARD 11)

include_directories(./  )
aux_source_directory(./ lib_src)
set(target_name "model_crypt")
add_library(${target_name}  STATIC ${lib_src} )
set_property(TARGET ${target_name} PROPERTY POSITION_INDEPENDENT_CODE ON)

