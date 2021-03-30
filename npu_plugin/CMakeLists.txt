cmake_minimum_required(VERSION 3.17)
set(CMAKE_CXX_STANDARD 11)

project(PluginMultipNpuFrameWork)

SET(CMAKE_CXX_COMPILER /home/shankun/.conan/data/ingenic-mips/r3.3.0/hci_toolchain/stable/package/80206e07321641d2362b7a56ceaf8254914e44f7/bin/mips-linux-uclibc-gnu-g++)
SET(CMAKE_C_COMPILER /home/shankun/.conan/data/ingenic-mips/r3.3.0/hci_toolchain/stable/package/80206e07321641d2362b7a56ceaf8254914e44f7/bin/mips-linux-uclibc-gnu-gcc)

add_subdirectory(hardware/ingenic_t31)
add_subdirectory(model/t31_ssd_pad_pfmh_0316)
add_subdirectory(model_crypt)
add_executable(t31_test t31_test.cpp)
target_include_directories(t31_test PUBLIC include)
target_link_libraries(t31_test PUBLIC dl)
install(TARGETS t31_test
        CONFIGURATIONS Debug
        RUNTIME DESTINATION bin)
