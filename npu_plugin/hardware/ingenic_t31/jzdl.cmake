include(GNUInstallDirs)

add_library(iaac SHARED IMPORTED GLOBAL)
set_target_properties(
  iaac
  PROPERTIES IMPORTED_LOCATION "${JZDL_LIB_PATH}/libiaac.so"
)
INSTALL(FILES "${JZDL_LIB_PATH}/libiaac.so" DESTINATION ${CMAKE_INSTALL_LIBDIR})

add_library("jzdl.m" SHARED IMPORTED GLOBAL)
set_target_properties(
  "jzdl.m"
  PROPERTIES IMPORTED_LOCATION "${JZDL_LIB_PATH}/libjzdl.m.so"
)
INSTALL(FILES "${JZDL_LIB_PATH}/libjzdl.m.so" DESTINATION ${CMAKE_INSTALL_LIBDIR})

add_library(jzdl INTERFACE)
set_target_properties(
  jzdl
  PROPERTIES INTERFACE_LINK_LIBRARIES "jzdl.m;iaac"
  INTERFACE_INCLUDE_DIRECTORIES "${JZDL_INCLUDE_PATH}"
)
