# Find an installed OpenUCX distribution and expose its UCP interface.
find_path(UCX_INCLUDE_DIR NAMES ucp/api/ucp.h)
find_library(UCX_UCP_LIBRARY NAMES ucp)
find_library(UCX_UCS_LIBRARY NAMES ucs)
find_library(UCX_UCT_LIBRARY NAMES uct)
find_library(UCX_UCM_LIBRARY NAMES ucm)

set(UCX_VERSION "")
if(UCX_INCLUDE_DIR AND EXISTS "${UCX_INCLUDE_DIR}/ucp/api/ucp_version.h")
  file(STRINGS "${UCX_INCLUDE_DIR}/ucp/api/ucp_version.h" UCX_VERSION_LINES
       REGEX "^#define UCP_API_(MAJOR|MINOR|RELEASE) ")
  foreach(PART MAJOR MINOR RELEASE)
    string(REGEX MATCH "#define UCP_API_${PART} +([0-9]+)" MATCHED
                 "${UCX_VERSION_LINES}")
    set(UCX_VERSION_${PART} "${CMAKE_MATCH_1}")
  endforeach()
  set(UCX_VERSION
      "${UCX_VERSION_MAJOR}.${UCX_VERSION_MINOR}.${UCX_VERSION_RELEASE}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  UCX
  REQUIRED_VARS
    UCX_INCLUDE_DIR
    UCX_UCP_LIBRARY
    UCX_UCS_LIBRARY
    UCX_UCT_LIBRARY
    UCX_UCM_LIBRARY
  VERSION_VAR UCX_VERSION)

mark_as_advanced(
  UCX_INCLUDE_DIR
  UCX_UCP_LIBRARY
  UCX_UCS_LIBRARY
  UCX_UCT_LIBRARY
  UCX_UCM_LIBRARY)

if(UCX_FOUND AND NOT TARGET UCX::ucp)
  add_library(UCX::ucs UNKNOWN IMPORTED)
  set_target_properties(
    UCX::ucs
    PROPERTIES
      IMPORTED_LOCATION "${UCX_UCS_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${UCX_INCLUDE_DIR}")

  add_library(UCX::uct UNKNOWN IMPORTED)
  set_target_properties(
    UCX::uct
    PROPERTIES
      IMPORTED_LOCATION "${UCX_UCT_LIBRARY}"
      INTERFACE_LINK_LIBRARIES UCX::ucs)

  add_library(UCX::ucm UNKNOWN IMPORTED)
  set_target_properties(
    UCX::ucm
    PROPERTIES
      IMPORTED_LOCATION "${UCX_UCM_LIBRARY}"
      INTERFACE_LINK_LIBRARIES UCX::ucs)

  add_library(UCX::ucp UNKNOWN IMPORTED)
  set_target_properties(
    UCX::ucp
    PROPERTIES
      IMPORTED_LOCATION "${UCX_UCP_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${UCX_INCLUDE_DIR}"
      INTERFACE_LINK_LIBRARIES "UCX::uct;UCX::ucm;UCX::ucs")
endif()
