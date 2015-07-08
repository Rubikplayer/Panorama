include_directories( ${Panorama_SOURCE_DIR}/lib )
link_directories( ${Panorama_BINARY_DIR}/lib )

find_package( OpenCV REQUIRED )
#add_subdirectory( lib )

add_library( Panorama_lib ImageStitcher.cpp )
target_link_libraries( Panorama_lib ${OpenCV_LIBS} )

add_executable( panorama panorama.cpp )
target_link_libraries( panorama Panorama_lib ${OpenCV_LIBS} )