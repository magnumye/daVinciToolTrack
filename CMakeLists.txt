cmake_minimum_required(VERSION 2.8)

if(POLICY CMP0020)
  cmake_policy(SET CMP0020 NEW)
endif()

PROJECT(daVinciToolTrack)

list( APPEND CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")

# Find OpenCV
find_package( OpenCV REQUIRED )

find_package(VTK 6.0 REQUIRED)
include(${VTK_USE_FILE})

# Instruct CMake to run moc automatically when needed.
set(CMAKE_AUTOMOC ON)
find_package(Qt5Widgets REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR})

file(GLOB UI_FILES src/*.ui)

SET( CXX_FILES
	src/camera.cpp
	src/GraphicalUserInterface.cpp
	src/main.cpp
	src/mouse_interactor_style.cpp
	src/reader_camera.cpp
	src/reader_robot.cpp
	src/simulator.cpp
	src/stereo_base.cpp
	src/BasicStructs.cpp
	src/QGODetector.cpp
	src/QuantisedGradOrientation.cpp
	src/ShapeContextPro.cpp
	src/ExtendedKalmanFilter.cpp
	src/CameraProcessor.cpp
	src/psm_tool.cpp
)
SET( QT_WRAP
	include/camera.h
	include/GraphicalUserInterface.h
	include/mouse_interactor_style.h
	include/reader_camera.h
	include/reader_robot.h
	include/simulator.h
	include/stereo_base.h
	include/BasicStructs.h
	include/QGODetector.h
	include/QuantisedGradOrientation.h
	include/ShapeContextPro.h
	include/ExtendedKalmanFilter.h
	include/CameraProcessor.h
	include/psm_tool.h
)

qt5_wrap_ui(UISrcs ${UI_FILES} )
# CMAKE_AUTOMOC in ON so the MocHdrs will be automatically wrapped.
add_executable(daVinciToolTrack MACOSX_BUNDLE
  ${CXX_FILES} ${UISrcs} ${QT_WRAP})
qt5_use_modules(daVinciToolTrack Core Gui Widgets)

target_link_libraries(daVinciToolTrack ${VTK_LIBRARIES} ${OpenCV_LIBS})
