cmake_minimum_required(VERSION 2.8.9)

project(Homework330_1)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "http://slicer.org/slicerWiki/index.php/Documentation/Nightly/Extensions/Homework330_1")
set(EXTENSION_CATEGORY "Examples")
set(EXTENSION_CONTRIBUTORS "Rebecca Hisey (AnyWare Corp.)")
set(EXTENSION_DESCRIPTION "CISC 330 Assignment 1")
set(EXTENSION_ICONURL "http://www.example.com/Slicer/Extensions/Homework330_1.png")
set(EXTENSION_SCREENSHOTURLS "http://www.example.com/Slicer/Extensions/Homework330_1/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a space separated string, a list or 'NA' if any

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(Intersections)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_CPACK})
