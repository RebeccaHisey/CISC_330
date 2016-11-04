cmake_minimum_required(VERSION 2.8.9)

project(C-Arm Fluoroscopy)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "http://slicer.org/slicerWiki/index.php/Documentation/Nightly/Extensions/C-Arm Fluoroscopy")
set(EXTENSION_CATEGORY "Examples")
set(EXTENSION_CONTRIBUTORS "Rebecca Hisey (Queen's University)")
set(EXTENSION_DESCRIPTION "This is an example of a simple extension")
set(EXTENSION_ICONURL "http://www.example.com/Slicer/Extensions/C-Arm Fluoroscopy.png")
set(EXTENSION_SCREENSHOTURLS "http://www.example.com/Slicer/Extensions/C-Arm Fluoroscopy/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a space separated string, a list or 'NA' if any

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(SimulateAndReconstruct)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_CPACK})
