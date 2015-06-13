##############################################################################
# @file  CMakeLists.txt
# @brief Build configuration of software.
#
# Copyright (c) 2011, 2012 University of Pennsylvania. All rights reserved.
# See http://www.rad.upenn.edu/sbia/software/license.html or COPYING file.
#
# Contact: SBIA Group <sbia-software at uphs.upenn.edu>
##############################################################################

#-----------------------------------------------------------------------------
# gondola
#
# The gondola package can be either used directly in MATLAB, installed as
# sources by setting BASIS_COMPILE_MATLAB to OFF, or be compiled using
# the MATLAB Compiler. Note that the + prefix is required to indicate that
# this is a MATLAB package.
add_subdirectory (+gondola)

basis_add_executable (gondola.m +gondola/*.m)
basis_set_target_properties (gondola PROPERTIES COMPILE_FLAGS "${BASIS_MCC_FLAGS} -R -nodisplay")
basis_target_link_libraries (
  gondola
    blockdiag
    readmedicalimage
    writemedicalimage
    ${LIBLINEAR_MEX_FILES}
    ${MOSEK_MEX_FILES}
)

#-----------------------------------------------------------------------------
# scripts
basis_add_library (paramsearch)
basis_target_link_libraries(paramsearch "${Weka_CLASSPATH}")

add_subdirectory (tools)
add_subdirectory (main)
