##############################################################################
# @file  CMakeLists.txt
# @brief Build configuration of gondola command-line tool.
#
# Copyright (c) 2011, 2012 University of Pennsylvania. All rights reserved.
# See http://www.rad.upenn.edu/sbia/software/license.html or COPYING file.
#
# Contact: SBIA Group <sbia-software at uphs.upenn.edu>
##############################################################################

#--------------------------------------------------------------
# readmedicalimage
basis_add_library (readmedicalimage MEX ReadMedicalImage.cxx ReadMedicalImagePipeline.cxx)
basis_set_target_properties (
  readmedicalimage
  PROPERTIES
    PREFIX +gondola/
    MFILE  ReadMedicalImage.m
)
basis_target_link_libraries (readmedicalimage ${ITK_LIBRARIES})

#--------------------------------------------------------------
# writemedicalimage
basis_add_library (writemedicalimage MEX WriteMedicalImage.cxx)
basis_set_target_properties (
  writemedicalimage
  PROPERTIES
    PREFIX +gondola/
    MFILE  WriteMedicalImage.m
)
basis_target_link_libraries (writemedicalimage ${ITK_LIBRARIES})

#--------------------------------------------------------------
# blockdiag
basis_add_library (blockdiag.cxx MEX)
basis_set_target_properties (blockdiag PROPERTIES PREFIX +gondola/)
