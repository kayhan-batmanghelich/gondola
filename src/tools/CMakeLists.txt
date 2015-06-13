##############################################################################
# @file  CMakeLists.txt
# @brief Build configuration of command-line tools.
#
# Copyright (c) 2011, 2012 University of Pennsylvania. All rights reserved.
# See http://www.rad.upenn.edu/sbia/software/license.html or COPYING file.
#
# Contact: SBIA Group <sbia-software at uphs.upenn.edu>
##############################################################################

macro (add_weka_executable EXENAME)
  get_filename_component (TARGET_NAME "${EXENAME}" NAME_WE)
  basis_add_executable (${TARGET_NAME} "${EXENAME}" ${ARGN})
  basis_target_link_libraries (${TARGET_NAME} "${Weka_CLASSPATH}")
endmacro ()

basis_add_executable(createFoldsFromIdFnLists.py LIBEXEC)

add_weka_executable(mergeArffFiles.py                LIBEXEC)
add_weka_executable(wekaClassifier.py                LIBEXEC)
add_weka_executable(wekaClassifierWithProbability.py LIBEXEC)
add_weka_executable(wekaParamSearchForClassifier.py  LIBEXEC)

basis_target_link_libraries(wekaParamSearchForClassifier jythonlib)

if (Weka_attributeSelectionSearchMethods_CLASSPATH)
  add_weka_executable(wekaRankBasis.py LIBEXEC)
  basis_target_link_libraries (wekaRankBasis "${Weka_CLASSPATH}" "${Weka_attributeSelectionSearchMethods_CLASSPATH}")
endif ()
