##############################################################################
# @file  CMakeLists.txt
# @brief Build configuration of main command-line tools.
#
# Copyright (c) 2011, 2012 University of Pennsylvania. All rights reserved.
# See http://www.rad.upenn.edu/sbia/software/license.html or COPYING file.
#
# Contact: SBIA Group <sbia-software at uphs.upenn.edu>
##############################################################################

basis_add_executable(gondola-config.py)   # generate folds and configuration files
basis_add_executable(gondola-crossval.py) # perform cross-validation experiments
basis_add_executable(gondola-sbia.sh)     # setup environment and run gondola (SBIA)

basis_target_link_libraries (
  gondola-crossval
    "${PythonModules_numpy_PATH}"
    "${PythonModules_scipy_PATH}"
    sbiautilities.pythonlib
)
