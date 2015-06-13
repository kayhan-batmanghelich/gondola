##############################################################################
# @file  util.py
# @brief Common utilities for best parameter search modules.
#
# Copyright (c) 2011, 2012 University of Pennsylvania. All rights reserved.
# See http://www.rad.upenn.edu/sbia/software/license.html or COPYING file.
#
# Contact: SBIA Group <sbia-software at uphs.upenn.edu>
##############################################################################

import weka.core.Version
if not weka.core.Version().isOlder("3.7.0"):
    import weka.classifiers.evaluation.output.prediction.PlainText
import java.lang.StringBuffer

# ----------------------------------------------------------------------------
def get_buffer_for_predictions(instances=None):
    if weka.core.Version().isOlder("3.7.0"):
        buffer = java.lang.StringBuffer()
        return (buffer, buffer)
    else:
        buffer = java.lang.StringBuffer()
        output = weka.classifiers.evaluation.output.prediction.PlainText()
        output.setHeader(instances)
        output.setBuffer(buffer)
        return (output, buffer)
