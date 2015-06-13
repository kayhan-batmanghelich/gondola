##############################################################################
# @file  logistic.py
# @brief Best parameter search for Logistic classifier.
#
# Copyright (c) 2011, 2012 University of Pennsylvania. All rights reserved.
# See http://www.rad.upenn.edu/sbia/software/license.html or COPYING file.
#
# Contact: SBIA Group <sbia-software at uphs.upenn.edu>
##############################################################################

import sys

import java.io.FileReader as FileReader
import weka.core.Instances as Instances
import weka.classifiers.functions.Logistic as Logistic
import weka.classifiers.Evaluation as Evaluation
import weka.core.Range as Range
import weka.classifiers.meta.GridSearch as GridSearch
import weka.core.SelectedTag as SelectedTag
import weka.filters.AllFilter as AllFilters

import java.lang.StringBuffer as StringBuffer
import java.lang.String as String
import java.lang.Boolean as Boolean
import java.util.Random as Random

from . import util

# ----------------------------------------------------------------------------
def myGridSearch(data,RBound,MBound):
    bestlogistic = None
    best_acc     = -float('inf')
    class bestValues(object):
        m = float('nan')
        r = float('nan')
    for r in range(RBound[0],RBound[1]+RBound[2],RBound[2]):
        for m in range(MBound[0],MBound[1]+MBound[2],MBound[2]):
            logistic = Logistic()
            logistic.setMaxIts(int(m))
            logistic.setRidge(pow(10,r))
            evaluation = Evaluation(data)
            output = util.get_buffer_for_predictions()[0]
            attRange = Range()  # no additional attributes output
            outputDistribution = Boolean(False)  # we don't want distribution
            random = Random(1)
            numFolds = min(10,data.numInstances())
            evaluation.crossValidateModel(logistic,data,numFolds,random,[output, attRange, outputDistribution])
            acc = evaluation.pctCorrect()
            if (acc>best_acc):
                bestlogistic = logistic
                best_acc = acc
                bestValues.m = int(m)
                bestValues.r = pow(10,r)
    print "Best accuracy: ", best_acc
    print "Best values:   M = ", bestValues.m, ", Ridge = ", bestValues.r
    print "-----------------------------------------"
    return bestlogistic, bestValues.r, bestValues.m, best_acc

# ----------------------------------------------------------------------------
# searching for the best parameters for the Logistic classifier
def Logistic_ParamFinder(data): 
    # Possible set for Ridge-value
    RBounds = [-10,2,1]
    # possible set for maximum Iteration
    MBounds = [-1,10,1]
    if (data.numInstances()>10):     # grid search does 10-fold cross validation; hence number of samples must be more than 10
        gridsearch = GridSearch()
        acctag = gridsearch.getEvaluation()
        acctag = SelectedTag('ACC',acctag.getTags())
        gridsearch.setEvaluation(acctag)
        allfilters = AllFilters()
        gridsearch.setFilter(allfilters)
        gridsearch.setGridIsExtendable(Boolean(True))
        logistic = Logistic()
        gridsearch.setClassifier(logistic)
        gridsearch.setXProperty(String('classifier.maxIts'))
        gridsearch.setYProperty(String('classifier.ridge'))
        gridsearch.setXExpression(String('I'))
        gridsearch.setYExpression(String('pow(BASE,I)'))
        gridsearch.setXMin(MBounds[0])
        gridsearch.setXMax(MBounds[1])
        gridsearch.setXStep(MBounds[2])
        gridsearch.setYMin(RBounds[0])
        gridsearch.setYMax(RBounds[1])
        gridsearch.setYStep(RBounds[2])
        gridsearch.setYBase(10)
        print "searching for logistic lcassifier Max Iteration = [", MBounds[0], ",", MBounds[1], "], Ridge = [ 10E", RBounds[0], ",10E", RBounds[1], "] ...."
        gridsearch.buildClassifier(data)
        bestValues = gridsearch.getValues()
        # -----------------------  Evaluation
        bestlogistic = Logistic()
        bestlogistic.setMaxIts(int(bestValues.x))
        bestlogistic.setRidge(pow(10,bestValues.y))
        evaluation = Evaluation(data)
        output = util.get_buffer_for_predictions()[0]
        attRange = Range()  # no additional attributes output
        outputDistribution = Boolean(False)  # we don't want distribution
        random = Random(1)
        numFolds = min(10,data.numInstances())
        evaluation.crossValidateModel(bestlogistic,data,numFolds,random,[output, attRange, outputDistribution])
        acc = evaluation.pctCorrect()
        print "best accuracy: ", acc
        print "best logistic classifier with Ridge = ", bestlogistic.getRidge(), " Max Iteration = ", bestlogistic.getMaxIts()
        OptLog = bestlogistic
        OptLogp1 = bestlogistic.getRidge()
        OptLogp2 = bestlogistic.getMaxIts()
        OptLogAcc = acc
    else:
        OptLog, OptLogp1, OptLogp2, OptLogAcc = myGridSearch(data,RBounds,MBounds)
    Description = 'Logistic classifier OptRidge = ' + str(OptLogp1) + \
            ', OptMaxIts = ' + str(OptLogp2) + ', OptAcc = ' + str(OptLogAcc)
    print "-----------------------------------------"
    return OptLog, OptLogp1, OptLogp2, OptLogAcc, Description
