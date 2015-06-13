##############################################################################
# @file  simplelogistic.py
# @brief Best parameter search for Simple Logistic classifier.
#
# Copyright (c) 2011, 2012 University of Pennsylvania. All rights reserved.
# See http://www.rad.upenn.edu/sbia/software/license.html or COPYING file.
#
# Contact: SBIA Group <sbia-software at uphs.upenn.edu>
##############################################################################

import sys

import java.io.FileReader as FileReader
import weka.core.Instances as Instances
import weka.classifiers.functions.SimpleLogistic as SimpleLogistic
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
def myGridSearch(data,HsBounds,NumBoostBounds):
    best_acc = -float('inf')
    class bestValues(object):
        b = float('nan')
        h = float('nan')
    for h in range(HsBounds[0],HsBounds[1]+HsBounds[2],HsBounds[2]):
        for b in range(NumBoostBounds[0],NumBoostBounds[1]+NumBoostBounds[2],NumBoostBounds[2]):
            simplelogistic = SimpleLogistic()
            simplelogistic.setHeuristicStop(int(h))
            simplelogistic.setNumBoostingIterations(int(b))
            simplelogistic.setUseCrossValidation(False)
            evaluation = Evaluation(data)
            output = output = util.get_buffer_for_predictions()[0]
            attRange = Range()  # no additional attributes output
            outputDistribution = Boolean(False)  # we don't want distribution
            random = Random(1)
            numFolds = min(10,data.numInstances())
            evaluation.crossValidateModel(simplelogistic,data,numFolds,random,[output, attRange, outputDistribution])
            acc = evaluation.pctCorrect()
            if (acc>best_acc):
                bestsimplelogistic = simplelogistic
                best_acc = acc
                bestValues.b = b
                bestValues.h = h
    print "Best accuracy: ", best_acc
    print "Best values:   HsBounds = ", bestValues.h, ", NumBoostBounds = ", bestValues.b
    print "-----------------------------------------"
    return bestsimplelogistic, bestValues.h, bestValues.b, best_acc
 
# ----------------------------------------------------------------------------
# searching for the best parameters for the Logistic classifier
def SimpleLogistic_ParamFinder(data):   
    # Possible set for heuristic stop value
    HsBounds = [10,100,5]
    # Possible set for num of boosting
    NumBoostBounds = [0,100,10]
    if (data.numInstances()>10):     # grid search does 10-fold cross validation; hence number of samples must be more than 10
        gridsearch = GridSearch()
        acctag = gridsearch.getEvaluation()
        acctag = SelectedTag('ACC',acctag.getTags())
        gridsearch.setEvaluation(acctag)
        allfilters = AllFilters()
        gridsearch.setFilter(allfilters)
        gridsearch.setGridIsExtendable(Boolean(True))
        simplelogistic = SimpleLogistic()
        gridsearch.setClassifier(simplelogistic)
        gridsearch.setXProperty(String('classifier.heuristicStop'))
        gridsearch.setYProperty(String('classifier.numBoostingIterations'))
        gridsearch.setXExpression(String('I'))
        gridsearch.setYExpression(String('I'))
        gridsearch.setXMin(HsBounds[0])
        gridsearch.setXMax(HsBounds[1])
        gridsearch.setXStep(HsBounds[2])
        gridsearch.setYMin(NumBoostBounds[0])
        gridsearch.setYMax(NumBoostBounds[1])
        gridsearch.setYStep(NumBoostBounds[2])
        print "searching for simple logistic lcassifier heuristicStop = [", HsBounds[0], ",", HsBounds[1], "], Num of Boosting Bounds = [", NumBoostBounds[0], ",", NumBoostBounds[1], "] ...."
        gridsearch.buildClassifier(data)
        bestValues = gridsearch.getValues()
        # --------------------------- Evaluation
        bestsimplelogistic = SimpleLogistic()
        bestsimplelogistic.setHeuristicStop(int(bestValues.x))
        bestsimplelogistic.setNumBoostingIterations(int(bestValues.y))
        evaluation = Evaluation(data)
        output = util.get_buffer_for_predictions()[0]
        attRange = Range()  # no additional attributes output
        outputDistribution = Boolean(False)  # we don't want distribution
        random = Random(1)
        numFolds = min(10,data.numInstances())
        evaluation.crossValidateModel(bestsimplelogistic,data,numFolds,random,[output, attRange, outputDistribution])
        acc = evaluation.pctCorrect()
        print "best accuracy: ", acc
        print "best simple logistic classifier with Heuristic Stop=",bestsimplelogistic.getHeuristicStop() , "Num Boosting Iterations = ", bestsimplelogistic.getNumBoostingIterations()
        OptSimpLog = bestsimplelogistic
        OptSimpLogp1 = bestsimplelogistic.getHeuristicStop()
        OptSimpLogp2 = bestsimplelogistic.getNumBoostingIterations()
        OptSimpLogAcc = acc
    else:
        OptSimpLog, OptSimpLogp1, OptSimpLogp2, OptSimpLogAcc = myGridSearch(data,HsBounds,NumBoostBounds)
    Description = 'Simple logistic classifier: OptHeuristicStop= ' + str(OptSimpLogp1) + \
            ', OptNumBoostingIterations=' + str(OptSimpLogp2) + ', OptAcc = ' + str(OptSimpLogAcc)
    print "-----------------------------------------"
    return OptSimpLog, OptSimpLogp1, OptSimpLogp2, OptSimpLogAcc, Description
