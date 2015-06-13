##############################################################################
# @file  simplelogisticadaboost.py
# @brief Best parameter search for Boosted Simple Logistic classifier.
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
import weka.classifiers.meta.AdaBoostM1 as AdaBoostM1
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
# searching for the best parameters for the boosted simple Logisitc
# this finction search for two kinds of boosting
#   1) it tries to boost the optimal simple Logistic by grid-searching for weight Threshold and number of Iteration
#   2) it tries to boost logistic by grid-searching for the best num Boosting Iterations and num Iteration of the boosting procedure
#   param1 is HeuristicStop and param2 is NumBoostingIterations
def AdaBoostedSimpleLogistic_ParamFinder(data, param1, param2):
    # Adaboost params: Possible set for Weight Threshold 
    WeightThresholdBounds = [99,100,1]
    # Adaboost params: possible set for NumIteration
    NumItrBound = [5,50,5]
    # Simple Logisitic params: Possible set for num of boosting
    NumBoostIterationBounds = [0,200,10]
    # This section tries to boost the best simple logistic
    print "searching for the best parameters to boosting on the optimal simple Logistic ...."
    gridsearch = GridSearch()
    acctag = gridsearch.getEvaluation()
    acctag = SelectedTag('ACC',acctag.getTags())
    gridsearch.setEvaluation(acctag)
    allfilters = AllFilters()
    gridsearch.setFilter(allfilters)
    gridsearch.setGridIsExtendable(Boolean(True))
    simplelogistic = SimpleLogistic()
    adaboostm = AdaBoostM1()
    simplelogistic.setHeuristicStop(param1)
    simplelogistic.setNumBoostingIterations(param2)
    adaboostm.setClassifier(simplelogistic)
    gridsearch.setClassifier(adaboostm)
    gridsearch.setXProperty(String('classifier.weightThreshold'))
    gridsearch.setYProperty(String('classifier.numIterations'))
    gridsearch.setXExpression(String('I'))
    gridsearch.setYExpression(String('I'))
    gridsearch.setXMin(WeightThresholdBounds[0])
    gridsearch.setXMax(WeightThresholdBounds[1])
    gridsearch.setXStep(WeightThresholdBounds[2])
    gridsearch.setYMin(NumItrBound[0])
    gridsearch.setYMax(NumItrBound[1])
    gridsearch.setYStep(NumItrBound[2])
    print "searching for best parameters for boosting simple Logistic weightThreshold = [", WeightThresholdBounds[0], ",", WeightThresholdBounds[1], "], # Iterations = [", NumItrBound[0], ",", NumItrBound[1], "] ...."
    gridsearch.buildClassifier(data)
    bestValues1 = gridsearch.getValues()
    # ------------------------------ Evaluation
    simplelogistic = SimpleLogistic()
    bestadaboostm1 = AdaBoostM1()
    simplelogistic.setHeuristicStop(param1)
    simplelogistic.setNumBoostingIterations(param2)
    bestadaboostm1.setWeightThreshold(int(bestValues1.x))
    bestadaboostm1.setNumIterations(int(bestValues1.y))
    bestadaboostm1.setClassifier(simplelogistic)
    evaluation = Evaluation(data)
    output = util.get_buffer_for_predictions()[0]
    attRange = Range()  # no additional attributes output
    outputDistribution = Boolean(False)  # we don't want distribution
    random = Random(1)
    numFolds = min(10,data.numInstances())
    evaluation.crossValidateModel(bestadaboostm1,data,numFolds,random,[output, attRange, outputDistribution])
    best_acc1 = evaluation.pctCorrect()
    print "best accuracy by boosting the optimal simple Logistic classifier: ", best_acc1
    print "Optimal weight Threshold  Percent : ", bestValues1.x , "Optimal number of Iterations : ", bestValues1.y
    print "-----------------------------------------"
    # -------------------------------------------------------------------------------------------------------------------------
    # in this section we set the weak classifier to the linear SMO and optimize over c-value of the SMO and number of iteration  
    simplelogistic = SimpleLogistic()
    adaboostm = AdaBoostM1()
    adaboostm.setClassifier(simplelogistic)
    gridsearch.setClassifier(adaboostm)
    gridsearch.setXProperty(String('classifier.classifier.numBoostingIterations'))
    gridsearch.setYProperty(String('classifier.numIterations'))
    gridsearch.setXExpression(String('I'))
    gridsearch.setYExpression(String('I'))
    gridsearch.setXBase(10)
    gridsearch.setXMin(NumBoostIterationBounds[0])
    gridsearch.setXMax(NumBoostIterationBounds[1])
    gridsearch.setXStep(NumBoostIterationBounds[2])
    gridsearch.setYMin(NumItrBound[0])
    gridsearch.setYMax(NumItrBound[1])
    gridsearch.setYStep(NumItrBound[2])
    print "searching for number of boosting Iterations bound  = [", NumBoostIterationBounds[0], ",", NumBoostIterationBounds[1], "], # Iteration = [", NumItrBound[0], ",", NumItrBound[1], "] ...."
    gridsearch.buildClassifier(data)
    bestValues2 = gridsearch.getValues()
    # ------------------ Evaluation
    simplelogistic = SimpleLogistic()
    bestadaboostm2 = AdaBoostM1()
    simplelogistic.setNumBoostingIterations(int(bestValues2.x))
    bestadaboostm2.setNumIterations(int(bestValues2.y))
    bestadaboostm2.setClassifier(simplelogistic)    
    evaluation = Evaluation(data)
    output = util.get_buffer_for_predictions()[0]
    attRange = Range()  # no additional attributes output
    outputDistribution = Boolean(False)  # we don't want distribution
    random = Random(1)
    numFolds = min(10,data.numInstances())
    evaluation.crossValidateModel(bestadaboostm2,data,numFolds,random,[output, attRange, outputDistribution])
    best_acc2 = evaluation.pctCorrect()
    print "best accuracy by boosting the Simple Logistic classifier (with optimization over ridge): ", best_acc2
    print "Optimal number of boosting Iteration : ", bestValues2.x , "Optimal number of Iteration : ", bestValues2.y
    print "-----------------------------------------"
    print "Final optimal boosting classifier:"
    if (best_acc2 > best_acc1):
        print "     Best boosting is based on simple logistic with optimal numBoostingIterations :",\
             bestValues2.x, " optimal numIteration :", bestValues2.y
        print "     optimal accuracy: ", best_acc2
        IsOptimalBoostingOnOptSimpleLogistic = False    # is optimal boosting based on optimal simple Logistic ?
        IsOptBoostOnOptSimpLog = IsOptimalBoostingOnOptSimpleLogistic
        OptBoostSimpLog = bestadaboostm2
        OptBoostSimpLogp1 = bestValues2.x
        OptBoostSimpLogp2 = bestValues2.y
        OptBoostSimpLogAcc = best_acc2
    else:
        print "     Best boosting is based on optimal simple Logistic with optimal weight Threshold :",\
             bestValues1.x, " optimal numIteration :", bestValues1.y
        print "     optimal accuracy: ", best_acc1
        IsOptimalBoostingOnOptSimpleLogistic = True # is optimal boosting based on optimal simple Logistic ?
        IsOptBoostOnOptSimpLog = IsOptimalBoostingOnOptSimpleLogistic
        OptBoostSimpLog = bestadaboostm1
        OptBoostSimpLogp1 = bestValues1.x
        OptBoostSimpLogp2 = bestValues1.y
        OptBoostSimpLogAcc = best_acc1
    if IsOptBoostOnOptSimpLog:
        Description = 'Boosting optimal simple logistic classifier: OptWeightThreshold = ' + \
            str(OptBoostSimpLogp1) + ', OptNumIterations=' + \
            str(OptBoostSimpLogp2) + ', OptAcc = ' + str(OptBoostSimpLogAcc)
    else:
        Description = 'Boosting simple logistic classifier: OptNumBoostingIterations = ' + \
            str(OptBoostSimpLogp1) + ', OptNumIterations=' + \
            str(OptBoostSimpLogp2) + ', OptAcc = ' + str(OptBoostSimpLogAcc)
    return IsOptBoostOnOptSimpLog, OptBoostSimpLog, OptBoostSimpLogp1, OptBoostSimpLogp2, \
            OptBoostSimpLogAcc, Description
