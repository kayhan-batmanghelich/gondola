##############################################################################
# @file  logisticbagging.py
# @brief Best parameter search for Boosted Logistic classifier.
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
import weka.classifiers.meta.Bagging as Bagging
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
# searching for the best parameters for the boosted Logisitc
# this finction search for two kinds of boosting
#   1) it tries to boost the best Logistic by grid-searching for bagSizePercent and Iteration
#   2) it tries to logistic by grid-searching for the best Ridge-value and num Iteration of the bagging procedure
#   param1 is Ridge-value and param2 is maxIts value
def BaggingLogistic_ParamFinder(data, param1, param2):
    # Possible set for Ridge-value
    RBounds = [-10,2,1]
    # possible set bag size percent
    BagSizePercentBound = [ max(10, int(float(1)/float(data.numInstances())*100)+1 )  ,100,10]    # max operation is to make sure that least number of samples are provided to the classifier
    # possible set for Iteration
    ItrBound = [5,50,5]
    # This section tries to boost the best logistic
    print "searching for the best parameters to Bag the optimal Logistic ...."
    gridsearch = GridSearch()
    acctag = gridsearch.getEvaluation()
    acctag = SelectedTag('ACC',acctag.getTags())
    gridsearch.setEvaluation(acctag)
    allfilters = AllFilters()
    gridsearch.setFilter(allfilters)
    gridsearch.setGridIsExtendable(Boolean(False))
    logistic = Logistic()
    bagging = Bagging()
    logistic.setRidge(param1)
    logistic.setMaxIts(param2)
    bagging.setClassifier(logistic)
    gridsearch.setClassifier(bagging)
    gridsearch.setXProperty(String('classifier.bagSizePercent'))
    gridsearch.setYProperty(String('classifier.numIterations'))
    gridsearch.setXExpression(String('I'))
    gridsearch.setYExpression(String('I'))
    gridsearch.setXMin(BagSizePercentBound[0])
    gridsearch.setXMax(BagSizePercentBound[1])
    gridsearch.setXStep(BagSizePercentBound[2])
    gridsearch.setYMin(ItrBound[0])
    gridsearch.setYMax(ItrBound[1])
    gridsearch.setYStep(ItrBound[2])
    print "searching for best parameters for bagging Logistic bagSizePercent = [", BagSizePercentBound[0], ",", BagSizePercentBound[1], "], # Iteration = [", ItrBound[0], ",", ItrBound[1], "] ...."
    gridsearch.buildClassifier(data)
    #bestbagging1 = gridsearch.getBestClassifier()
    bestValues1 = gridsearch.getValues()
    # ------------------------------ Evaluation
    logistic = Logistic()
    bestbagging1 = Bagging()
    logistic.setRidge(param1)
    logistic.setMaxIts(param2)
    bestbagging1.setBagSizePercent(int(bestValues1.x))
    bestbagging1.setNumIterations(int(bestValues1.y))
    bestbagging1.setClassifier(logistic)
    evaluation = Evaluation(data)
    output = output = util.get_buffer_for_predictions()[0]
    attRange = Range()  # no additional attributes output
    outputDistribution = Boolean(False)  # we don't want distribution
    random = Random(1)
    numFolds = min(10,data.numInstances())
    evaluation.crossValidateModel(bestbagging1,data,numFolds,random,[output, attRange, outputDistribution])
    best_acc1 = evaluation.pctCorrect()
    print "best accuracy by bagging the optimal Logistic classifier: ", best_acc1
    print "Optimal Bag size Percent: ", bestValues1.x, " Optimal number of Iterations: ", bestValues1.y
    print "-----------------------------------------"
    # -------------------------------------------------------------------------------------------------------------------------
    # in this section we set the weak classifier to the linear SMO and optimize over c-value of the SMO and number of iteration  
    logistic = Logistic()
    bagging = Bagging()
    bagging.setClassifier(logistic)
    gridsearch.setClassifier(bagging)
    gridsearch.setXProperty(String('classifier.classifier.ridge'))
    gridsearch.setYProperty(String('classifier.numIterations'))
    gridsearch.setXExpression(String('pow(BASE,I)'))
    gridsearch.setYExpression(String('I'))
    gridsearch.setXBase(10)
    gridsearch.setGridIsExtendable(Boolean(True))
    gridsearch.setXMin(RBounds[0])
    gridsearch.setXMax(RBounds[1])
    gridsearch.setXStep(RBounds[2])
    gridsearch.setYMin(ItrBound[0])
    gridsearch.setYMax(ItrBound[1])
    gridsearch.setYStep(ItrBound[2])
    print "searching for ridge bound  = [10^", RBounds[0], ",10^", RBounds[1], "], # Iteration = [", ItrBound[0], ",", ItrBound[1], "] ...."
    gridsearch.buildClassifier(data)
    #bestbagging = gridsearch.getBestClassifier()
    bestValues2 = gridsearch.getValues()
    # ------------------ Evaluation
    logistic = Logistic()
    bestbagging2 = Bagging()
    logistic.setRidge(pow(10,bestValues2.x))
    bestbagging2.setNumIterations(int(bestValues2.y))
    bestbagging2.setClassifier(logistic)    
    evaluation = Evaluation(data)
    output = output = util.get_buffer_for_predictions()[0]
    attRange = Range()  # no additional attributes output
    outputDistribution = Boolean(False)  # we don't want distribution
    random = Random(1)
    numFolds = min(10,data.numInstances())
    evaluation.crossValidateModel(bestbagging2,data,numFolds,random,[output, attRange, outputDistribution])
    best_acc2 = evaluation.pctCorrect()
    print "best accuracy by bagging the Logistic classifier (with optimization over ridge): ", best_acc2
    print "Optimal Ridge value : ", bestValues2.x , "Optimal number of Iteration : ", bestValues2.y
    print "-----------------------------------------"
    print "Final optimal bagging classifier:"
    if (best_acc2 > best_acc1):
        print "     Best bagging is based on logistic with optimal ridge-value :", bestValues2.x, " optimal numIteration :", bestValues2.y
        print "     optimal accuracy: ", best_acc2
        IsOptimalBaggingIsOptLogistic = False   # is optimal bagging based on optimal Logistic ?
        IsOptBagOnOptLog = IsOptimalBaggingIsOptLogistic
        OptBagLog = bestbagging2
        OptBagLogp1 = pow(10,bestValues2.x)
        OptBagLogp2 = bestValues2.y
        OptBagLogAcc = best_acc2
    else:
        print "     Best bagging is based on optimal Logistic with optimal bagSizePercent :", bestValues1.x, " optimal numIteration :", bestValues1.y
        print "     optimal accuracy: ", best_acc1
        IsOptimalBaggingIsOptLogistic = True        # is optimal bagging based on optimal Logistic ?
        IsOptBagOnOptLog = IsOptimalBaggingIsOptLogistic
        OptBagLog = bestbagging1
        OptBagLogp1 = bestValues1.x
        OptBagLogp2 = bestValues1.y
        OptBagLogAcc = best_acc1
    if IsOptBagOnOptLog:
        Description = 'Bagging on optimal logistic classifier: OptBagSizePercent= ' + str(OptBagLogp1) + \
                ', OptNumIterations=' + str(OptBagLogp2) + ', OptAcc = ' + str(OptBagLogAcc)
    else:
        Description = 'Bagging on logistic classifier: OptRidge= ' + str(OptBagLogp1) + \
                ', OptNumIterations=' + str(OptBagLogp2) + ', OptAcc = ' + str(OptBagLogAcc)
    return IsOptBagOnOptLog, OptBagLog,  OptBagLogp1, OptBagLogp2, OptBagLogAcc, Description
