##############################################################################
# @file  smobagging.py
# @brief Best parameter search for Boosted SMO classifier.
#
# Copyright (c) 2011, 2012 University of Pennsylvania. All rights reserved.
# See http://www.rad.upenn.edu/sbia/software/license.html or COPYING file.
#
# Contact: SBIA Group <sbia-software at uphs.upenn.edu>
##############################################################################

import sys

import java.io.FileReader as FileReader
import weka.core.Instances as Instances
import weka.classifiers.functions.SMO as SMO
import weka.classifiers.functions.supportVector.PolyKernel as PolyKernel
import weka.classifiers.functions.supportVector.RBFKernel as RBFKernel
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
# searching for the best parameters for the boosted SMO
# this finction search for two kinds of boosting
#   1) it tries to boost the best SMO bu grid-searching for bagSizePercent and Iteration
#   2) it tries to boost linear SMO by grid-searching for the best c-value and num Iteration of the bagging procedure
#  BestSMOIsRBFKernel : it is a boolean value variable which determines the best smo was a RBF kernel or Poly kernel. 
#  param1, param2: based on values of the BestSMOIsRBFKernel, they can be interpreted as follows:
#   BestSMOIsRBFKernel = true   ---> param1 is c-value and param2 is gamma value
#   BestSMOIsRBFKernel = false  ---> param1 is c-value and param2 is exponent value
def BaggingSMO_ParamFinder(data, BestSMOIsRBFKernel, param1, param2):
    # Possible set for C-value
    cBounds = [[1,10,1],[10,100,10],[100,300,20]]
    # possible set bag size percent
    BagSizePercentBound = [ max(10, int(float(1)/float(data.numInstances())*100)+1 )  ,100,10]    # max operation is to make sure that least number of samples are provided to the classifier
    # possible set for Iteration
    ItrBound = [5,50,5]
    # This section tries to boost the best smo
    print "searching for the best parameters to Bag the best SMO ...."
    gridsearch = GridSearch()
    acctag = gridsearch.getEvaluation()
    acctag = SelectedTag('ACC',acctag.getTags())
    gridsearch.setEvaluation(acctag)
    allfilters = AllFilters()
    gridsearch.setFilter(allfilters)
    gridsearch.setGridIsExtendable(Boolean(False))
    smo = SMO()
    bagging = Bagging()
    if BestSMOIsRBFKernel:
        kernel = RBFKernel()
        kernel.setGamma(param2)
        smo.setKernel(kernel)
        smo.setC(param1)
    else:
        kernel = PolyKernel()
        kernel.setExponent(param2)
        smo.setKernel(kernel)
        smo.setC(param1)
    bagging.setClassifier(smo)
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
    print "searching for best parameters for bagging SMO bagSizePercent = [", BagSizePercentBound[0], ",", BagSizePercentBound[1], "], # Iteration = [", ItrBound[0], ",", ItrBound[1], "] ...."
    gridsearch.buildClassifier(data)
    #bestbagging1 = gridsearch.getBestClassifier()
    bestValues1 = gridsearch.getValues()
    # ------------------ Evaluation
    smo = SMO()
    bestbagging1 = Bagging()
    smo.setKernel(kernel)
    smo.setC(param1)
    bestbagging1.setBagSizePercent(int(bestValues1.x))
    bestbagging1.setNumIterations(int(bestValues1.y))
    bestbagging1.setClassifier(smo)
    evaluation = Evaluation(data)
    output = util.get_buffer_for_predictions()[0]
    attRange = Range()  # no additional attributes output
    outputDistribution = Boolean(False)  # we don't want distribution
    random = Random(1)
    numFolds = min(10,data.numInstances())
    evaluation.crossValidateModel(bestbagging1,data,numFolds,random,[output, attRange, outputDistribution])
    best_acc1 = evaluation.pctCorrect()
    bestValues1 = gridsearch.getValues()
    print "best accuracy by bagging the optimal SMO classifier: ", best_acc1
    print "Optimal Bag size Percent : ", bestValues1.x , "Optimal number of Iteration : ", bestValues1.y
    print "-----------------------------------------"
    # ------------------------------------------------------------------------------------------------------------------------
    # in this section we set the weak classifier to the linear SMO and optimize over c-value of the SMO and number of iteration  
    smo = SMO()
    kernel = PolyKernel()
    smo.setKernel(kernel)
    bagging.setClassifier(smo)
    gridsearch.setClassifier(bagging)
    gridsearch.setXProperty(String('classifier.classifier.c'))
    gridsearch.setYProperty(String('classifier.numIterations'))
    gridsearch.setXExpression(String('I'))
    gridsearch.setYExpression(String('I'))
    gridsearch.setGridIsExtendable(Boolean(True))
    best_acc2 = -float('inf')
    for cnt in range(0,len(cBounds)):
        cbound = cBounds[cnt]
        cmin =  cbound[0]
        cmax =  cbound[1]
        cstep = cbound[2]           
        gridsearch.setXMin(cmin)
        gridsearch.setXMax(cmax)
        gridsearch.setXStep(cstep)
        gridsearch.setYMin(ItrBound[0])
        gridsearch.setYMax(ItrBound[1])
        gridsearch.setYStep(ItrBound[2])
        print "searching for RBF Kernel C = [", cmin, ",", cmax, "], # Iteration = [", ItrBound[0], ",", ItrBound[1], "] ...."
        gridsearch.buildClassifier(data)
        bestValues = gridsearch.getValues()
        # ------------ Evaluation
        smo = SMO()
        bestbagging = Bagging()
        kernel = PolyKernel()
        smo.setKernel(kernel)
        smo.setC(bestValues.x)
        bestbagging.setNumIterations(int(bestValues.y))
        bestbagging.setClassifier(smo)
        evaluation = Evaluation(data)
        output = util.get_buffer_for_predictions()[0]
        attRange = Range()  # no additional attributes output
        outputDistribution = Boolean(False)  # we don't want distribution
        random = Random(1)
        numFolds = min(10,data.numInstances())
        evaluation.crossValidateModel(bestbagging,data,numFolds,random,[output, attRange, outputDistribution])
        acc = evaluation.pctCorrect()
        if (acc>best_acc2):
            bestbagging2 = bestbagging
            best_acc2 = acc
            bestValues2 = bestValues
            print "Best accuracy so far by bagging linear SMO: ", best_acc2 
            print "Best values so far by bagging linear SMO:   ", bestValues2 
    print "Best accuracy by bagging linear SMO: ", best_acc2
    print "Best values by bagging linear SMO:   ", bestValues2
    print "-----------------------------------------"   
    print "Final optimal bagging classifier:"
    if (best_acc2 > best_acc1):
        print "     Best bagging is based on linear SMO with optimal c-value :", bestValues2.x, " optimal numIteration = ", bestValues2.y
        print "     optimal accuracy: ", best_acc2
        IsOptimalBaggingIsOptSMO = False    # is optimal bagging based on optimal SMO ?
        IsOptBagOnOptSMO = IsOptimalBaggingIsOptSMO
        OptBagSMO = bestbagging2
        OptBagSMOp1 = bestValues2.x
        OptBagSMOp2 = bestValues2.y
        OptBagSMOAcc = best_acc2
    else:
        print "     Best bagging is based on optimal SMO with optimal bagSizePercent :", bestValues1.x, " optimal numIteration = ", bestValues1.y
        print "     optimal accuracy: ", best_acc1
        IsOptimalBaggingIsOptSMO = True     # is optimal bagging based on optimal SMO ?
        IsOptBagOnOptSMO = IsOptimalBaggingIsOptSMO
        OptBagSMO = bestbagging1
        OptBagSMOp1 = bestValues1.x
        OptBagSMOp2 = bestValues1.y
        OptBagSMOAcc = best_acc1
    if IsOptBagOnOptSMO:
        Description = 'Bagging on optimal SMO classifier: OptBagSizePercent=' + str(OptBagSMOp1) + \
                ', OptNumIterations=' + str(OptBagSMOp2) + ', OptAcc=' + str(OptBagSMOAcc)
    else:
        Description = 'Bagging on linear SMO classifier: OptC=' + str(OptBagSMOp1) + \
                 ', OptNumIterations=' + str(OptBagSMOp2) + ', OptAcc=' + str(OptBagSMOAcc)
    return IsOptBagOnOptSMO, OptBagSMO,  OptBagSMOp1, OptBagSMOp2, OptBagSMOAcc, Description
    

