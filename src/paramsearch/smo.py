##############################################################################
# @file  smo.py
# @brief Best parameter search for SMO classifier.
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
def myGridSearch(data,cBounds,GBound,eBounds):
    IsBestRBFKernel = False
    best_acc_poly = -float('inf')
    best_acc_rbf = -float('inf')
    # Poly Kernel 
    class bestValues_poly(object):
        x = float('nan')
        y = float('nan')
    for Cbnd in cBounds:
        for c in range(Cbnd[0],Cbnd[1]+Cbnd[2],Cbnd[2]):
            for e in range(eBounds[0],eBounds[1]+eBounds[2],eBounds[2]):
                smo = SMO()
                kernel = PolyKernel()
                kernel.setExponent(e)
                smo.setC(c)
                smo.setKernel(kernel)
                evaluation = Evaluation(data)
                output = util.get_buffer_for_predictions()[0]
                attRange = Range()  # no additional attributes output
                outputDistribution = Boolean(False)  # we don't want distribution
                random = Random(1)
                numFolds = min(10,data.numInstances())
                evaluation.crossValidateModel(smo,data,numFolds,random,[output, attRange, outputDistribution])
                acc = evaluation.pctCorrect()
                if (acc>best_acc_poly):
                    best_smo_poly = smo
                    best_acc_poly = acc
                    bestValues_poly.x = c
                    bestValues_poly.y = e
    print "Best accuracy (Poly Kernel): ", best_acc_poly
    print "Best values (Poly Kernel):   C = ", bestValues_poly.x, ", exponent = ", bestValues_poly.y
    print "-----------------------------------------"
    # RBF Kernel
    class bestValues_rbf(object):
        x = float('nan')
        y = float('nan')
    for Cbnd in cBounds:
        for c in range(Cbnd[0],Cbnd[1]+Cbnd[2],Cbnd[2]):
            for g in range(GBound[0],GBound[1]+GBound[2],GBound[2]):
                smo = SMO()
                kernel = RBFKernel()
                kernel.setGamma(pow(10,g))
                smo.setC(c)
                smo.setKernel(kernel)
                evaluation = Evaluation(data)
                output = util.get_buffer_for_predictions()[0]
                attRange = Range()  # no additional attributes output
                outputDistribution = Boolean(False)  # we don't want distribution
                random = Random(1)
                numFolds = min(10,data.numInstances())
                evaluation.crossValidateModel(smo,data,numFolds,random,[output, attRange, outputDistribution])
                acc = evaluation.pctCorrect()
                if (acc>best_acc_rbf):
                    best_smo_rbf = smo
                    best_acc_rbf = acc
                    bestValues_rbf.x = c
                    bestValues_rbf.y = g 
    print "Best accuracy (RBF Kernel): ", best_acc_rbf
    print "Best values (RBF Kernel):   C = ", bestValues_rbf.x, ", gamma = ", bestValues_rbf.y
    if (best_acc_rbf > best_acc_poly):
        IsBestRBFKernel = True
        print "best smo classifier is RBF kernel with C = ", bestValues_rbf.x," and gamma = ", pow(10,bestValues_rbf.y)
        best_smo = best_smo_rbf
        OptSMOp1 = bestValues_rbf.x
        OptSMOp2 = pow(10,bestValues_rbf.y)
        OptSMOAcc = best_acc_rbf
        OptSMOIsRBF = IsBestRBFKernel
    else:
        IsBestRBFKernel = False
        print "best smo classifier is Poly kernel with C = ", bestValues_poly.x," and exponent = ", bestValues_poly.y
        best_smo = best_smo_poly
        OptSMOp1 = bestValues_poly.x
        OptSMOp2 = bestValues_poly.y
        OptSMOAcc = best_acc_poly
        OptSMOIsRBF = IsBestRBFKernel
    return IsBestRBFKernel, best_smo, OptSMOp1, OptSMOp2, OptSMOAcc

# ----------------------------------------------------------------------------
# searching for the best parameters for the SMO
def SMO_ParamFinder(data):
    # Possible set for C-value
    cBounds = [[1,10,1],[10,100,10],[100,300,20]]
    # possible set for exponents
    eBounds = [1,3,1]
    # possible set for Gamma
    GBound = [-5,2,1]
    if (data.numInstances()>10):     # grid search does 10-fold cross validation; hence number of samples must be more than 10
        # Polynomials Kernel
        gridsearch = GridSearch()
        acctag = gridsearch.getEvaluation()
        acctag = SelectedTag('ACC',acctag.getTags())
        gridsearch.setEvaluation(acctag)
        allfilters = AllFilters()
        gridsearch.setFilter(allfilters)
        gridsearch.setGridIsExtendable(Boolean(True))
        smo = SMO()
        kernel = PolyKernel()
        smo.setKernel(kernel)
        gridsearch.setClassifier(smo)
        gridsearch.setXProperty(String('classifier.c'))
        gridsearch.setYProperty(String('classifier.kernel.Exponent'))
        gridsearch.setXExpression(String('I'))
        gridsearch.setYExpression(String('I'))
        best_acc_poly = -float('inf')
        for cnt in range(0,len(cBounds)):
            cbound = cBounds[cnt]
            cmin =  cbound[0]
            cmax =  cbound[1]
            cstep = cbound[2]           
            gridsearch.setXMin(cmin)
            gridsearch.setXMax(cmax)
            gridsearch.setXStep(cstep)
            gridsearch.setYMin(eBounds[0])
            gridsearch.setYMax(eBounds[1])
            gridsearch.setYStep(eBounds[2])
            print "searching for Polykernel C = [", cmin, ",", cmax, "], exponent = [", eBounds[0], ",", eBounds[1], "] ...."
            gridsearch.buildClassifier(data)
            bestValues = gridsearch.getValues()
            # --------------------------------- Evaluation
            bestsmo = SMO()
            kernel = PolyKernel()
            kernel.setExponent(bestValues.y)
            bestsmo.setC(bestValues.x)
            bestsmo.setKernel(kernel)
            evaluation = Evaluation(data)
            output = util.get_buffer_for_predictions()[0]
            attRange = Range()  # no additional attributes output
            outputDistribution = Boolean(False)  # we don't want distribution
            random = Random(1)
            numFolds = min(10,data.numInstances())
            print "numFolds : ", numFolds
            evaluation.crossValidateModel(bestsmo,data,numFolds,random,[output, attRange, outputDistribution])
            acc = evaluation.pctCorrect()
            if (acc>best_acc_poly):
                best_smo_poly = bestsmo
                best_acc_poly = acc
                bestValues_poly = bestValues
                print "Best accuracy so far: ",best_acc_poly
                print "Best values so far:   ",bestValues_poly 
        print "Best accuracy (Poly Kernel): ", best_acc_poly
        print "Best values (Poly Kernel):   ", bestValues_poly
        print "-----------------------------------------"
        # RBF Kernel
        smo = SMO()
        kernel = RBFKernel()
        smo.setKernel(kernel)
        gridsearch.setClassifier(smo)
        gridsearch.setXProperty(String('classifier.c'))
        gridsearch.setYProperty(String('classifier.kernel.gamma'))
        gridsearch.setXExpression(String('I'))
        gridsearch.setYExpression(String('pow(BASE,I)'))
        gridsearch.setYBase(10)
        best_acc_rbf = -float('inf')
        for cnt in range(0,len(cBounds)):
            cbound = cBounds[cnt]
            cmin =  cbound[0]
            cmax =  cbound[1]
            cstep = cbound[2]           
            gridsearch.setXMin(cmin)
            gridsearch.setXMax(cmax)
            gridsearch.setXStep(cstep)
            gridsearch.setYMin(GBound[0])
            gridsearch.setYMax(GBound[1])
            gridsearch.setYStep(GBound[2])
            gridsearch.setYBase(10)
            print "searching for RBF Kernel C = [", cmin, ",", cmax, "], gamma = [10^", GBound[0], ",10^", GBound[1], "] ...."
            gridsearch.buildClassifier(data)
            bestValues = gridsearch.getValues()
            # ----------------------------------- Evaluation
            bestsmo = SMO()
            kernel = RBFKernel()
            kernel.setGamma(pow(10,bestValues.y))
            bestsmo.setC(bestValues.x)
            bestsmo.setKernel(kernel)
            evaluation = Evaluation(data)
            output = util.get_buffer_for_predictions()[0]
            attRange = Range()  # no additional attributes output
            outputDistribution = Boolean(False)  # we don't want distribution
            random = Random(1)
            numFolds = min(10,data.numInstances())
            evaluation.crossValidateModel(bestsmo,data,numFolds,random,[output, attRange, outputDistribution])
            acc = evaluation.pctCorrect()
            if (acc>best_acc_rbf):
                best_smo_rbf = bestsmo
                best_acc_rbf = acc
                bestValues_rbf = bestValues
                print "Best accuracy so far: ",best_acc_rbf
                print "Best values so far:   ",bestValues_rbf 
        print "Best accuracy (RBF Kernel): ", best_acc_rbf
        print "Best values (RBF Kernel):   ", bestValues_rbf
        print "-----------------------------------------" 
        if (best_acc_rbf > best_acc_poly):
            IsBestRBFKernel = True
            print "best smo classifier is RBF kernel with C = ", bestValues_rbf.x, " and gamma = ", pow(10,bestValues.y)
            best_smo = best_smo_rbf
            OptSMOp1 = bestValues_rbf.x
            OptSMOp2 = pow(10,bestValues.y)
            OptSMOAcc = best_acc_rbf
            OptSMOIsRBF = IsBestRBFKernel
        else:
            IsBestRBFKernel = False
            print "best smo classifier is Poly kernel with C = ", bestValues_poly.x, " and exponent = ", bestValues_poly.y
            best_smo = best_smo_poly
            OptSMOp1 = bestValues_poly.x
            OptSMOp2 = bestValues_poly.y
            OptSMOAcc = best_acc_poly
            OptSMOIsRBF = IsBestRBFKernel
    else:    # we have very small ssample size
        OptSMOIsRBF, best_smo, OptSMOp1, OptSMOp2, OptSMOAcc  = myGridSearch(data,cBounds,GBound,eBounds)
    if OptSMOIsRBF:
        Description = 'SMO classifier(RBF kernel): OptC=' + str(OptSMOp1) + \
                ', OptGamma=' + str(OptSMOp2) + ', OptAcc=' + str(OptSMOAcc) 
    else:
        Description = 'SMO classifier(Poly kernel): OptC=' + str(OptSMOp1) + \
                ', OptExponent=' + str(OptSMOp2) + ', OptAcc=' + str(OptSMOAcc)
    return OptSMOIsRBF, best_smo, OptSMOp1, OptSMOp2, OptSMOAcc, Description
