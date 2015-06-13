##############################################################################
# @file  randomforest.py
# @brief Best parameter search for Random Forest classifier.
#
# Copyright (c) 2011, 2012 University of Pennsylvania. All rights reserved.
# See http://www.rad.upenn.edu/sbia/software/license.html or COPYING file.
#
# Contact: SBIA Group <sbia-software at uphs.upenn.edu>
##############################################################################

import sys

import java.io.FileReader as FileReader
import weka.core.Instances as Instances
import weka.classifiers.trees.RandomForest as RandomForest
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
def myGridSearch(data,NTreeBounds,NFeaturesBounds):
    best_acc = -float('inf')
    bestrandomforest = None
    class bestValues(object):
        t = float('nan')
        f = float('nan')
    for t in range(NTreeBounds[0],NTreeBounds[1]+NTreeBounds[2],NTreeBounds[2]):
        for f in range(NFeaturesBounds[0],NFeaturesBounds[1]+NFeaturesBounds[2],NFeaturesBounds[2]):
            randomforest = RandomForest()
            randomforest.setNumTrees(int(t))
            randomforest.setNumFeatures(int(f))
            evaluation = Evaluation(data)
            output = output = util.get_buffer_for_predictions()[0]
            attRange = Range()  # no additional attributes output
            outputDistribution = Boolean(False)  # we don't want distribution
            random = Random(1)
            numFolds = min(10,data.numInstances())
            evaluation.crossValidateModel(randomforest,data,numFolds,random,[output, attRange, outputDistribution])
            acc = evaluation.pctCorrect()
            if (acc>best_acc):
                bestrandomforest = randomforest
                best_acc = acc
                bestValues.t = t
                bestValues.f = f
    print "Best accuracy:", best_acc
    print "Best values:  NTreeBounds = ", bestValues.t, ", NFeaturesBounds = ", bestValues.f
    print "-----------------------------------------"
    return bestrandomforest, bestValues.t, bestValues.f, best_acc
 
# ----------------------------------------------------------------------------
# searching for the best parameters for the Random Forest classifier
def RandomForest_ParamFinder(data): 
    # possible set for Number of trees
    NTreeBounds = [1,20,1]
    # possible set for number of features
    NFeaturesBounds = [0,20,1]
    if (data.numInstances()>10):     # grid search does 10-fold cross validation; hence number of samples must be more than 10
        gridsearch = GridSearch()
        acctag = gridsearch.getEvaluation()
        acctag = SelectedTag('ACC',acctag.getTags())
        gridsearch.setEvaluation(acctag)
        allfilters = AllFilters()
        gridsearch.setFilter(allfilters)
        gridsearch.setGridIsExtendable(Boolean(True))
        randomforest = RandomForest()
        gridsearch.setClassifier(randomforest)
        gridsearch.setXProperty(String('classifier.numTrees'))
        gridsearch.setYProperty(String('classifier.numFeatures'))
        gridsearch.setXExpression(String('I'))
        gridsearch.setYExpression(String('I'))
        gridsearch.setXMin(NTreeBounds[0])
        gridsearch.setXMax(NTreeBounds[1])
        gridsearch.setXStep(NTreeBounds[2])
        gridsearch.setYMin(NFeaturesBounds[0])
        gridsearch.setYMax(NFeaturesBounds[1])
        gridsearch.setYStep(NFeaturesBounds[2])
        gridsearch.setYBase(10)
        print "searching for random-forest NumTrees = [", NTreeBounds[0], ",", NTreeBounds[1], "], NumFeatures = [ ", NFeaturesBounds[0], ",", NFeaturesBounds[1], "] ...."
        gridsearch.buildClassifier(data)
        bestValues = gridsearch.getValues()
        # -----------------------  Evaluation
        bestrandomforest = RandomForest()
        bestrandomforest.setNumTrees(int(bestValues.x))
        bestrandomforest.setNumFeatures(int(bestValues.y))
        evaluation = Evaluation(data)
        output = output = util.get_buffer_for_predictions()[0]
        attRange = Range()  # no additional attributes output
        outputDistribution = Boolean(False)  # we don't want distribution
        random = Random(1)
        numFolds = min(10,data.numInstances())
        evaluation.crossValidateModel(bestrandomforest,data,numFolds,random,[output, attRange, outputDistribution])
        acc = evaluation.pctCorrect()
        print "best accuracy: ", acc
        print "best random-forest classifier with NumTrees=",bestValues.x , ", NumFeatures = ", bestValues.y
        OptRndFrst = bestrandomforest
        OptRndFrstp1 = bestValues.x
        OptRndFrstp2 = bestValues.y
        OptRndFrstAcc = acc
    else:
        OptRndFrst, OptRndFrstp1, OptRndFrstp2, OptRndFrstAcc = myGridSearch(data,NTreeBounds,NFeaturesBounds) 
    Description = 'Random-Forest classifier: OptNumTrees = ' + str(OptRndFrstp1) + \
            ', OptNumFeatures = ' + str(OptRndFrstp2) + ', OptAcc = ' + str(OptRndFrstAcc)
    print "-----------------------------------------"
    return OptRndFrst, OptRndFrstp1, OptRndFrstp2, OptRndFrstAcc, Description
