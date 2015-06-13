##############################################################################
# @file  bayesian.py
# @brief Best parameter search for Bayesian classifier.
#
# Copyright (c) 2011, 2012 University of Pennsylvania. All rights reserved.
# See http://www.rad.upenn.edu/sbia/software/license.html or COPYING file.
#
# Contact: SBIA Group <sbia-software at uphs.upenn.edu>
##############################################################################

import sys

import weka.core.Version

import java.io.FileReader as FileReader
import weka.core.Instances as Instances
import weka.classifiers.bayes.NaiveBayes as NaiveBayes
import weka.classifiers.bayes.NaiveBayesMultinomial as NaiveBayesMultinomial 
import weka.classifiers.Evaluation as Evaluation
import weka.core.Range as Range
import weka.core.SelectedTag as SelectedTag

import java.lang.StringBuffer as StringBuffer
import java.lang.String as String
import java.lang.Boolean as Boolean
import java.util.Random as Random

from . import util

# ----------------------------------------------------------------------------
# this function check that all attributes are positive
def allAttributesPositive(data):
    OK = True
    minval = -float('inf')
    for inst_cnt in range(0,data.numInstances()):
        inst = data.instance(inst_cnt)
        for att_cnt in range(0,inst.numAttributes()):
            if (minval>inst.value(att_cnt)):
                OK = False
                return OK
    return OK

# ----------------------------------------------------------------------------
# searching for the best bayes classifier:
# since bayes classifiers do not have any parameter, it tries different configurations
def Bayes_ParamFinder(data):
    # -----------------------  Evaluation of Naive Bayes without kernel estimation
    naivebayes = NaiveBayes()
    evaluation = Evaluation(data)
    output = util.get_buffer_for_predictions()[0]
    attRange = Range()  # no additional attributes output
    outputDistribution = Boolean(False)  # we don't want distribution
    random = Random(1)
    numFolds = min(10,data.numInstances())
    evaluation.crossValidateModel(naivebayes,data,numFolds,random,[output, attRange, outputDistribution])
    acc_naivebayes = evaluation.pctCorrect()
    print "Naive Bayesisn accuracy (without kernel density estimation): ", acc_naivebayes
    # -----------------------  Evaluation of Naive Bayes with kernel estimation
    naivebayes = NaiveBayes()
    naivebayes.setUseKernelEstimator(Boolean(True))   # use kernel density estimation
    evaluation = Evaluation(data)
    attRange = Range()  # no additional attributes output
    outputDistribution = Boolean(False)  # we don't want distribution
    random = Random(1)
    numFolds = min(10,data.numInstances())
    evaluation.crossValidateModel(naivebayes,data,numFolds,random,[output, attRange, outputDistribution])
    acc_naivebayes_withkernel = evaluation.pctCorrect()
    print "Naive Bayesisn accuracy (with kernel density estimation): ", acc_naivebayes_withkernel
    # -----------------------  Evaluation of Naive bayes multinomial
    naivebayesmultinomial = NaiveBayesMultinomial()
    evaluation = Evaluation(data)
    attRange = Range()  # no additional attributes output
    outputDistribution = Boolean(False)  # we don't want distribution
    random = Random(1)
    if (allAttributesPositive(data)):  # multinomial bayes classifier only work on positive attributes
        numFolds = min(10,data.numInstances())
        evaluation.crossValidateModel(naivebayesmultinomial,data,numFolds,random,[output, attRange, outputDistribution])
        acc_naivemultinomialbayes = evaluation.pctCorrect()
    else:
        acc_naivemultinomialbayes = 0
    print "Naive Multinomial Bayesisn accuracy : ", acc_naivemultinomialbayes
    # ------------------------- Comparision
    if (acc_naivemultinomialbayes > acc_naivebayes):
        if (acc_naivemultinomialbayes > acc_naivebayes_withkernel):
            IsOptMultinomialBayes = True
            IsOptNaiveKernelDensity = False
            acc = acc_naivemultinomialbayes
        else:
            IsOptMultinomialBayes = False
            IsOptNaiveKernelDensity = True
            acc = acc_naivebayes_withkernel
    else:
        if (acc_naivebayes > acc_naivebayes_withkernel):
            IsOptMultinomialBayes = False
            IsOptNaiveKernelDensity = False
            acc = acc_naivebayes
        else:
            IsOptMultinomialBayes = False
            IsOptNaiveKernelDensity = True
            acc = acc_naivebayes_withkernel
    print "-----------------------------------------"
    OptBayesAcc = acc
    if IsOptMultinomialBayes:
        Description = 'Optimal Bayes classifier is Multinomial Bayes: OptAcc = ' + str(OptBayesAcc)
    elif IsOptNaiveKernelDensity:
        Description = 'Optimal Bayes classifier is Naive Bayes with kernel density estimation: OptAcc = ' +\
             str(OptBayesAcc)
    else:
        Description = 'Optimal Bayes classifier is Naive Bayes: OptAcc = ' + str(OptBayesAcc)
    return IsOptMultinomialBayes, IsOptNaiveKernelDensity, OptBayesAcc, Description
