#! /usr/bin/env jython

##############################################################################
# @file  wekaClassifier.py
# @brief Trains classifier using Weka.
#
# Copyright (c) 2011, 2012 University of Pennsylvania. All rights reserved.
# See http://www.rad.upenn.edu/sbia/software/license.html or COPYING file.
#
# Contact: SBIA Group <sbia-software at uphs.upenn.edu>
##############################################################################

# BASIS utilities
from gondola import basis

# General system imports
import sys
import os as os
import getopt
import os.path
import string as JyString

# General Java imports
import java.io.FileReader as FileReader
import java.lang.StringBuffer as StringBuffer
import java.lang.Boolean as Boolean
import java.lang.String as String

# General Weka imports
import weka.filters.Filter as Filter
import weka.core.Instances as Instances
import weka.core.Utils as Utils
import weka.core.AttributeStats as AttributeStats
import weka.classifiers.Evaluation as Evaluation
import weka.core.Range as Range
import weka.filters.unsupervised.instance.RemoveWithValues as RemoveWithValues
import weka.filters.unsupervised.attribute.Remove as AttributeRemove

# SMO imports 
import weka.classifiers.functions.SMO as SMO
import weka.classifiers.functions.supportVector.PolyKernel as PolyKernel
import weka.classifiers.functions.supportVector.RBFKernel as RBFKernel

# Logistic imports
import weka.classifiers.functions.Logistic as Logistic

# Simple Logisitc imports
import weka.classifiers.functions.SimpleLogistic as SimpleLogistic

# Random Forests imports
import weka.classifiers.trees.RandomForest as RandomForest

# Bayesian imports
import weka.classifiers.bayes.NaiveBayes as NaiveBayes
import weka.classifiers.bayes.NaiveBayesMultinomial as NaiveBayesMultinomial 

# Bagging imports
import weka.classifiers.meta.Bagging as Bagging

# AdaBoostM1 imports
import weka.classifiers.meta.AdaBoostM1 as AdaBoostM1

from gondola.paramsearch import util

# ============================================================================
# help
# ============================================================================

# ---------------------------------------------------------------------------- 
def usage():
    """usage information"""
    EXENAME = basis.exename()
    print """\
Usage:
  %(EXENAME)s [options]

Description:
  This code parses input arguments and train a classifiers from the following list
  of classifiers with determined parameters and apply it on test data. Then, it
  saves the results in two csv-files for test and train results.

  The available classifiers are:""" % {'EXENAME': EXENAME}
    for cls in CLASSIFIER.keys():
        print "  - %s" % cls
    print """
Options:
  [-r --trainArff]        Specifies the arff-file for training
  [-s --testArff]         Specifies the arff-file for testing
  [-b --bestClassifier]   Specifies the name of the best classifiers
  [-p --bestParam]        Specifies the best parameters for the classifiers (embraced inside of parentheses)
  [-l --removeLabel]      Specifies the labels which to be removed
  [-i --trainCSV]         Specifies the csv-file-name for the training 
  [-j --testCSV]          Specifies the csv-file-name for the testing 
  [-a --hdrTrain]         Specifies the header of training 
  [-g --hdrTest]          Specifies the header of testing 
  [-x --extraParam]       Specifies extra parameters
  [-w --weightFlag]       Specifies whether classifiers should use weights to balance inbalanced classes (default: False)

Examples:
  %(EXENAME)s --trainArff=$NMFTV_ResPATH/CV\(1_10\)-train-exp702-Features.arff --testArff=$NMFTV_ResPATH/CV\(1_10\)-test-exp702-Features.arff --bestClassifier="Bagging SMO" --bestParam="(1, 100.0, 10.0)" --removeLabel=1 --trainCSV=$HOME/train_alaki.csv --testCSV=$HOME/test_alaki.csv --hdrTrain="CV(1_10)-Class Label" --hdrTest="Class Label"  --extraParam="(0, 11.0, 2.0)"   --weightFlag
   I will describe it later...
""" % {'EXENAME': EXENAME}
    basis.print_contact()

# ============================================================================
# auxiliary functions
# ============================================================================

# ----------------------------------------------------------------------------
def getEvalwAUC(m):
    return m.weightedAreaUnderROC()

# ----------------------------------------------------------------------------
def getEvalpctCrr(m):
    return m.pctCorrect()

# ----------------------------------------------------------------------------
def getEvalpctInCrr(m):
    return m.pctIncorrect() 

# ----------------------------------------------------------------------------
def getEvalwPct(m):
    return m.weightedPrecision() 

# ----------------------------------------------------------------------------
def getEvalwTP(m):
    return m.weightedTruePositiveRate() 

# ----------------------------------------------------------------------------
def getEvalwFP(m):
    return m.weightedFalsePositiveRate() 

# ----------------------------------------------------------------------------
def getEvalCrr(m):
    return m.correct()

# ----------------------------------------------------------------------------
def getEvalIncrr(m):
    return m.incorrect() 

# ----------------------------------------------------------------------------
# split parameter string such as (10, 1) into the list [10, 1]
def splitparams(params):
    params = params.replace('(','')
    params = params.replace(')','')
    return [x.strip() for x in params.split(',')]

# ----------------------------------------------------------------------------
# convert string to boolean
def str2bool(st):
    map = {"false":False, "true":True}
    b = map[String(JyString.lower(st)).trim()]
    return b

# ----------------------------------------------------------------------------
# make a summary (dictionary) of evaluation model
def makeTrainEvalSummary(evalModel):
    SUMMARY_IDS = {
        '__wAuc__':     getEvalwAUC,
        '__pctCrr__':   getEvalpctCrr,
        '__pctInCrr__': getEvalpctInCrr,
        '__wPct__':     getEvalwPct,
        '__wTP__':      getEvalwTP,
        '__wFP__':      getEvalwFP,
        '__Crr__':      getEvalCrr,
        '__Incrr__':    getEvalIncrr
    }
    summary = {}
    for k in SUMMARY_IDS.keys():
        summary[k] = SUMMARY_IDS[k](evalModel)
    return summary

# ----------------------------------------------------------------------------
# this function calls pyxel.py to write csv file
def WriteCsvFile(CsvFilename,ID,Header,Value):
    outPath = os.path.dirname(CsvFilename)
    csvCommand = "pyxel.py "
    cmdLine = csvCommand + " -d %s  %s  %s  %s -o %s" %\
         ( '"' + CsvFilename + '"' ,\
           '"' + ID + '"' ,\
           '"' + Header + '"' , \
           '"'  + Value + '"' ,
           '"'  + outPath + '"')
    print "Saving results for subject %s to file %s" % (ID, CsvFilename)
    # TODO import pyxel module instead and its functions to perform this task
    basis.execute(cmdLine)

# ----------------------------------------------------------------------------
# this function stores accepts result buffer in addition to hear 
# and IDs and write it into the csv-file
def StoreInCSVResult(CsvFilename,header,resBuffer,ids,summary):
    resBuffer = str(resBuffer)
    lines = resBuffer.split('\n')
    cnt = 0
    for cntLine in range(1,len(lines)):
        l  = lines[cntLine-1]
        elements = l.split()
        ID  = ids.instance(cnt).toString()
        # fill the actual-label column
        print "l = ", l
        print "ID = ", ID
        Value = l.split()[1].split(':')[1]
        hdr = header + "-actual"
        WriteCsvFile(CsvFilename,ID,hdr,Value)
        # fill the prediction-label column
        Value = l.split()[2].split(':')[1]
        hdr = header + "-prediction"
        WriteCsvFile(CsvFilename,ID,hdr,Value)
        # fill incorrect-label flag column
        if len(elements)==5:    # it has extra element ('+') denoting incorrect prediction  
            Value = 'Y'
        elif len(elements)==4:  # correct prediction
            Value = 'N'
        else:  # what?!
            print "there is something wrong with the result buffer !!!"
            return 1
        hdr = header + "-incorrect"
        WriteCsvFile(CsvFilename,ID,hdr,Value)
        cnt = cnt + 1
    if not(summary==''):   # it means that training results are provided; thus summary should be written in the csv
        hdr = header + "-prediction"
        for k in summary.keys():
            WriteCsvFile(CsvFilename,k,hdr,str(summary[k]))

# ============================================================================
# preprocessing
# ============================================================================

# ----------------------------------------------------------------------------
def PreprocessData(Data,option):
    IDs = []
    if (option['idFlag']):    # means that the last attribute is id
        attributeremove = AttributeRemove()
        attributeremove.setInvertSelection(Boolean(True))  # remove every attribute but the last one which is ID
        attributeremove.setAttributeIndices(String(str(Data.numAttributes())))
        attributeremove.setInputFormat(Data)
        IDs = Filter.useFilter(Data, attributeremove)
        attributeremove = AttributeRemove()
        attributeremove.setInvertSelection(Boolean(False))  # remove IDs from dataset
        attributeremove.setAttributeIndices(String(str(Data.numAttributes())))
        attributeremove.setInputFormat(Data)
        Data = Filter.useFilter(Data, attributeremove)
    # set the class Index - the index of the dependent variable
    Data.setClassIndex(Data.numAttributes() - 1)
    # remove of the classes
    if (option['rmClassFlag']):    # means that instances with specified class label must be removed
        ClassLabel = option['rmClass']
        removewithvalues = RemoveWithValues()
        removewithvalues.setAttributeIndex(String('last'))
        removewithvalues.setNominalIndices(String(str(ClassLabel)))
        removewithvalues.setInputFormat(Data)
        newData = Filter.useFilter(Data, removewithvalues)
    else:
        newData = Data
    if (option['weightFlag']):    # it means that instances should be weighted according to number of samples
        # if there is only two classes, do it as before
        if (Data.numClasses()==2):
            # weight instances with reciprocal weight with number of samples
            numInstancesC1 = 0
            numInstancesC2 = 0
            # get numerical value of the class attribute for the first class because we don't know it
            classLabel = newData.instance(1).classAttribute()
            c1 = newData.instance(1).value(classLabel)
            # find number of instances per class
            for   cnt  in   range(0,newData.numInstances()):
                if (newData.instance(cnt).value(classLabel) == c1):
                    numInstancesC1 = numInstancesC1 + 1
                else:
                    numInstancesC2 = numInstancesC2 + 1
            # calculate weights
            weightC1 = numInstancesC2 /(numInstancesC2 + numInstancesC1 + 0.0)
            weightC2 = numInstancesC1 /(numInstancesC2 + numInstancesC1 + 0.0)
            # assign weight to instances of classes
            for cnt in range(0,newData.numInstances()):
                if (newData.instance(cnt).value(classLabel) == c1):
                    newData.instance(cnt).setWeight(weightC1)
                else:
                    newData.instance(cnt).setWeight(weightC2)
        # if number of class are more than two then .... 
        elif (Data.numClasses()>2):
            numClasses = Data.numClasses()
            stats = Data.attributeStats(Data.classIndex())
            AttributeStats = stats.nominalCounts
            classLabels = Data.instance(1).classAttribute()
            # assign weight to instances of classes
            cnt = 0
            sumWeigths = 0.0
            numInstancesPerClass = {}
            weightPerClass = {}
            mapClassLabels = {}
            for e in classLabels.enumerateValues():
                numInst = AttributeStats[cnt] + 0.0
                w = 1.0 / numInst
                mapClassLabels.update({e:cnt})
                weightPerClass.update({cnt:w})
                numInstancesPerClass.update({cnt:numInst})
                sumWeigths = sumWeigths + w
                cnt = cnt + 1 

            # normalize weights
            for k in weightPerClass.keys():
                weightPerClass[k] = weightPerClass[k]/sumWeigths

            for cnt in range(0,newData.numInstances()):
                w = weightPerClass[ newData.instance(cnt).value(classLabels) ]
                newData.instance(cnt).setWeight(w)
    return newData, IDs

# ============================================================================
# implementation of classifiers
# ============================================================================

# ----------------------------------------------------------------------------
# logistic classifier
def logistic(trainData,testData,params,exparams):
    ridge = float(params[0])
    maxIt = int(float(params[1]))
    print "Ridge=%s, maxIt=%s" %(str(ridge),str(maxIt))
    logistic = Logistic()
    logistic.setMaxIts(maxIt)
    logistic.setRidge(ridge)
    logistic.buildClassifier(trainData)  # only a trained classifier can be evaluated
    # evaluate it on the training
    evaluation = Evaluation(trainData)
    (trainOutput, trainBuffer) = util.get_buffer_for_predictions(trainData)
    attRange = Range()  # no additional attributes output
    outputDistribution = Boolean(False)  # we don't want distribution
    evaluation.evaluateModel(logistic, trainData, [trainOutput, attRange, outputDistribution])
    print "--> Evaluation:\n"
    print evaluation.toSummaryString()
    trainSummary = makeTrainEvalSummary(evaluation)
    # evaluate it on testing
    evaluation = Evaluation(testData)
    (testOutput, testBuffer) = util.get_buffer_for_predictions(testData)
    attRange = Range()  # no additional attributes output
    outputDistribution = Boolean(False)  # we don't want distribution
    evaluation.evaluateModel(logistic, testData, [testOutput, attRange, outputDistribution])
    return trainBuffer, testBuffer, trainSummary

# ----------------------------------------------------------------------------
# bagging on logistic classifier
def bagging_logistic(trainData,testData,params,exparams):
    IsOptBagOnOptLog = str2bool(params[0])
    logistic = Logistic()
    bagging = Bagging()
    if IsOptBagOnOptLog:    # optimal bagging is based on optimal logistic
        ridge = float(exparams[0])
        maxIt = int(float(exparams[1]))
        logistic.setMaxIts(maxIt)
        bagSizePercent = int(float(params[1]))
        bagging.setBagSizePercent(bagSizePercent)
    else:   # ridge parameter is also optimized in the process
        ridge = float(params[1])
    numIterations = int(float(params[2]))
    bagging.setNumIterations(numIterations)
    logistic.setRidge(ridge)
    bagging.setClassifier(logistic)
    bagging.buildClassifier(trainData)  # only a trained classifier can be evaluated
    # evaluate it on the training
    evaluation = Evaluation(trainData)
    (trainOutput, trainBuffer) = util.get_buffer_for_predictions(trainData)
    attRange = Range()  # no additional attributes output
    outputDistribution = Boolean(False)  # we don't want distribution
    evaluation.evaluateModel(bagging, trainData, [trainOutput, attRange, outputDistribution])
    print "--> Evaluation:\n"
    print evaluation.toSummaryString()
    trainSummary = makeTrainEvalSummary(evaluation)
    # evaluate it on testing
    evaluation = Evaluation(testData)
    (testOutput, testBuffer) = util.get_buffer_for_predictions(testData)
    attRange = Range()  # no additional attributes output
    outputDistribution = Boolean(False)  # we don't want distribution
    evaluation.evaluateModel(bagging, testData, [testOutput, attRange, outputDistribution])
    return trainBuffer, testBuffer, trainSummary

# ----------------------------------------------------------------------------
# SMO classifier
def smo(trainData,testData,params,exparams):
    kerType = str2bool(params[0]) 
    cValue = float(params[1])
    kerParam = float(params[2])
    if kerType:     # RBF kernel
        kernel = RBFKernel()
        kernel.setGamma(kerParam)
    else:       # Polynomial kernel
        kernel = PolyKernel()
        kernel.setExponent(kerParam)
    smo = SMO()
    smo.setKernel(kernel)
    smo.setC(cValue)
    smo.buildClassifier(trainData)  # only a trained classifier can be evaluated
    # evaluate it on the training
    evaluation = Evaluation(trainData)
    (trainOutput, trainBuffer) = util.get_buffer_for_predictions(trainData)
    attRange = Range()  # no additional attributes output
    outputDistribution = Boolean(False)  # we don't want distribution
    evaluation.evaluateModel(smo, trainData, [trainOutput, attRange, outputDistribution])
    print "--> Evaluation:\n"
    print evaluation.toSummaryString()
    trainSummary = makeTrainEvalSummary(evaluation)
    # evaluate it on testing
    evaluation = Evaluation(testData)
    (testOutput, testBuffer) = util.get_buffer_for_predictions(testData)
    attRange = Range()  # no additional attributes output
    outputDistribution = Boolean(False)  # we don't want distribution
    evaluation.evaluateModel(smo, testData, [testOutput, attRange, outputDistribution])
    return trainBuffer, testBuffer, trainSummary

# ----------------------------------------------------------------------------
# random-forest classifier
def random_forest(trainData,testData,params,exparams):
    numTrees = int(float(params[0]))
    numFeatures = int(float(params[1]))
    randomforest = RandomForest()
    randomforest.setNumTrees(numTrees)
    randomforest.setNumFeatures(numFeatures)
    randomforest.buildClassifier(trainData)  # only a trained classifier can be evaluated
    # evaluate it on the training
    evaluation = Evaluation(trainData)
    (trainOutput, trainBuffer) = util.get_buffer_for_predictions(trainData)
    attRange = Range()  # no additional attributes output
    outputDistribution = Boolean(False)  # we don't want distribution
    evaluation.evaluateModel(randomforest, trainData, [trainOutput, attRange, outputDistribution])
    print "--> Evaluation:\n"
    print evaluation.toSummaryString()
    trainSummary = makeTrainEvalSummary(evaluation)
    # evaluate it on testing
    evaluation = Evaluation(testData)
    (testOutput, testBuffer) = util.get_buffer_for_predictions(testData)
    attRange = Range()  # no additional attributes output
    outputDistribution = Boolean(False)  # we don't want distribution
    evaluation.evaluateModel(randomforest, testData, [testOutput, attRange, outputDistribution])
    return trainBuffer, testBuffer, trainSummary

# ----------------------------------------------------------------------------
# bayesian classifier
def bayesian(trainData,testData,params,exparams):
    IsOptMultinomialBayes   = str2bool(params[0]) 
    IsOptNaiveKernelDensity = str2bool(params[1]) 
    if IsOptMultinomialBayes:    # optimal bayesian classifier is multinomial
        bayes = NaiveBayesMultinomial()
    else:
        bayes = NaiveBayes()
        if IsOptNaiveKernelDensity:   # use kernel density estimation
            bayes.setUseKernelEstimator(Boolean(True))   
    bayes.buildClassifier(trainData)  # only a trained classifier can be evaluated
    # evaluate it on the training
    evaluation = Evaluation(trainData)
    (trainOutput, trainBuffer) = util.get_buffer_for_predictions(trainData)
    attRange = Range()  # no additional attributes output
    outputDistribution = Boolean(False)  # we don't want distribution
    evaluation.evaluateModel(bayes, trainData, [trainOutput, attRange, outputDistribution])
    print "--> Evaluation:\n"
    print evaluation.toSummaryString()
    trainSummary = makeTrainEvalSummary(evaluation)
    # evaluate it on testing
    evaluation = Evaluation(testData)
    (testOutput, testBuffer) = util.get_buffer_for_predictions(testData)
    attRange = Range()  # no additional attributes output
    outputDistribution = Boolean(False)  # we don't want distribution
    evaluation.evaluateModel(bayes, testData, [testOutput, attRange, outputDistribution])   
    return trainBuffer, testBuffer, trainSummary

# ----------------------------------------------------------------------------
# bagging over SMO
def baggin_smo(trainData,testData,params,exparams):
    IsOptBagOnOptSMO =  str2bool(params[0]) 
    if IsOptBagOnOptSMO:    # optimal bagging is based on optimal SMO thus I should use extra params
        kerType =  str2bool(params[0]) 
        cValue = float(exparams[1])
        kerParam = float(exparams[2])
        if kerType:     # RBF kernel
            kernel = RBFKernel()
            kernel.setGamma(kerParam)
        else:       # Polynomial kernel
            kernel = PolyKernel()
            kernel.setExponent(kerParam)
        bagSizePercent = int(float(params[1]))
        numIterations = int(float(params[2]))
        smo = SMO()
        bagging = Bagging()
        smo.setKernel(kernel)
        smo.setC(cValue)
        bagging.setBagSizePercent(bagSizePercent)
        bagging.setNumIterations(numIterations)
        bagging.setClassifier(smo)
    else:   # optimal bagging is based on linear SMO
        cValue = float(params[1])
        numIterations = int(float(params[2]))
        smo = SMO()
        bagging = Bagging()
        kernel = PolyKernel()
        smo.setKernel(kernel)
        smo.setC(cValue)
        bagging.setNumIterations(numIterations)
        bagging.setClassifier(smo)
    bagging.buildClassifier(trainData)  # only a trained classifier can be evaluated
    # evaluate it on the training
    evaluation = Evaluation(trainData)
    (trainOutput, trainBuffer) = util.get_buffer_for_predictions(trainData)
    attRange = Range()  # no additional attributes output
    outputDistribution = Boolean(False)  # we don't want distribution
    evaluation.evaluateModel(bagging, trainData, [trainOutput, attRange, outputDistribution])
    print "--> Evaluation:\n"
    print evaluation.toSummaryString()
    trainSummary = makeTrainEvalSummary(evaluation)
    # evaluate it on testing
    evaluation = Evaluation(testData)
    (testOutput, testBuffer) = util.get_buffer_for_predictions(testData)
    attRange = Range()  # no additional attributes output
    outputDistribution = Boolean(False)  # we don't want distribution
    evaluation.evaluateModel(bagging, testData, [testOutput, attRange, outputDistribution])
    return trainBuffer, testBuffer, trainSummary

# ----------------------------------------------------------------------------
# simple-logistic classifier
def simple_logistic(trainData,testData,params,exparams):
    heuristicStop = int(float(params[0]))
    numBoostingIterations = int(float(params[1]))
    simplelogistic = SimpleLogistic()
    simplelogistic.setHeuristicStop(heuristicStop)
    simplelogistic.setNumBoostingIterations(numBoostingIterations)
    if (trainData.numInstances()<5):   # special case for small sample size
        simplelogistic.setUseCrossValidation(False) 
    simplelogistic.buildClassifier(trainData)  # only a trained classifier can be evaluated
    # evaluate it on the training
    evaluation = Evaluation(trainData)
    (trainOutput, trainBuffer) = util.get_buffer_for_predictions(trainData)
    attRange = Range()  # no additional attributes output
    outputDistribution = Boolean(False)  # we don't want distribution
    evaluation.evaluateModel(simplelogistic, trainData, [trainOutput, attRange, outputDistribution])
    print "--> Evaluation:\n"
    print evaluation.toSummaryString()
    trainSummary = makeTrainEvalSummary(evaluation)
    # evaluate it on testing
    evaluation = Evaluation(testData)
    (testOutput, testBuffer) = util.get_buffer_for_predictions(testData)
    attRange = Range()  # no additional attributes output
    outputDistribution = Boolean(False)  # we don't want distribution
    evaluation.evaluateModel(simplelogistic, testData, [testOutput, attRange, outputDistribution])
    return trainBuffer, testBuffer, trainSummary

# ----------------------------------------------------------------------------
# AdaboostM1 on simple-logistic classifier
def adaboostM1_simple_logistic(trainData,testData,params,exparams):
    IsOptBoostOnOptSimpLog = str2bool(params[0])  
    simplelogistic = SimpleLogistic()
    adaboostm = AdaBoostM1()
    if IsOptBoostOnOptSimpLog:  # optimal adaboost is based on optimal simple logisatic 
        heuristicStop = int(float(exparams[0]))
        numBoostingIterations = int(float(exparams[1]))
        weightThreshold = int(float(params[1]))
        numIterations = int(float(params[2]))
        simplelogistic.setHeuristicStop(heuristicStop)
        simplelogistic.setNumBoostingIterations(numBoostingIterations)
        adaboostm.setWeightThreshold(weightThreshold)
        adaboostm.setNumIterations(numIterations)       
    else:
        numBoostingIterations = int(float(params[1]))
        numIterations = int(float(params[2]))
        simplelogistic.setNumBoostingIterations(numBoostingIterations)
        adaboostm.setNumIterations(numIterations)       
    adaboostm.setClassifier(simplelogistic)
    adaboostm.buildClassifier(trainData)  # only a trained classifier can be evaluated
    # evaluate it on the training
    evaluation = Evaluation(trainData)
    (trainOutput, trainBuffer) = util.get_buffer_for_predictions(trainData)
    attRange = Range()  # no additional attributes output
    outputDistribution = Boolean(False)  # we don't want distribution
    evaluation.evaluateModel(adaboostm, trainData, [trainOutput, attRange, outputDistribution])
    print "--> Evaluation:\n"
    print evaluation.toSummaryString()
    trainSummary = makeTrainEvalSummary(evaluation)
    # evaluate it on testing
    evaluation = Evaluation(testData)
    (testOutput, testBuffer) = util.get_buffer_for_predictions(testData)
    attRange = Range()  # no additional attributes output
    outputDistribution = Boolean(False)  # we don't want distribution
    evaluation.evaluateModel(adaboostm, testData, [testOutput, attRange, outputDistribution])
    return trainBuffer, testBuffer, trainSummary

# ============================================================================
# constants
# ============================================================================

# Attention: Must be following the definition of the classifier functions!

CLASSIFIER = {
    'BaggingLogistic':       bagging_logistic,
    'Logistic':              logistic,
    'SimpleLogistic':        simple_logistic,
    'SMO':                   smo,
    'RandomForest':          random_forest,
    'Bayesian':              bayesian,
    'BaggingSMO':            baggin_smo,
    'BoostedSimpleLogistic': adaboostM1_simple_logistic
}

# ============================================================================
# main
# ============================================================================

# ----------------------------------------------------------------------------
def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hr:s:b:p:l:i:j:a:g:x:w",\
            ["help", "trainArff=", "testArff=","bestClassifier=","bestParam=","removeLabel="\
            ,"trainCSV=","testCSV=","hdrTrain=","hdrTest=","extraParam=","weightFlag"])
    except getopt.GetoptError, err:
        sys.stderr.write("%s\n" % err)
        return 1
    extraParam = ""
    weightFlag = False
    rmClassFlag = False
    removeLabel = 0
    for o, a in opts:
        if o in ("-h", "--help"):
            usage()
            return 0
        elif o in ("-r", "--trainArff"):
            trainArff = a
        elif o in ("-s", "--testArff"):
            testArff = a 
        elif o in ("-b","--bestClassifier"):
            bestClassifier = a
        elif o in ("-p","--bestParam"):
            bestParam = a
        elif o in ("-l","--removeLabel"):
            rmClassFlag = True
            removeLabel = int(float(a))
        elif o in ("-i","--trainCSV"):
            trainCSV = a
        elif o in ("-j","--testCSV"):
            testCSV = a
        elif o in ("-a","--hdrTrain"):
            hdrTrain = a
        elif o in ("-g","--hdrTest"):
            hdrTest = a
        elif o in ("-x","--extraParam"):
            extraParam = a
        elif o in ("-w","--weightFlag"):
            weightFlag = True
        else:
            assert False, "unhandled option"
    if len(opts) < 9:
        usage()
        return 1
    # reading training files
    f = FileReader(trainArff)
    traindata = Instances(f)
    # reading testing files
    f = FileReader(testArff)
    testdata = Instances(f)
    # remove and edit train/test data
    options = {'idFlag':True, 'weightFlag': weightFlag, 'rmClassFlag': rmClassFlag, 'rmClass': removeLabel}
    newTrainData, trainIDs = PreprocessData(traindata, options)
    newTestData,  testIDs  = PreprocessData(testdata,  options)
    # run classifier
    handler = CLASSIFIER[bestClassifier]
    trainResult, testResult, trainSummary = handler(newTrainData, newTestData, splitparams(bestParam), splitparams(extraParam))
    # store results in spreadsheet
    StoreInCSVResult(trainCSV, hdrTrain, trainResult, trainIDs, trainSummary)
    StoreInCSVResult(testCSV,  hdrTest,  testResult,  testIDs,  '')
    return 0

if __name__ == '__main__': sys.exit(main())
