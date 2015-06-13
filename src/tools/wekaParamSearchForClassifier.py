#! /usr/bin/env jython

##############################################################################
# @file  wekaParamSearchForClassifier.py
# @brief Performs best parameter search for the specified classifiers.
#
# Copyright (c) 2011, 2012 University of Pennsylvania. All rights reserved.
# See http://www.rad.upenn.edu/sbia/software/license.html or COPYING file.
#
# Contact: SBIA Group <sbia-software at uphs.upenn.edu>
##############################################################################

# BASIS utilities
from gondola import basis

# general imports
import sys
import os
import getopt

# Java general imports
import java.io.FileReader as FileReader
import java.lang.StringBuffer as StringBuffer
import java.lang.Boolean as Boolean
import java.lang.String as String

# Weka general imports
import weka.filters.Filter as Filter
import weka.core.Instances as Instances
import weka.core.Utils as Utils
import weka.core.AttributeStats as AttributeStats
import weka.classifiers.Evaluation as Evaluation
import weka.core.Range as Range
import weka.filters.unsupervised.instance.RemoveWithValues as RemoveWithValues
import weka.filters.unsupervised.attribute.Remove as AttributeRemove

# load modules to search for the best parameters
from gondola.paramsearch.smo import *
from gondola.paramsearch.logistic import *
from gondola.paramsearch.simplelogistic import *
from gondola.paramsearch.smobagging import *
from gondola.paramsearch.logisticbagging import *
from gondola.paramsearch.bayesian import *
from gondola.paramsearch.randomforest import *
from gondola.paramsearch.simplelogisticadaboost import *


# ============================================================================
# help
# ============================================================================

# ----------------------------------------------------------------------------
def usage():
  """usage information"""
  print """
%(EXENAME)s--
  This code looks for the best parameters for classification of ADNI-data.
  It produces the best parameters of the following classifiers:
    1. SMO 
    2. Bagging of SMO
    3. Logistic
    4. Bagging of Logistic
    5. Simple Logistic
    6. AdaboostM1 over Simple-Logistic
    7. Random Forest
    8. Naive Bayes



Usage: %(EXENAME)s [options]

Options:
  [-a --arffFile]             Specify the input arff file assumed to be csv (MANDATORY to use)
  [-c --csvFile]          Specify the input-output data file assumed to be csv (MANDATORY to use)
  [-i --idFlag]               If it is used, it means that the last feature is IDs
  [-w --weightFlag]           If it is used, instances would be weighted according to number of samples in the corresponding class
  [-r --rmClass]              If it is used, all instances specified by this options will be removed from data set
  [-l --listOfClassifiers]     Specify the list of classifiers (Optional)

  

Examples:
  %(EXENAME)s -a CV(3_10)-train-exp702-Features.arff  -c  CV(3_10)-train-exp702-BestParams.csv
    It works on the the arff file and save the results in the csv file

Examples:
  %(EXENAME)s -a CV(3_10)-train-exp702-Features.arff  -c  CV(3_10)-train-exp702-BestParams.csv -i -w -r 2
    It works on the the arff file and save the results in the csv file


""" % {'EXENAME': basis.exename()}


# ============================================================================
# preprocessing
# ============================================================================

# ----------------------------------------------------------------------------
def PreprocessData(Data,option):
    if (option['idFlag']):    # means that the last attribute is id
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
            for cnt in range(0,newData.numInstances()):
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
    return newData

# ============================================================================
# auxiliary functions
# ============================================================================

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
    print "Saving results for %s classifier to file %s" % (ID, CsvFilename)
    basis.execute(cmdLine)

# ----------------------------------------------------------------------------
# this function takes five arguments and is aware what each class label means
def StoreInCSVTable(CsvFilename,classifierName,accValue,Description,outParama):
    WriteCsvFile(CsvFilename, classifierName, 'Accuracy',    str(accValue))
    WriteCsvFile(CsvFilename, classifierName, 'Description', Description)
    WriteCsvFile(CsvFilename, classifierName, 'Parameters',  str(outParama))

# ============================================================================
# main
# ============================================================================

# ----------------------------------------------------------------------------
def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "ha:c:iwr:l:",\
            ["help", "arffFile=", "csvFile=","idFlag","weightFlag","rmClass=","listOfClassifiers="])
    except getopt.GetoptError, err:
        usage()
        sys.stderr.write(err + '\n')
        return 1
    idFlag = False
    weightFlag = False
    rmClassFlag = False
    rmClass = 0
    listOfClassifier = ['Logistic','Bagging Logistic','SMO','Bagging SMO','Simple Logistic','Bayesian','Random Forest']
    for o, a in opts:
        if o in ("-h", "--help"):
            usage()
            return 0
        elif o in ("-a", "--arffFile"):
            arffFile = a
        elif o in ("-c", "--csvFile"):
            CsvFilename = a
        elif o in ("-i","--idFlag"):
            idFlag = True
        elif o in ("-w","--weightFlag"):
            weightFlag = True
        elif o in ("-r","--rmClass"):
            rmClassFlag = True
            rmClass = int(float(a))
        elif o in ("-l","--listOfClassifiers"):
            listOfClassifier = [x.strip() for x in a.split(',')]
        else:
            assert False, "unhandled option"
    if len(opts) < 3:
        usage()
        return 1
    # load data file
    print "Loading data..."
    print "-------------- Input arffFile: %s" % arffFile
    print "-------------- Output CsvFile: %s" % CsvFilename
    # make sure that csv file does not exist there and you are creating it for the first time
    if os.path.exists(CsvFilename): os.remove(CsvFilename)
    file = FileReader(arffFile)
    data = Instances(file)
    # remove one of the classes andweight instances properly to compensate imbalanced number of intances
    options = {'idFlag':idFlag, 'weightFlag': weightFlag, 'rmClassFlag': rmClassFlag, 'rmClass': rmClass}
    newData = PreprocessData(data,options)
    # Iterate over schmes to find optimal sets of parameters for each classifier
    if ('SMO' in listOfClassifier):
        # ----- SMO
        OptSMOIsRBF, OptSMO, OptSMOp1, OptSMOp2, OptSMOAcc, Description = SMO_ParamFinder(newData)
        outParam = (OptSMOIsRBF, OptSMOp1, OptSMOp2)
        StoreInCSVTable(CsvFilename,'SMO',OptSMOAcc,Description,outParam)
    if ('Bagging SMO' in listOfClassifier):
        # ----- Bagging SMO
        IsOptBagOnOptSMO, OptBagSMO,  OptBagSMOp1, OptBagSMOp2, OptBagSMOAcc, Description = \
        BaggingSMO_ParamFinder(newData, OptSMOIsRBF, OptSMOp1, OptSMOp2)
        outParam = (IsOptBagOnOptSMO, OptBagSMOp1, OptBagSMOp2)
        StoreInCSVTable(CsvFilename,'Bagging SMO',OptBagSMOAcc,Description,outParam)
    if ('Logistic' in listOfClassifier):
        # ----- Logistic
        OptLog, OptLogp1, OptLogp2, OptLogAcc, Description = Logistic_ParamFinder(newData)
        outParam = (OptLogp1, OptLogp2)
        StoreInCSVTable(CsvFilename,'Logistic',OptLogAcc,Description,outParam)
    if ('Bagging Logistic' in listOfClassifier):
        # ----- Bagging Logistic
        IsOptBagOnOptLog, OptBagLog,  OptBagLogp1, OptBagLogp2, OptBagLogAcc, Description  = \
               BaggingLogistic_ParamFinder(newData, OptLogp1, OptLogp2)
        outParam = (IsOptBagOnOptLog, OptBagLogp1, OptBagLogp2)
        StoreInCSVTable(CsvFilename,'Bagging Logistic',OptBagLogAcc,Description,outParam)
    if ('Simple Logistic' in listOfClassifier):
        # ----- Simple Logistic
        OptSimpLog, OptSimpLogp1, OptSimpLogp2, OptSimpLogAcc, Description = \
               SimpleLogistic_ParamFinder(newData)
        outParam = (OptSimpLogp1, OptSimpLogp2)
        StoreInCSVTable(CsvFilename,'Simple Logistic',OptSimpLogAcc,Description,outParam)
    if ('Bayesian' in listOfClassifier):
        # ----- Find the best configuration for Bayesian classifier 
        IsOptMultinomialBayes, IsOptNaiveKernelDensity, OptBayesAcc, Description = Bayes_ParamFinder(newData)
        outParam = (IsOptMultinomialBayes, IsOptNaiveKernelDensity)
        StoreInCSVTable(CsvFilename,'Bayesian',OptBayesAcc,Description,outParam)
    if ('Random Forest' in listOfClassifier):
        # ----- Find the best parameter for Random-Forest classifier
        OptRndFrst, OptRndFrstp1, OptRndFrstp2, OptRndFrstAcc, Description = RandomForest_ParamFinder(newData)
        outParam = ( OptRndFrstp1, OptRndFrstp2)
        StoreInCSVTable(CsvFilename,'Random Forest',OptRndFrstAcc,Description,outParam)
    return 0

# ----------------------------------------------------------------------------
if __name__ == '__main__': sys.exit(main())
