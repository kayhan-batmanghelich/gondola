#! /usr/bin/env jython

##############################################################################
# @file  mergeArffFiles.py
# @brief Utility script that can be used to merge ARFF files.
#
# Copyright (c) 2011, 2012 University of Pennsylvania. All rights reserved.
# See http://www.rad.upenn.edu/sbia/software/license.html or COPYING file.
#
# Contact: SBIA Group <sbia-software at uphs.upenn.edu>
##############################################################################

# General system imports
import sys
import os as os
import getopt
import os.path
import string as JyStri


# General Java imports
import java.io.FileReader as FileReader
import java.io.File as FileWriter
import java.lang.StringBuffer as StringBuffer
import java.lang.Boolean as Boolean
import java.lang.String as String
from   jarray import zeros, array
import java.util.Arrays as arrayUtil


# General Weka imports
import weka.filters.Filter as Filter
import weka.core.Instances as Instances
import weka.core.Utils as Utils
import weka.core.AttributeStats as AttributeStats
import weka.classifiers.Evaluation as Evaluation
import weka.core.Range as Range
import weka.filters.unsupervised.instance.RemoveWithValues as RemoveWithValues
import weka.filters.unsupervised.attribute.Remove as AttributeRemove
import weka.core.converters.ArffSaver as ArffSaver



EXEC_NAME = sys.argv[0]

def usage():
  """usage information"""
  print """
%(EXEC)s--
  This script merges the arff files and save it into one file:

   NOTICE: this script assumes that arff files have the same ordering of the subjects and it just concatenate features, BE CAREFUL !!!
        
	
Usage: %(EXEC)s [options]

Options:
  [-i --inputArff]  	       Specify the list of arff files to be merged. It should be comma separated
  [-o --outputArff]            Specify the output arff file



Examples:
  
  %(EXEC)s  --inputArff=training1.arff,training2.arff  --outputArff=training.arff
  
""" % {'EXEC':EXEC_NAME}


# convert string to boolean
def     str2bool(st):
        map = {"false":False, "true":True}
        b = map[JyString.lower(st)]
        return b


def	PreprocessData(Data,option):
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



def main():
	try:
          opts, args = getopt.getopt(sys.argv[1:], "hi:o:",\
      		["help", "inputArff=","outputArff="])
  
  	except getopt.GetoptError, err:
    		usage()
    		print err
	
	inputList  = [] 
        outputFn = ''
        numReqOpt = 0 
  	for o, a in opts:
    		if o in ("-h", "--help"):
      			usage()
      			sys.exit(0)
    		elif o in ("-i", "--inputArff"):
      			inputList = a.split(',')
                        numReqOpt = numReqOpt + 1 
		elif o in ("-o","--outputArff"):
			outputFn = a
			numReqOpt = numReqOpt + 1
    		else:
      			assert False, "unhandled option"

  	if (numReqOpt < 2):
    		usage()
    		return 1


        options = {'idFlag':True, 'weightFlag': False, 'rmClassFlag': False, 'rmClass': 0}
        # read the first dataset
        fn = inputList[0]
        fid = FileReader(fn)
	Data = Instances(fid)
        Data, IDs = PreprocessData(Data,options)
        # remove class label
        attributeremove = AttributeRemove()
        attributeremove.setInvertSelection(Boolean(False))  # remove class labels from dataset
        attributeremove.setAttributeIndices(String(str(Data.numAttributes())))
        attributeremove.setInputFormat(Data)
        newData = Filter.useFilter(Data, attributeremove)
        # loop over input arff file
        cnt = Data.numAttributes() 
        for fnCnt in range(1,len(inputList)):
             fn = inputList[fnCnt]
             fid = FileReader(fn)
	     Data = Instances(fid)
             Data, IDs = PreprocessData(Data,options)
             # remove class label
             attributeremove = AttributeRemove()
	     attributeremove.setInvertSelection(Boolean(True))  # remove every attribute but the last one which is class label
	     attributeremove.setAttributeIndices(String(str(Data.numAttributes())))
	     attributeremove.setInputFormat(Data)
	     labels = Filter.useFilter(Data, attributeremove)
             attributeremove = AttributeRemove()
             attributeremove.setInvertSelection(Boolean(False))  # remove class labels from dataset
             attributeremove.setAttributeIndices(String(str(Data.numAttributes())))
             attributeremove.setInputFormat(Data)
             Data = Filter.useFilter(Data, attributeremove)
             # rename features
             for  i in range(1,Data.numAttributes()+1):
                  Data.renameAttribute(i-1, String('W%d'%(cnt)))
                  cnt = cnt + 1 
             # merge data
             newData = Instances.mergeInstances(newData, Data)
 
        # put label data and IDs back
        newData = Instances.mergeInstances(newData, labels)
        newData = Instances.mergeInstances(newData, IDs)
    
        # save the data
        fid = FileWriter(outputFn) 
        arffsaver = ArffSaver()
        arffsaver.setFile(fid)
        arffsaver.setInstances(newData)
        arffsaver.writeBatch()
         


if __name__ == '__main__': main()

