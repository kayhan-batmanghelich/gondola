#! /usr/bin/env jython

##############################################################################
# @file  wekaRankBasis.py
# @brief Rank basis vectors according to their relevance.
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
import string as JyString


# General Java imports
import java.io.FileReader as FileReader
import java.lang.StringBuffer as StringBuffer
import java.lang.Boolean as Boolean
import java.lang.String as String
from jarray import zeros, array
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



# import Wrap WEKAs' Algorithms in bridge
import   weka.attributeSelection.AttributeSelection  as  AttributeSelection

# imports for InfoGainAttEval
import weka.attributeSelection.InfoGainAttributeEval as InfoGainAttributeEval

# imports for GainRatioAttEval
import weka.attributeSelection.GainRatioAttributeEval  as  GainRatioAttributeEval

# import for OneRAttributeEval
import weka.attributeSelection.OneRAttributeEval  as  OneRAttributeEval

# import for ReliefFAttributeEval
import weka.attributeSelection.ReliefFAttributeEval  as ReliefFAttributeEval

# import for SignificanceAttributeEval
import weka.attributeSelection.SignificanceAttributeEval as SignificanceAttributeEval

# import for SymmetricalUncertAttributeEval
import weka.attributeSelection.SymmetricalUncertAttributeEval  as SymmetricalUncertAttributeEval

# import different search method
import weka.attributeSelection.Ranker as Ranker



EXEC_NAME = sys.argv[0]

def usage():
  """usage information"""
  print """
%(EXEC)s--
  This script can be used to rank basis vectors. It uses several ranking methods and find the consensus among their ranking results. Here is the list of ranking method it applies:
        1. InfoGainAttributeRanking
        2. GainRatioAttributeRanking
        3. OneRAttributeRanking
        4. ReliefFAttributeRanking
        5. SymmetricalUncertAttributeRanking
        6. SignificanceAttributeRanking     (Note:  make sure that you have this addon in you CLASSPATH environment variable)

	
Usage: %(EXEC)s [options]

Options:
  [-t --trainArff]  		Specifies the arff-file for training (MANDATORY options)
  [-l --removeLabel]            Specify labels of the data if you want to remove it from data 
  [-m --basisLearning]          Specify which method you used for basis learning:  (MANDATORY options)
                                    "1" for "MultiViewXY": Basis vectors are shared across modalities
                                    "2" for "MultiViewY" : Each modality has it own matrix; hence we have basis tensor
  [-c --numChannels]            Specify number of channels (modalities) (MANDATORY options)
  [-b --numBasis]               Specify number of basis. For "MultiViewY", number of columns in of the faces. (MANDATORY options)
  [-r --rankingMethods]         Specify ranking methods for ranking. The list should be comma separated. Here is the list of ranking method it applies  (Default: 1,2,3,4,5):
                                  1. InfoGainAttributeRanking
                                  2. GainRatioAttributeRanking
                                  3. OneRAttributeRanking
                                  4. ReliefFAttributeRanking
                                  5. SymmetricalUncertAttributeRanking
                                  6. SignificanceAttributeRanking     (Note:  make sure that you have this addon in you CLASSPATH environment variable)



Examples:
  
  %(EXEC)s  --trainArff=training.arff   --basisLearning=1  --numBasis=120  --rankingMethods=1,2,3   --numChannels=3
  
""" % {'EXEC':EXEC_NAME}

# convert string to boolean
def     str2bool(st):
        map = {"false":False, "true":True}
        b = map[String(JyString.lower(st)).trim()]
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




#  rank attributes using InfoGainAttribute ranking
def    InfoGainAttributeRanking(trainData):
       evaluate = InfoGainAttributeEval()
       search = Ranker()
       attsel = AttributeSelection()
       attsel.setSearch(search)
       attsel.setEvaluator(evaluate)
       attsel.SelectAttributes(trainData)
       rankedAtt = attsel.rankedAttributes()
       return rankedAtt



#  rank attributes using InfoGainAttribute ranking
def    GainRatioAttributeRanking(trainData):
       evaluate = GainRatioAttributeEval()
       search = Ranker()
       attsel = AttributeSelection()
       attsel.setSearch(search)
       attsel.setEvaluator(evaluate)
       attsel.SelectAttributes(trainData)
       rankedAtt = attsel.rankedAttributes()
       return rankedAtt


# rank attributes using OneRAttribute ranking
def    OneRAttributeRanking(trainData):
       evaluate = OneRAttributeEval()
       search = Ranker()
       attsel = AttributeSelection()
       attsel.setSearch(search)
       attsel.setEvaluator(evaluate)
       attsel.SelectAttributes(trainData)
       rankedAtt = attsel.rankedAttributes()
       return rankedAtt


# rank attributes using ReliefFAttribute ranking
def    ReliefFAttributeRanking(trainData):
       evaluate = ReliefFAttributeEval()
       search = Ranker()
       attsel = AttributeSelection()
       attsel.setSearch(search)
       attsel.setEvaluator(evaluate)
       attsel.SelectAttributes(trainData)
       rankedAtt = attsel.rankedAttributes()
       return rankedAtt


# rank attributes using SignificanceAttribute ranking 
def    SignificanceAttributeRanking(trainData):
       evaluate = SignificanceAttributeEval()
       search = Ranker()
       attsel = AttributeSelection()
       attsel.setSearch(search)
       attsel.setEvaluator(evaluate)
       attsel.SelectAttributes(trainData)
       rankedAtt = attsel.rankedAttributes()
       return rankedAtt


# rank attributes using SymmetricalUncertAttribute ranking
def    SymmetricalUncertAttributeRanking(trainData):
       evaluate = SymmetricalUncertAttributeEval()
       search = Ranker()
       attsel = AttributeSelection()
       attsel.setSearch(search)
       attsel.setEvaluator(evaluate)
       attsel.SelectAttributes(trainData)
       rankedAtt = attsel.rankedAttributes()
       return rankedAtt

      


# normalize scores between [0,1]
def    normalizeScores(rankingScores):
       newRankingScores = []
       for  rScore in rankingScores:
            minScore = rScore[-1][1]
            maxScore = rScore[0][1]
            newScore = zeros(len(rScore),'d') 
            for s in rScore:
                newScore[int(s[0])] = (s[1] - minScore)/(maxScore - minScore)
            newRankingScores.append(newScore) 
       return newRankingScores

#  This function finds consensus among different methods
def  findConsensusRanking(rankingScores):
     sumScores = zeros(len(rankingScores[0]),'d')
     for  i in range(0,len(sumScores)):
          for j in range(0,len(rankingScores)):
              sumScores[i] = sumScores[i] + rankingScores[j][i]
     # normalize the sumScore
     for  i in range(0,len(sumScores)):
          sumScores[i] = sumScores[i]/len(rankingScores) 
     inx = range(len(sumScores))
     inx.sort(lambda x,y: cmp(sumScores[x],sumScores[y]))
     return sumScores, inx

# interprete the ranking
def  interpretRanking(finalScore,sortIdx,basisLearningMethod,numBasis,numChannels):
     if  (basisLearningMethod==1):     # MultiViewXY
         print "You have chosen MultiViewXY. If you want to check the basis image, check the image specified by basis vector."
         print "Features are ranked according to their importance, from the most important to the least important "
         print "#Basis Vector  \t  #Modality  \t   Score :"
         print "==============================================="
         for i in range(len(finalScore)-1,-1,-1):
               idxBasis = (sortIdx[i] ) % numBasis + 1
               idxChannel = (sortIdx[i] ) / numBasis +  1
               print "Basis (%d) Applied on Modality (%d) with Score (%f)"%(idxBasis,idxChannel,finalScore[sortIdx[i]])               
     elif (basisLearningMethod==2):    # MultiViewY
         print "You have chosen MultiViewY. If you want to check the basis image, check the image specified by basis vector and channel index."
         print "Features are ranked according to their importance, from the most important to the least important "
         print "#Basis Vector  \t  #Modality  \t   Score :"
         print "==============================================="
         for i in range(len(finalScore)-1,-1,-1):
               idxBasis = (sortIdx[i] ) % numBasis + 1
               idxChannel = (sortIdx[i] ) / numBasis +  1
               print "Basis (%d) Applied on Modality (%d) with Score (%f)"%(idxBasis,idxChannel,finalScore[sortIdx[i]])               
     else:
        usage()
        assert False , "I don't know how to interprete this type of basis learning, check the documentation !!! "



handlerMapping = {'1':InfoGainAttributeRanking,\
                  '2':GainRatioAttributeRanking,\
                  '3':OneRAttributeRanking,\
                  '4':ReliefFAttributeRanking,\
                  '5':SymmetricalUncertAttributeRanking,\
                  '6':SignificanceAttributeRanking}

def main():
	try:
          opts, args = getopt.getopt(sys.argv[1:], "ht:l:m:c:b:r:",\
      		["help", "trainArff=","removeLabel=","basisLearning=","numChannels=","numBasis=","rankingMethods="])
  
  	except getopt.GetoptError, err:
    		usage()
    		print err
	
	weightFlag = False
        rmClassFlag = False
        removeLabel = 0
        rankingMethodList = [handlerMapping['1'],handlerMapping['2'],handlerMapping['3'],handlerMapping['4'],handlerMapping['5']]
        basisLearning = []
        numReqOpt = 0
  	for o, a in opts:
    		if o in ("-h", "--help"):
      			usage()
      			sys.exit(0)
    		elif o in ("-t", "--trainArff"):
      			trainArff = a
                        numReqOpt = numReqOpt + 1 
		elif o in ("-l","--removeLabel"):
			rmClassFlag = True
			removeLabel = int(float(a))
                elif o in ("-c","--numChannels"):
                        numChannels = int(float(a))
                        numReqOpt = numReqOpt + 1
                elif o in ("-b","--numBasis"):
                        numBasis = int(float(a))
                        numReqOpt = numReqOpt + 1
                elif o in ("-m","--basisLearning"):
                        basisLearningMethod = int(float(a))
                        numReqOpt = numReqOpt + 1 
                        if  not((basisLearningMethod==1) or (basisLearningMethod==2)):
                           usage()
                           assert False, "basisLearning should be either 1 or 2 !!!!"
                elif o in ("-r","--rankingMethods"):
                       rankingMethodList = []
                       list = a.split(',')
                       for l in list:
                           try:
                              rankingMethodList.append(handlerMapping[l])
                           except:
                              usage() 
                              assert False, "One of the ranking you asked for is not supported !!!!"
    		else:
      			assert False, "unhandled option"

  	if (numReqOpt < 4):
    		usage()
    		return 1


        # reading training files
	file = FileReader(trainArff)
	traindata = Instances(file)
	#traindata.setClassIndex(traindata.numAttributes() - 2)   # last attribute is ID and one before is class-label

	# remove and edit train/test data
	options = {'idFlag':True, 'weightFlag': False, 'rmClassFlag': rmClassFlag, 'rmClass': removeLabel}
        newTrainData, trainIDs = PreprocessData(traindata,options)

       
        rankingScores = [] 
        for handler in rankingMethodList:
           score = handler(newTrainData)
           rankingScores.append(score)
           #print score

        # normalize scores between [0,1]
        rankingScores = normalizeScores(rankingScores)
 
        # find the consensus among rankers
        finalScore, sortIdx = findConsensusRanking(rankingScores)
        #print sortIdx
        #for i in range(len(finalScore)-1,-1,-1):
        #    print finalScore[sortIdx[i]]

        # interprete the ranking
        interpretRanking(finalScore,sortIdx,basisLearningMethod,numBasis,numChannels)

if __name__ == '__main__': main()

