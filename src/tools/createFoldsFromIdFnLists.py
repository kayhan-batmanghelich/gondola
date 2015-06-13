#! /usr/bin/env python

##############################################################################
# @file  createFoldsFromIdFnLists.py
# @brief Auxiliary script for creation of cross-validation folds.
#
# Copyright (c) 2011, 2012 University of Pennsylvania. All rights reserved.
# See http://www.rad.upenn.edu/sbia/software/license.html or COPYING file.
#
# Contact: SBIA Group <sbia-software at uphs.upenn.edu>
##############################################################################

import os
import sys
import getopt
import random
import numpy


from gondola import basis


def usage():
    """usage information"""
    print """
%(EXEC)s--
  This script reads a pair of filename and ID lists and creates n separate pairs of training/testing filename and ID lists.

Usage: %(EXEC)s [options]

Required Options:
  [-f --fileList]               Specify the list of filenames. The file should be in COMPARE format (required)
  [-i --idList]                 Specify the list of IDs. These should be in the same order as the filenames (required)
  [-n --numFolds]               Either specify number of folds, or specify a comma-separated list of the number of samples per fold (required)
  
Options:
  [-t --suffixTemp]             Specify the suffix for training/testing id and filename lists. Under default settings, a separate folder is created for each fold.
  [-s --shuffleList]            Shuffle the input lists (default: off)
  [-r --regresFlag]             Divide entire list into folds, rather than individual classes. This is used when creating folds for regression (default: off)



Example 1:
  %(EXEC)s -f Filename_List.lst  -i Ids_List.lst -n 5
  
Creates five pairs of ID/filename lists.

Example 2:
  %(EXEC)s -f Filename_List.lst  -i Ids_List.lst -n 10,10,10,10,10

Creates five pairs of ID/filename lists, each containing 10 samples.

Examples 3:
  %(EXEC)s -f Filename_List.lst  -i Ids_List.lst -n 5 -t _fold%d.txt


  
""" % {'EXEC': basis.exename()}
    basis.print_contact()


class FoldExp(object):
    def __init__(self, 
                 Dim=[1,1,1],numChannel=1, rootPath='',
                 trainList=[],trainlabelList=[],trainIDList=[],
                 testList=[],testlabelList=[],testIDList=[]):
        self.Dim = Dim
        self.numChannel = numChannel
        self.rootPath = rootPath
        self.trainList = trainList
        self.trainlabelList = trainlabelList
        self.trainIDList = trainIDList 
        self.testList = testList
        self.testlabelList = testlabelList
        self.testIDList = testIDList 

    def __get__(self, obj, objtype):
        return self.val

    def __set__(self, obj, val):
        self.val = val

    def dumpExpLists(self,trainListFn,trainIDListFn,testListFn,testIDListFn):
        # write the training list
        header = [str(len(self.trainList)) + '   ' + str(self.numChannel) +  '\n',
                  str(self.Dim[0]) + '  ' + str(self.Dim[1]) + '  ' + str(self.Dim[2]) + '\n',
                  self.rootPath + '\n']  
        lines = []
        lines = lines + header
        for cnt in range(0,len(self.trainList)):
              lines.append(self.trainList[cnt] + '   ' + str(self.trainlabelList[cnt]) + '\n')
        fid = open(trainListFn,'wt')
        fid.writelines(lines)
        fid.close()
        # write the testing list
        header = [str(len(self.testList)) + '   ' + str(self.numChannel) +  '\n',
                  str(self.Dim[0]) + '  ' + str(self.Dim[1]) + '  ' + str(self.Dim[2]) + '\n',
                  self.rootPath + '\n']  
        lines = []
        lines = lines + header
        for cnt in range(0,len(self.testList)):
              lines.append(self.testList[cnt] + '   ' + str(self.testlabelList[cnt]) + '\n')
        fid = open(testListFn,'wt')
        fid.writelines(lines)
        fid.close() 
        # write the training IDs
        lines = []
        for cnt in range(0,len(self.trainIDList)):
              lines.append(self.trainIDList[cnt] + '\n')
        fid = open(trainIDListFn,'wt')
        fid.writelines(lines)
        fid.close() 
        # write the testing IDs
        lines = []
        for cnt in range(0,len(self.testIDList)):
              lines.append(self.testIDList[cnt] + '\n')
        fid = open(testIDListFn,'wt')
        fid.writelines(lines)
        fid.close() 
 

# [numChannels, numFiles, Dim, rootPath, ImgFnList,ImgLabels] = readCompareList(fnList)
def readCompareList(fnList):
    regressFlag = False
    fid = open(fnList,'rt')
    lines = fid.readlines()
    fid.close()
    numFiles = int(lines[0].split()[0])
    numChannels = int(lines[0].split()[1])
    Dim = lines[1].split()
    Dim[0] = int(Dim[0])
    Dim[1] = int(Dim[1])
    Dim[2] = int(Dim[2])
    rootPath = lines[2].split()[0]
    if not os.path.isabs(rootPath):
        rootPath = os.path.join(os.path.dirname(fnList), rootPath)
    rootPath = os.path.realpath(rootPath)
    ImgFnList = [] 
    ImgLabels = [] 
    for cnt in range(3,len(lines)):
        if (len(lines[cnt].split())==numChannels+1):
            sps = lines[cnt].split()
            l = ''
            for cnt in range(0,len(sps)-1):
                l = l + sps[cnt] + '  '  
            ImgFnList.append(l)
            ImgLabels.append(float(sps[-1]))
            if (sps[-1].find('.')>-1):
                regressFlag = True
        else:
            assert False, "make sure you have provided correct number of channels in the filelist: " + fnList
    if not(regressFlag):
        # if it is classification problem numFiles should be a dictionary
        # find unique lables
        cnt = 0
        labels = []
        labels.append(int(ImgLabels[0]))
        tmpNumFiles = {}
        for l in ImgLabels:
            if not(l==labels[cnt]):
                cnt = cnt + 1
                labels.append(int(l))
        for l in labels:
            tmpNumFiles[l] = 0
        for l in ImgLabels:
            tmpNumFiles[l] = tmpNumFiles[l]+1
        numFiles = tmpNumFiles
    return numChannels, numFiles, Dim, rootPath, ImgFnList,ImgLabels  



#IDs = readIDsList(idList) 
def readIDsList(idListFn):
    fid = open(idListFn,'rt')
    lines = fid.readlines()
    fid.close()
    IDs = []
    for l in lines:
        if (len(l.split())==1):
            IDs.append(l.split()[0])
    return IDs


# ImgFnList_dict, IDs_dict = extractClass(ImgFnList,IDs,ImgLabels)
def extractClass(ImgFnList,IDs,ImgLabels):
    ImgFnList_dict = {}
    IDs_dict = {}
    # find unique lables
    cnt = 0
    labels = []
    labels.append(int(ImgLabels[0]))
    for l in ImgLabels:
        if not(l==labels[cnt]):
            cnt = cnt + 1
            labels.append(int(l))
    #  loop over possible labels and assign ids and Image files
    for l in labels:
        id_list = [] 
        fn_list = [] 
        for cnt in range(0,len(IDs)):
            if (ImgLabels[cnt]==l):
                id_list.append(IDs[cnt])
                fn_list.append(ImgFnList[cnt])
        ImgFnList_dict.update({l:fn_list})
        IDs_dict.update({l:id_list}) 
    return ImgFnList_dict, IDs_dict  


def shuffleList(mylist,myid,mylabels):
    index = range(0,len(mylist))
    random.shuffle(index)
    newlist = []
    newid = []
    newlabels = []
    for cnt in range(0,len(mylist)):
        newlist.append(mylist[index[cnt]])
        newid.append(myid[index[cnt]])
        if not(mylabels==[]):
            newlabels.append(mylabels[index[cnt]])
    if not(mylabels==[]):
        outputs = (newlist, newid, newlabels)
    else:
        outputs = (newlist, newid)
    return outputs



#  ImgFnList_dict, IDs_dict =  shuffleList(ImgFnList_dict, IDs_dict)
def shuffleDict(ImgFnList_dict, IDs_dict):
    newImgFnList_dict = {} 
    newIDs_dict = {}
    for k in ImgFnList_dict.keys():
        newFnlist,newIDlist = shuffleList(ImgFnList_dict[k],IDs_dict[k],[])
        newImgFnList_dict.update(newFnlist)
        newIDs_dict.update(newIDlist)
    return newIDs_dict, newIDs_dict
 

#FoldExpList = makeFolds(ImgFnList_dict, IDs_dict, [] , numTestSub, numChannels, Dim, rootPath, sortFlag)
#OR:
#FoldExpList = makeFolds(ImgFnList, IDs, ImgLabels, numTestSub, numChannels, Dim, rootPath, sortFlag)
def makeFolds(ImgFnList, IDs, ImgLabels , numTestSub, numChannels, Dim, rootPath, sortFlag):
    FoldExpList = []
    if (type(ImgFnList)==type({})):   # a dictionary is provided: classification problem
        ImgFnList_dict = ImgFnList
        IDs_dict = IDs
        cnt = {}
        numFolds = len(numTestSub[ ImgFnList_dict.keys()[0] ])
        for k in ImgFnList_dict.keys():
          cnt.update({k:0})   
        for foldCnt in range(0,numFolds):
            trainList = []
            trainlabelList = []
            trainIDList = []
            testList = []
            testlabelList = []
            testIDList = []
            for k in ImgFnList_dict.keys():
                testList_tmp = ImgFnList_dict[k][ cnt[k]:cnt[k] + numTestSub[k][ foldCnt ] ]
                testList = testList + testList_tmp 
                trainList_tmp = ImgFnList_dict[k][ 0:cnt[k] ] + ImgFnList_dict[k][ cnt[k] + numTestSub[k][ foldCnt ] : ]
                trainList = trainList + trainList_tmp
                testIDList_tmp = IDs_dict[k][ cnt[k]:cnt[k] + numTestSub[k][ foldCnt] ] 
                testIDList = testIDList + testIDList_tmp 
                trainIDList_tmp = IDs_dict[k][ 0:cnt[k] ] + IDs_dict[k][ cnt[k] + numTestSub[k][ foldCnt ] : ]
                trainIDList = trainIDList + trainIDList_tmp
                for i in range(0,len(IDs_dict[k]) - numTestSub[k][ foldCnt ]):
                      trainlabelList.append(k)
                for i in range(0,numTestSub[k][ foldCnt ] ):
                      testlabelList.append(k)
                cnt[k] = cnt[k] + numTestSub[k][foldCnt]
            f = FoldExp(Dim,numChannels, rootPath,\
                    trainList,trainlabelList,trainIDList,\
                    testList,testlabelList,testIDList)
            FoldExpList.append(f)
    if (type(ImgFnList)==type([])):   # a list is provided: regression problem
        cnt = 0
        for foldCnt in range(0,len(numTestSub)):
            trainList = []
            trainlabelList = []
            trainIDList = []
            testList = []
            testlabelList = []
            testIDList = []
            # making test/train ID and FnList
            testList = testList + ImgFnList[cnt:cnt+numTestSub[foldCnt]] 
            trainList = trainList + ImgFnList[0:cnt] + ImgFnList[cnt+numTestSub[foldCnt]:]
            testIDList = testIDList + IDs[cnt:cnt+numTestSub[foldCnt]] 
            trainIDList = trainIDList + IDs[0:cnt] + IDs[cnt+numTestSub[foldCnt]:]
            trainlabelList = trainlabelList +  ImgLabels[0:cnt] + ImgLabels[cnt+numTestSub[foldCnt]:]
            testlabelList  = testlabelList  + ImgLabels[cnt:cnt+numTestSub[foldCnt]]
            f = FoldExp(Dim,numChannels, rootPath,\
                    trainList,trainlabelList,trainIDList,\
                    testList,testlabelList,testIDList)
            FoldExpList.append(f)
            cnt = cnt + numTestSub[foldCnt]
    return FoldExpList                    

            
# numTestSub = getNumTestSample(numFiles,numFolds)
def getNumTestSample(numFiles,numFolds):
    if ( type(numFiles)==type(1) ):
        tmp = numpy.linspace(1,numFiles,numFolds+1)
        numTestSub = [] 
        cnt = 0 
        for i in range(1,len(tmp)):
            if ( i==len(tmp) ):
                tmpNum = int(numpy.floor(tmp[i] - cnt + 1))
                numTestSub.append(tmpNum)
            else:
                tmpNum = int(numpy.floor(tmp[i] - cnt))
                numTestSub.append(tmpNum)
                cnt = cnt + tmpNum
    elif ( type(numFiles)==type({}) ):
        numTestSub = {}
        for k in numFiles.keys():
            tmp = numpy.linspace(1,numFiles[k],numFolds+1)
            numTestSub[k] = [] 
            cnt = 0
            for i in range(1,len(tmp)):
                if ( i==len(tmp) ):
                    tmpNum = int(numpy.floor(tmp[i] - cnt + 1))
                    numTestSub.append(tmpNum)
                else:
                    tmpNum = int(numpy.floor(tmp[i] - cnt))
                    numTestSub[k].append(tmpNum)
                    cnt = cnt + tmpNum
    return numTestSub





def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hi:f:n:t:sr",\
            ["help", "idList=","fileList=","numFolds=","suffixTemp=","shuffleList","regresFlag"])
    except getopt.GetoptError, err:
        sys.stderr.write(err + '\n')
        return 1

    numReqOpt = 0   # number of required options
    idList = ''
    fnList = ''
    suffix_template = None
    shuffleFlag = False
    regresFlag = False
    numFolds = 1
    for o, a in opts:
        if o in ("-h", "--help"):
            usage()
            sys.exit(0)
        elif o in ("-i", "--idList"):
            idList = a
            numReqOpt = numReqOpt + 1
        elif o in ("-f","--fileList"):
            fnList = a
            numReqOpt = numReqOpt + 1
        elif o in ("-n","--numFolds"):
            if (a.find(',')>-1):
                numTestSub = []
                for n in a.split(','):
                    numTestSub.append(int(n))
                numFolds = len(numTestSub)
            else:
                numTestSub = []
                numFolds = int(float(a))
            numReqOpt = numReqOpt + 1
        elif o in ("-t","--suffixTemp"):
            suffix_template = a
        elif o in ("-r","--regresFlag"):
            regresFlag = True
        elif o in ("-s","--shuffleList"):
          shuffleFlag = True
        else:
            assert False, "unhandled option"

    if numReqOpt < 3:
        sys.stderr.write("Not all required arguments specified!\n")
        return 1

    if suffix_template and '%d' in suffix_template:
        sys.stderr.write("Invalid suffix template given! The template must contain the pattern '%d' as placeholder for the fold number.\n")
        return 1

    # read whole filename list
    [numChannels, numFiles, Dim, rootPath, ImgFnList,ImgLabels] = readCompareList(fnList)
    # read whole id lists
    IDs = readIDsList(idList) 
    # create two separate files and IDs
    if not(regresFlag):
        ImgFnList_dict, IDs_dict = extractClass(ImgFnList,IDs,ImgLabels)
    # should I shuffle the list
    if (shuffleFlag):
        # permute each list and corresponding id
        if not(regresFlag):
            ImgFnList_dict, IDs_dict =  shuffleDict(ImgFnList_dict, IDs_dict)
        else:
            ImgFnList, IDs, ImgLabels =  shuffleList(ImgFnList, IDs,ImgLabels) 
    # create training/testing list and ID lists
    if (numTestSub==[]):
        numTestSub = getNumTestSample(numFiles,numFolds)
    if not(regresFlag):
        FoldExpList = makeFolds(ImgFnList_dict, IDs_dict, [], numTestSub, numChannels, Dim, rootPath, False)
    else:
        FoldExpList = makeFolds(ImgFnList, IDs, ImgLabels , numTestSub, numChannels, Dim, rootPath, False)
    # loop over number of folds
    for i in range(numFolds):
        if suffix_template:
            suffix        = suffix_template % (i + 1)
            trainListFn   = "training%s" % suffix
            trainIDListFn = "trainids%s" % suffix
            testListFn    = "testing%s"  % suffix
            testIDListFn  = "testids%s"  % suffix
        else:
            folddir = "%d" % (i + 1)
            os.mkdir(folddir)
            trainListFn   = os.path.join(folddir, "training.lst")
            trainIDListFn = os.path.join(folddir, "trainids.lst")
            testListFn    = os.path.join(folddir, "testing.lst")
            testIDListFn  = os.path.join(folddir, "testids.lst")
        FoldExpList[i].dumpExpLists(trainListFn,trainIDListFn,testListFn,testIDListFn)
    return 0

if __name__ == '__main__': sys.exit(main())
