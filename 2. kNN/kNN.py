import numpy as np
from collections import Counter
from os import listdir


def createDataSet():
    group = np.array([[1., 1.1], [1., 1.], [0., 0.], [0., 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def kNN_classifier0(point, group, labels, k):
    point = np.array(point)
    labels = np.array(labels)
    diffs = point - group
    distances = np.linalg.norm(diffs, axis=1)

    sortedIndices = distances.argsort()
    topkDistances = sortedIndices[:k]
    counts = Counter(labels[topkDistances])
    return counts.most_common()[0][0]


def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())
    returnMat = np.empty((numberOfLines, 3))
    classLabelVector = []
    fr = open(filename)
    for index, line in enumerate(fr.readlines()):
        listFromLine = line.strip().split('\t')
        returnMat[index, :] = listFromLine[: 3]
        classLabelVector.append(listFromLine[-1])
    return returnMat, classLabelVector


def autoNorm(dataSet):
    minVals, maxVals = dataSet.min(0), dataSet.max(0)
    ranges = maxVals - minVals
    return (dataSet - minVals) / ranges, ranges, minVals


def datingClassTest(filename='datingTestSet.txt', hoRatio=0.1):
    datingDataMat, datingLabels = file2matrix(filename)
    normMat, _, _ = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = kNN_classifier0(
            normMat[i, :],
            normMat[numTestVecs:m, :],
            datingLabels[numTestVecs: m],
            3
        )
        print(f'the classifier came back with {classifierResult}, the real answer is {datingLabels[i]}')
        if classifierResult != datingLabels[i]: errorCount += 1
    print(f'the total error rate is: {errorCount / numTestVecs}')


def classifyPerson(filename='datingTestSet.txt'):
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequency flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consume per year?"))
    datingDataMat, datingLabels = file2matrix(filename)
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles, percentTats, iceCream])
    classifierResult = kNN_classifier0(
            (inArr - minVals) / ranges,
            normMat,
            datingLabels,
            3
        )
    print(f"You will probably like this person {classifierResult}")


def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect


def get_data(m, FileList, data_type):
    hwLabels = []
    digits = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = FileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        digits[i, :] = img2vector(f'{data_type}Digits/{fileNameStr}')
    return hwLabels, digits


def handwritingClassTest():
    trainingFileList, testFileList = listdir('trainingDigits'), listdir('testDigits')
    m, mTest = len(trainingFileList), len(testFileList)
    hwLabels, trainingMat = get_data(m, trainingFileList, 'training')
    hwTestLabels, testMat = get_data(mTest, testFileList, 'test')
    errorCount = 0
    for i in range(mTest):
        classifierResult = kNN_classifier0(
            testMat[i],
            trainingMat,
            hwLabels,
            3
        )
        print(f'the classififer came back with {classifierResult}, the real answer is {hwTestLabels[i]}')
        if classifierResult != hwTestLabels[i]: errorCount += 1
    print(f'the total number of errors is {errorCount}')
    print(f'the total eroor rate is {errorCount / mTest}')
