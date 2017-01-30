from NeuralNetUtil import buildExamplesFromCarData,buildExamplesFromPenData, buildExamplesFromXorData, buildExamplesFromExtraData
from NeuralNet import buildNeuralNet
import cPickle 
from math import pow, sqrt

def average(argList):
    return sum(argList)/float(len(argList))

def stDeviation(argList):
    mean = average(argList)
    diffSq = [pow((val-mean),2) for val in argList]
    return sqrt(sum(diffSq)/len(argList))

penData = buildExamplesFromPenData() 
def testPenData(hiddenLayers = [24]):
    return buildNeuralNet(penData,maxItr = 200, hiddenLayerList =  hiddenLayers)

carData = buildExamplesFromCarData()
def testCarData(hiddenLayers = [16]):
    return buildNeuralNet(carData,maxItr = 200,hiddenLayerList =  hiddenLayers)

extraData = buildExamplesFromExtraData()
def testExtraData(hiddenLayers = [80]):
    return buildNeuralNet(extraData,alpha = 10,maxItr = 200,hiddenLayerList =  hiddenLayers)

	
xorData = ( [ ( [0,0],[0] ), 
              ( [0,1],[1] ),
              ( [1,0],[1] ),
              ( [1,1],[0] ) ],
              
            [ ( [0,0],[0] ), 
              ( [0,1],[1] ),
              ( [1,0],[1] ),
              ( [1,1],[0] ) ] )
			  

def testXorData(hiddenLayers = [6]):
    return buildNeuralNet(xorData, maxItr = 200, alpha = 10, hiddenLayerList = hiddenLayers)

def printResults( tuple ):
    results, max, avg, stdev = tuple

    print
    print "Raw results:"
    print results
    print
    print "Max:      " + str(max)
    print "Average:  " + str(avg)
    print "Std.Dev.: " + str(stdev)

def runXorTest(func, name, hiddenLayers = None):
    print 
    if hiddenLayers == None:
        print "Running 5 iterations of " + name + " with default number of hidden layers."
    else:
        print "Running 5 iterations of " + name + " with " + str(hiddenLayers) + " hidden layers."
    print

    results = []
        
    for i in xrange(5):
        if hiddenLayers == None:
            results.append(func()[1])
        elif hiddenLayers == 0:
            results.append(func([])[1])
        else:
            results.append(func([hiddenLayers])[1])

    if hiddenLayers == None:
        print "Running 5 iterations of " + name + " with default number of hidden layers."
    else:
        print "Running 5 iterations of " + name + " with " + str(hiddenLayers) + " hidden layers."
    print
    print "Raw results:"
    print results
    print
    print "Max:      " + str(max(results))
    print "Average:  " + str(average(results))
    print "Std.Dev.: " + str(stDeviation(results))

    print
    print "######################"
    print

    return (results, max(results), average(results), stDeviation(results))

import sys
def main():
    args = sys.argv
    if args[1] == "-q7":
        if len(args) <= 2:
            hiddenLayers = None
        else:
            hiddenLayers = int(args[2])

        resultsXor = runXorTest(testXorData, "testXorData", hiddenLayers)

if __name__=='__main__':
	main()