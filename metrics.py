def coefficients(predScore, realScore):
    global predCondition
    if(predScore == 1 and realScore == 1):
        predCondition = "True Positive"
    if(realScore == 0 and predScore == 1):
        predCondition = "False Positive"
    if(predScore == 0 and realScore == 1):
        predCondition = "False Negative"
    if(predScore == 0 and realScore == 0):
        predCondition = "True Negative"
    return predCondition

def mean(lst):
    return sum(lst) / len(lst)

def stddev(lst):
    mn = mean(lst)
    variance = sum([(e-mn)**2 for e in lst]) / len(lst)
    return sqrt(variance)

def recall(truePos, falseNeg):
    return truePos/(truePos+falseNeg)

def specificity(trueNeg, falsePos):
    return trueNeg/(trueNeg+falsePos)

def precision(truePos, falsePos):
    return truePos/(truePos+falsePos)

def negPredValue(trueNeg, falseNeg):
    return trueNeg/(trueNeg+falseNeg)

def fallOut(trueNeg, falsePos):
    return 1 - specificity(trueNeg, falsePos)

def falseDiscoveryRate(truePos, falsePos):
    return 1 - precision(truePos, falsePos)

def missRate(truePos, falseNeg):
    return 1 - recall(truePos, falseNeg)

def f1Rate(truePos, falsePos, trueNeg):
    return (2*truePos)/(2*truePos+falsePos+trueNeg)

def MCC(truePos, falsePos, trueNeg, falseNeg):
    MCCVal = ((truePos*trueNeg)-(falsePos*falseNeg))/sqrt((truePos+falsePos)(truePos+falseNeg)(trueNeg+falsePos)(trueNeg+falseNeg))
    return MCCVal

def informedness(truePos, falseNeg, trueNeg, falsePos):
    return recall(truePos, falseNeg) + specificity(trueNeg, falsePos) - 1

def markedness(truePos, falsePos, trueNeg, falseNeg):
    return precision(truePos, falsePos)+ negPredValue(trueNeg, falseNeg) - 1

def unweightedAvgRecall(truePos, trueNeg, falseNeg, falsePos):
    return ((truePos/(truePos+falseNeg))+(trueNeg/(trueNeg+falsePos)))/2