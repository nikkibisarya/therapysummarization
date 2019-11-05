# SVM, MLP, LDA -- Classification
# Support Vector Regression
# what type of parameters they will need
# write a report comparing these and their results in terms of unweighted average recall for classification
# and pearson's correlation for regression
from sklearn import svm
from sklearn.svm import SVR
from sklearn.svm import SVC
from metrics import unweightedAvgRecall
from metrics import f1Rate
from metrics import coefficients
from sklearn.preprocessing import normalize
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from operator import itemgetter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import recall_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import statistics
from cmath import sqrt
from sklearn import preprocessing
from sklearn import utils
import sys
from psycholinguisticNorms import processText

filePath = '/home/nikki/clean_transcripts/'
fileName = 'transcript_splits_data_new.txt'
transcriptFile = '/home/nikki/transcripttoavgvector.txt'


def read_in_file(file):
    testSet = {}
    sessionVec = []
    # read in file by line
    with open(file, 'r') as f:
        for line in f:
            line = line.replace(" ", "")
            # split by :
            lineVec = line.split(':')
            # first is session name
            sessionName = lineVec[0]
            # second is vector
            # split vector by ,
            numberVector = lineVec[1].split(',')
            # convert each of those to floats
            for num in numberVector:
                # push that into a vector
                sessionVec.append(float(num))
            # pair session name with vector in dictionary
            testSet[sessionName] = sessionVec[:]
            sessionVec[:] = []
    # return dictionary
    return testSet


def convertFiletoDict(file):
    d = {}
    for line in file:
        lineContent = line.split(':')
        lineContent[0] = lineContent[0].strip(' ')
        strVecContent = lineContent[1].split(',')
        float_vector = []
        for str in strVecContent:
            str.strip(' ')
            float_vector.append(float(str))
        d[lineContent[0]] = float_vector
    return d


def classifier(X, Y, test_X, model):
    model.fit(np.array(X), np.array(Y))
    predictions_y = model.predict(np.array(test_X))
    return predictions_y


def regressor(X, Y, test_X, model):
    model.fit(np.array(X), np.array(Y))
    predictions_y = model.predict(np.array(test_X))
    return predictions_y


def main(model, isClassifier):
    # prepare data
    header = []
    data = []
    f = open("results.txt", "a+")

    with open(transcriptFile, 'r') as readFile:
        sessionToNumVec = convertFiletoDict(readFile)
    with open(''.join([filePath, fileName]), 'r') as file:
        for line in file:
            if len(header) == 0:
                header = line.strip().split(',')
            else:
                cur_line = line.strip().split(',')
                dictionary = dict(zip(header, cur_line))
                data.append(dictionary)
    sessions = []
    for cur_data in data:
        test = cur_data['split.patient.70/30'] == 'test'
        session_filepath = filePath + cur_data['session'] + '.txt'
        with open(session_filepath, 'r') as myfile:
            transcript = myfile.read().replace('\n', '')
        norms_ratings = processText(transcript).flatten()
        sessionToNumVec[cur_data['session']].extend(norms_ratings)
        sessions.append({
            'test': test,
            'session_name': cur_data['session'],
            'words': sessionToNumVec[cur_data['session']],
            'empathy': float(cur_data['empathy'])
        })
    test_and_val_set = [session for session in sessions if session['test']]
    transcript_split = len(test_and_val_set)
    val_set_cap = (2 / 3) * transcript_split
    cap = 0

    # Fill these out with data from training_set
    X = []  # word vectors?
    Y = []  # empathy ratings? y > 5 is 1, y < 5 is 0
    # Fill these out with data from test_set
    test_X = []  # word vectors?
    test_Y = []  # empathy ratings?
    validation_X = []
    validation_Y = []
    Y_binary = []
    test_Y_binary = []
    validation_Y_binary = []
    for session in sessions[:2]:
        if session['test']:
            if cap < val_set_cap:
                validation_X.append(session['words'])
                validation_Y.append(session['empathy'])
                cap = cap + 1
            else:
                test_X.append(session['words'])
                test_Y.append(session['empathy'])
        else:
            X.append(session['words'])
            Y.append(session['empathy'])
    print('validation set: ', len(validation_X), ' test set: ', len(test_X), ' training set: ', len(X))
    oneCount = 0
    twoCount = 0
    threeCount = 0
    fourCount = 0
    fiveCount = 0
    sixCount = 0
    sevenCount = 0
    for empathy_rating in Y:
        if 1 <= empathy_rating < 1.5:
            oneCount = oneCount + 1
        elif 1.5 <= empathy_rating < 2.5:
            twoCount = twoCount + 1
        elif 2.5 <= empathy_rating < 3.5:
            threeCount = threeCount + 1
        elif 3.5 <= empathy_rating < 4.5:
            fourCount = fourCount + 1
        elif 4.5 <= empathy_rating < 5.5:
            fiveCount = fiveCount + 1
        elif 5.5 <= empathy_rating < 6.5:
            sixCount = sixCount + 1
        elif 6.5 <= empathy_rating < 7.5:
            sevenCount = sevenCount + 1
  #  print('One: ', oneCount, 'Two: ', twoCount, 'Three: ', threeCount, 'Four: ', fourCount, 'Five: ', fiveCount, 'Six: ', sixCount, 'Seven: ', sevenCount)


    X = normalize(X)  # matrix of # of samples by # of features, check how these are aligned
    test_X = normalize(test_X)
    validation_X = normalize(validation_X)
    if isClassifier:
        truePos = 0
        trueNeg = 0
        falsePos = 0
        falseNeg = 0
        num = 0
        # reassign  to another list (Y_binary)
        for empathy_rating in Y:
            if empathy_rating < 5:
                empathy_rating = 0
                Y_binary.append(empathy_rating)
            else:
                empathy_rating = 1
                Y_binary.append(empathy_rating)
        for empathy_rating in test_Y:
            if empathy_rating < 5:
                empathy_rating = 0
                test_Y_binary.append(empathy_rating)
            else:
                empathy_rating = 1
                test_Y_binary.append(empathy_rating)
        for empathy_rating in validation_Y:
            if empathy_rating < 5:
                empathy_rating = 0
                validation_Y_binary.append(empathy_rating)
            else:
                empathy_rating = 1
                validation_Y_binary.append(empathy_rating)

        predictions_y = classifier(X[:2], Y_binary[:2], validation_X[:2], model)

        #         print('y binary: ', Y_binary)
        #         one = 0
        #         zero = 0
        #         for y in Y_binary:
        #             if y:
        #                 one = one + 1
        #             else:
        #                 zero = zero + 1
        #         print('num of one: ', one)
        #         print('num of zero: ', zero)
        f.write(str(model) + '\nRecall Score: ' + str(recall_score(validation_Y_binary[:2], predictions_y, average='macro')))
     #    for pred_empathy_score in predictions_y:
     #        pred_Conditions = coefficients(pred_empathy_score, validation_Y_binary[num])
     #        if pred_Conditions == "True Positive":
     #            truePos += 1
     #        elif pred_Conditions == "True Negative":
     #            trueNeg += 1
     #        elif pred_Conditions == "False Positive":
     #            falsePos += 1
     #        else:
     #            falseNeg += 1
     #       num = num + 1
      #  f.write('\nUnweighted Avg Recall: ' + str(unweightedAvgRecall(truePos, trueNeg, falseNeg, falsePos)))
      #  f.write('\nF1Score by hand: ' + str((2*truePos)/(2*truePos+falsePos+trueNeg)))
        f.write('\nMicro F1Score: ' + str(f1_score(validation_Y_binary[:2], predictions_y, average='micro')))
        f.write('\nMacro F1Score: ' + str(f1_score(validation_Y_binary[:2], predictions_y, average='macro')))
#get classwise macro and micro f1
    else:
        predictions_y = regressor(X, Y, validation_X, model)
        f.write(str(model) + ': \nMean Squared Error: ' + str(mean_squared_error(validation_Y, predictions_y)))
        f.write('\nPearsons Correlation Coefficient: ' + str(pearsonr(validation_Y, predictions_y)))
        f.write('\nMicro F1Score: ' + str(f1_score(validation_Y_binary[:2], predictions_y, average = 'micro')))
        f.write('\nMacro F1Score: ' + str(f1_score(validation_Y_binary[:2], predictions_y, average='macro')))
    f.write('\n')
    f.close()
    # summarize history for loss

  #  plt.plot(model.estimator.loss_curve_)
  #  plt.title('model loss')
   # plt.ylabel('loss')
  #  plt.xlabel('epoch')
 #   plt.legend(['train', 'test'], loc='upper left')
  #  plt.savefig('MLAloss.png')

f = open('results.txt', 'a')

main(svm.SVC(C=10, class_weight='balanced', kernel='linear'), True)
main(MLPClassifier(activation='logistic', learning_rate='constant'), True)
main(LinearDiscriminantAnalysis(solver='svd'), True)
main(svm.SVR(C=10, kernel='linear'), False)