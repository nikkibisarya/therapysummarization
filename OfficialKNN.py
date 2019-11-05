from operator import itemgetter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import statistics
import numpy as np
from cmath import sqrt
from scipy.stats.stats import pearsonr
import sys
from psycholinguisticNorms import processText
import collections
filePath = '/home/nikki/clean_transcripts/'
fileName = 'transcript_splits_data_new.txt'
transcriptFile = '/home/nikki/transcripttoavgvector.txt'
counter = collections.Counter()
realEmpathyLabelFile = 'realEmpLabels.txt'
# get words from all the text files
def get_all_words(textFileNames):
    all_words = {}
    for textFileName in textFileNames:
        with open(textFileName, 'r') as file:
            for line in file:
                for word in line.split():
                    if word not in all_words:
                        all_words[word] = 0

    return all_words

def stop():
    stop()

def read_in_file(file):
    testSet = {}
    sessionName = ""
    numberVector = []
    sessionVec = []
    #read in file by line
    with open(file, 'r') as f:
        for line in f:
            line = line.replace(" ", "")
            #split by :
            lineVec = line.split(':')
            #first is session name
            sessionName = lineVec[0]
            #second is vector
            #split vector by ,
            numberVector = lineVec[1].split(',')
            #convert each of those to floats
            for num in numberVector:
                #push that into a vector
                sessionVec.append(float(num))
            #pair session name with vector in dictionary
            testSet[sessionName] = sessionVec[:]
            sessionVec[:] = []
    #return dictionary
    return testSet

#make text file vectors
def text_file_to_vec(filePath, allWords):
    # Copy for the text file.
    wordFreqVec = {}
    for word in allWords:
        wordFreqVec[word] = 0

    with open(filePath, 'r') as file:
        for line in file:
            for word in line.split():
                wordFreqVec[word] += 1

    retVec = []
    for word in wordFreqVec:
        retVec.append(wordFreqVec[word])

    return retVec

#find the distance between two text file vectors
def text_file_vec_distance(v1, v2):
    dist = 0
    if len(v1) != len(v2):
        print (len(v1))
        print (len(v2))
        raise ValueError('Not matching dimensions')

    for i in range(len(v1)):
        dist += (v1[i] - v2[i]) ** 2

    dist = dist/len(v1)
    return dist

# make a dictionary with the first row in the transcipts file and each of the sessions
def make_dictionary(line):
    with open(filePath, 'r') as file:
        first_line = file.readline()
        orig_list = first_line.split(',')
        dictionary = dict(zip(orig_list, line))# line should be desired session info line
        return dictionary
    # testSet = read_in_file(file) ????????????????

# get the k-nearest neighbors of a specific test instance using all of the training set
def get_neighbors(trainingSet, testInstance, k):
    distances = {}
    for x in range(len(trainingSet)):
        dist = text_file_vec_distance(testInstance['words'], trainingSet[x]['words'])
        distances[x] = dist
    tuples = sorted(distances.items(), key = itemgetter(1))
    neighbors = []
    for x in range(k):
        training_set_key = tuples[x][0]
        training_set_point = trainingSet[training_set_key]
        neighbors.append(training_set_point)
    return neighbors

def coefficients(predScore, realScore):
    global predCondition
    if(predScore >= 5 and realScore >= 5):
        predCondition = "True Positive"
    if(realScore < 5 and predScore >= 5):
        predCondition = "False Positive"
    if(predScore < 5 and realScore >= 5):
        predCondition = "False Negative"
    if(predScore < 5 and realScore < 5):
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

def convertFiletoDict(file):
    d = {}
    for line in file:
        lineContent = line.split(':')
        lineContent[0] =  lineContent[0].strip(' ')
        #print(lineContent[0])
        strVecContent = lineContent[1].split(',')

        float_vector = []
        for str in strVecContent:
            str.strip(' ')
            float_vector.append(float(str))
        d[lineContent[0]] = float_vector
    return d

def main(model):
    # prepare data
    allWords = {}
    trainingSet=[]
    testSet=[]
    truePos = 0
    trueNeg = 0
    falsePos = 0
    falseNeg = 0
    header = []
    data = []
    calculatedVectOfVectors = []
    sessionToNumVec = {}

    with open(transcriptFile, 'r') as readFile:
        sessionToNumVec = convertFiletoDict(readFile)
    with open(''.join([filePath,fileName]), 'r') as file:
        for line in file:
            if len(header) == 0:
                header = line.strip().split(',')
            else:
                cur_line = line.strip().split(',')
                dictionary = dict(zip(header, cur_line))
                data.append(dictionary)

   # f = open('C:\Python\text_scanner\output.txt','w')

    sessions = []
    textFileNames = []
    count = 0
    for cur_data in data:
        study_folder = cur_data['study']
        study_folder = study_folder.split('_')[0]

        session_filepath = filePath + cur_data['session'] + '.txt'
        textFileNames.append(session_filepath)

        test = cur_data['split.patient.70/30'] == 'test'
        words = sessionToNumVec[cur_data['session']]
        with open(session_filepath, 'r') as myfile:
            transcript = myfile.read().replace('\n', '')
        norms_ratings = processText(transcript).flatten()
        print('norms_ratings: ', norms_ratings)
        print('type of norms rating: ', type(norms_ratings))
        print('words: ', sessionToNumVec[cur_data['session']])
        print('type words: ', type(sessionToNumVec[cur_data['session']]))
        sessionToNumVec[cur_data['session']].extend(norms_ratings)
        print('changed words: ', sessionToNumVec[cur_data['session']])
        sessions.append({
            'test': test,
            'session_filepath': session_filepath,
            'session_name': cur_data['session'],
            'words': sessionToNumVec[cur_data['session']],
            'empathy': float(cur_data['empathy'])
            })
    allWords = get_all_words(textFileNames)
    
    
    
    transcript_to_vec = read_in_file(transcriptFile)

#     for session in sessions:
#          session['words'] = text_file_to_vec(session['session_filepath'], allWords)
#          session['words'] = transcript_to_vec[session['session_name']]
#          print(len(session['words']))

    test_set = [session for session in sessions if session['test']]
    training_set = [session for session in sessions if not session['test']]


    # Fill these out with data from training_set
    X = [] #word vectors?
    Y = [] #empathy ratings? y > 5 is 1, y < 5 is 0

    # Fill these out with data from test_set
    test_X = [] #word vectors?
    test_Y = [] #empathy ratings?

    for set in test_set:
        count = count + 1
        test_X.append(set['words'])
        test_Y.append(set['empathy'])
    for set in training_set:
        X.append(set['words'])
        Y.append(set['empathy'])

    # Fit model to the data (call your .fit(X, Y) function)
  #  model.fit(np.array(X), np.array(Y,np.int8))

  #  generate predictions
    k = 3
    feature_labels = []
    accuracy = []
    pred_empathy_labels = []
    real_empathy_labels = []
    pred_empathy_list = []
    real_empathy_list = []
    numerator = 0
    first_den = 0
    second_den = 0
    pred_y = []
 #   predictions_y = model.predict(np.array(test_X))
 #   print("Mean Squared Error: ", mean_squared_error(test_Y, predictions_y))
    # Use KNeighborsClassifer's kneighbors
  #  array_of_closest_indices = model.kneighbors(test_X, return_distance=False)

#     for closest_indices in array_of_closest_indices:
#         neighbors = []
#         for x_index in closest_indices:
#             neighbors.append(training_set[x_index])

    for x in range(len(test_set)):
        neighbors = get_neighbors(training_set, test_set[x], k)
        total_empathy_score = 0.0
        for neighbor in neighbors:
             total_empathy_score += neighbor['empathy']
        pred_empathy_score = float(total_empathy_score) / float(k)
        pred_y.append(pred_empathy_score)
        for pred_empathy_score in pred_y:
            pred_empathy_list.append(pred_empathy_score)
        real_empathy_score = test_set[x]['empathy']
        real_empathy_list.append(real_empathy_score)
        print(str(test_set[x]['session_name']))
        print('Predicted Score:' + str(pred_empathy_score))
        print('Real Score:' + str(real_empathy_score))
        pred_Conditions = coefficients(pred_empathy_score, real_empathy_score)
        if pred_Conditions == "True Positive":
            truePos += 1
        elif pred_Conditions == "True Negative":
            trueNeg += 1
        elif pred_Conditions == "False Positive":
            falsePos += 1
        else:
            falseNeg += 1
        print('Predicted Condition: ' + pred_Conditions)
        pred_empathy_labels.append(pred_empathy_score >= 5)

        real_empathy_labels.append(real_empathy_score >= 5)

        # Pearson's Correlation Coefficient
        first_call = mean(pred_empathy_list)
        second_call = mean(real_empathy_list)
        numerator += (pred_empathy_score - first_call)*(real_empathy_score - second_call)
        first_den += (pred_empathy_score - first_call)*(pred_empathy_score - first_call)
        second_den += (real_empathy_score - second_call)*(real_empathy_score - second_call)
    print('predicted emp list', pred_empathy_list)
    print('real emp list', real_empathy_list)

    print('training set: ', len(Y))
    print('test set', len(test_Y))
    counter = collections.Counter(test_Y)
    print('counter', counter)
    with open(realEmpathyLabelFile, 'w') as file:
        file.write(str(real_empathy_list))

    r = numerator / (sqrt(first_den)*sqrt(second_den))
    total = 0

    for x in range(len(pred_empathy_labels)):
        total += int(pred_empathy_labels[x] == real_empathy_labels[x])
    count = 0
    falseCount = 0
    for label in real_empathy_labels:
            print('label: ', label)
            if label == True:
                count = count + 1
            else:
                falseCount = falseCount + 1

    print('len true: ', count)
    print('len false: ', falseCount)
    print("Average Accuracy: " + str(total/len(pred_empathy_labels)))
    print("Pearson Correlation Coefficient: ", str(r))
    print("Recall: " + str(recall(truePos, falseNeg)))
    print("Unweighted Avg. Recall: " + str(unweightedAvgRecall(truePos, trueNeg, falseNeg, falsePos)))
 #   print("Specificity: " + str(specificity(trueNeg, falsePos)))
#     print("Precision: " + str(precision(truePos, falsePos)))
#     print("Negative Predictive Value: " + str(negPredValue(trueNeg, falseNeg)))
#     print("Fall Out/False Positive Rate: " + str(fallOut(trueNeg, falsePos)))
#     print("False Discovery Rate: " + str(falseDiscoveryRate(truePos, falsePos)))
#     print("Miss Rate/ False Negative Rate: " + str(missRate(truePos, falseNeg)))
#     print("F1 Score: " + str(f1Rate(truePos, falsePos, trueNeg)))
#
#     print("Informedness: " + str(informedness(truePos, falseNeg, trueNeg, falsePos)))
#     print("Markedness: " + str(markedness(truePos, falsePos, trueNeg, falseNeg)))
   # f.close()


main(KNeighborsClassifier(n_neighbors=3))
#main(LinearRegression())

# if they are >= 5 set accuracy to 1 otherwise set them to 0
#for each session, make a tuple that includes its empathy rating
#make a dictionary of all of the words
#find k closest neighbors
#pull empathy rating from those k and return the most common one (the average rounded up)


