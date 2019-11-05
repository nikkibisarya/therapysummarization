from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import re, os, csv
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import numpy as np

textFileName = '/home/nikki/text_scanner/Usable Metadata.csv'
newTestFiles = '/home/nikki//Test Files/'

class_clusters = {
    '6': ['depression emotion', 'anger', 'mania', 'sadness', 'confusion', 'paranoia', 'panic', 'apathy', 'restlessness',
          'despair', 'guilt', 'resentment', 'indecisiveness'],
    '3': ['inattentiveness', 'suicidal ideation', 'fantasizing', 'obsessive behavior', 'isolation', 'racing thoughts',
          'withdrawn', 'dreams', 'social inhibition', 'problems concentrating', 'compulsive behavior',
          'suicidal behavior', 'severe sensitivity', 'cutting', 'aggression', 'acting out', 'danger to others',
          'academic failure', 'detached behavior', 'disorganized thoughts', 'danger to self', 'rash', 'avoidance',
          'deceitfulness'],
    '2': ['insomnia'],
    '0': ['fearfulness', 'suspiciousness'],
    '7': ['hallucinations', 'dysphoria', 'hypersomnia', 'hyperphagia', 'enuresis', 'anhedonia', 'dysphagia',
          'vomiting'],
    '1': ['fatigue', 'chronic pain', 'crying', 'loss of appetite', 'delusions', 'general pain', 'tremors', 'nausea',
          'headache', 'back pain', 'withdrawal sickness', 'sweating', 'fainting', 'itching', 'stuttering', 'seizures',
          'phantom pain']}


# removes anything within brackets including brackets for a given string
def remove_invalid(s):
    s = str(s)
    initial = 0
    i = 0
    while i < len(s):
        c = s[i]
        if c == '{':
            initial = i
            i += 1
        if c == '}':
            tmp = s[:initial]
            if i + 1 < len(s):
                tmp = tmp + s[i + 1:]
            s = tmp
            i = initial
        else:
            i += 1
    return s


# reads in transcripts from a folder containing pure transcript files
def testingModelTranscripts():
    newX_test = []
    for file in os.listdir(newTestFiles):
        print('file: ', file)
        with open(''.join([newTestFiles, file]), 'r') as f:
            lines = f.readlines()
            for line in lines:
                cleanLine = remove_invalid(line)
                line = cleanLine
            lines = str(lines)
            newX_test.append(lines)
            print('finished ', file)
    return newX_test

# pairs each fileID with its transcript
def corpus_dic(csvfile):
    csvfile = open(csvfile, 'rt')
    csvreader = csv.reader(csvfile, delimiter=',')
    corpus = {}
    i = 0
    for row in csvreader:
        if i != 0:
            fileID = str(row[1])
            if fileID not in corpus.keys():
                fileID = str(fileID)
                corpus[fileID] = ''
            corpus[fileID] = corpus[fileID] + row[-1]
        i = i + 1
    return corpus


# prepares dictionary of key of fileID with underlying multiple underlying keys
def ctrn_metadata():
    csvfile = open(textFileName, 'r', encoding="latin-1")
    csvreader = csv.reader(csvfile, delimiter=',')
    data = {}
    i = 0
    symcount = 0
    validCount = 0
    validSummaryCount = 0
    symptomDict = {}
    corpus = corpus_dic('/home/nikki/text_scanner/General_psychtx_corpus_phase1.1.csv')
    for row in csvreader:
        if i != 0:
            # GET FILE IDS
            if row[4] != '':
                getID = row[4].split('->')
                if getID[0] != 'NA':
                    if len(getID) == 1:
                        fileID = str(getID[0])
                    else:
                        fileID = str(getID[1])
                    if row[21] == 'NA':
                        symcount += 1
                    if row[21] != 'NA':
                        symptoms = row[21].split(';')
                        for symptom in symptoms:
                            if symptom in symptomDict:
                                symptomDict[symptom] += 1
                            else:
                                symptomDict[symptom] = 1
                        data[fileID] = {}
                        data[fileID]['symptoms'] = symptoms
                        data[fileID]['valid transcript'] = False
                        data[fileID]['valid summary'] = False
                        data[fileID]['label_num'] = -1
                        data[fileID]['cluster_presence'] = []
                        data[fileID]['weight'] = 1
                        data[fileID]['cluster_symp'] = []
                        if fileID in corpus:
                            data[fileID]['transcript'] = corpus[fileID]
                            data[fileID]['valid transcript'] = True
                            validCount += 1
                        if ':' in row[5]:
                            summary = row[5].split(":")[1]
                            data[fileID]['summary'] = summary
                            data[fileID]['valid summary'] = True
                            validSummaryCount += 1
        i = i + 1
    print('symptom dict: ', symptomDict)
    print('Overall, number of valid transcripts?: ', validCount)
    print('number of valid summaries: ', validSummaryCount)
    print('no symptoms: ', symcount)
    return data


# converts each symptom to its respective cluster number
def symptom2cluster(symptom, class_clusters=class_clusters):
    cluster = ''
    symptom = symptom.strip().lower()
    symptom = re.sub(r'[^\w\s]', '', symptom)
    for key in class_clusters.keys():
        if symptom in class_clusters[key]:
            cluster = int(key)
            break
    return cluster


# determines whether transcript has symptoms that exist in clusters and if not, marks them as invalid
def clusterPresence(data):
    count = 0
    for fileID in data.keys():
        hasSymp = False
        for symp in data[fileID]['symptoms']:
            num = symptom2cluster(symp)
            if str(num) != '':
                if str(num) in data[fileID]['cluster_presence']:
                    continue
                else:
                    hasSymp = True
                    data[fileID]['cluster_presence'].append(str(num))
        if hasSymp == False:
            data[fileID]['valid transcript'] = False
            data[fileID]['valid summary'] = False
            count += 1
    print('invalid transcripts from cluster presence method: ', count)


# function not used, for a vector of 6 numbers (corresponding to clusters)
# marks presence of cluster w/ '1' in corresponding index
def clusterVectCreation(data):
    clusterIndex = -1
    for fileID in data.keys():
        for symptom in data[fileID]['symptoms']:
            clusterIndex = symptom2cluster(symptom)
            if clusterIndex == '':
                data[fileID]['valid transcript'] = False
            else:
                data[fileID]['cluster_symp'][clusterIndex] = 1


# function not used, for a vector of 6 numbers (corresponding to clusters),
# increments each cluster according to amount of symptom presence from that cluster
def labelEncoder(data):
    labels = {}
    num = 0
    for fileID in data.keys():
        if str(data[fileID]['cluster_symp']) in labels.keys():
            continue
        else:
            labels[str(data[fileID]['cluster_symp'])] = num
            num += 1
    print('num: ', num)
    return labels


# function not used
def replaceLabelVects(labels, data):
    for fileID in data.keys():
        data[fileID]['label_num'] = labels[str(data[fileID]['cluster_symp'])]


# verify accuracy by counting actual and predicted cluster appearances
def countPred(pred_y, y_train):
    classZero = 0
    classOne = 0
    classTwo = 0
    classThree = 0
    classSix = 0
    classSeven = 0
    intVect = []
    cZero = 0
    cOne = 0
    cTwo = 0
    cThree = 0
    cSix = 0
    cSeven = 0
    iVect = []
    for vect in pred_y:
        for char in vect:
            if char == 1:
                intVect.append(int(char))
            if char == 0:
                intVect.append(int(char))
        if intVect[0] == 1:
            classZero += 1
        if intVect[1] == 1:
            classOne += 1
        if intVect[2] == 1:
            classTwo += 1
        if intVect[3] == 1:
            classThree += 1
        if intVect[4] == 1:
            classSix += 1
        if intVect[5] == 1:
            classSeven += 1
        del intVect[:]
    for vect in y_train:
        for char in vect:
            if char == 1:
                iVect.append(int(char))
            if char == 0:
                iVect.append(int(char))
        if iVect[0] == 1:
            cZero += 1
        if iVect[1] == 1:
            cOne += 1
        if iVect[2] == 1:
            cTwo += 1
        if iVect[3] == 1:
            cThree += 1
        if iVect[4] == 1:
            cSix += 1
        if iVect[5] == 1:
            cSeven += 1
        del iVect[:]
    print('Actual Instances of Cluster 0: ', cZero)
    print('Actual Instances of Cluster 1: ', cOne)
    print('Actual Instances of Cluster 2: ', cTwo)
    print('Actual Instances of Cluster 3: ', cThree)
    print('Actual Instances of Cluster 6: ', cSix)
    print('Actual Instances of Cluster 7: ', cSeven)
    print('Predicted Instances of Cluster 0: ', classZero)
    print('Predicted Instances of Cluster 1: ', classOne)
    print('Predicted Instances of Cluster 2: ', classTwo)
    print('Predicted Instances of Cluster 3: ', classThree)
    print('Predicted Instances of Cluster 6: ', classSix)
    print('Predicted Instances of Cluster 7: ', classSeven)


# predicting part of one vs rest, prints results
def predictor(count_vect_X, count_vect_Y, strictTestSet_X):
    X_train, X_test, y_train, y_test = train_test_split(count_vect_X, count_vect_Y, test_size=0.33, random_state=42)
    ovr = OneVsRestClassifier(LinearSVC(random_state=0))
    ovr.fit(X_train, y_train)
    pred_y = ovr.predict(X_test)
    # 5 test files
    pred_newY = ovr.predict(strictTestSet_X)
    countPred(pred_y, y_train)
    print('F1 score avg = MICRO: ', f1_score(y_test, pred_y, average='micro', labels=np.unique(pred_y)))
    print('F1 score avg = NONE: ', f1_score(y_test, pred_y, average=None, labels=np.unique(pred_y)))
    recall = recall_score(y_test, pred_y, average=None)
    print('Recall: ', recall, '\n')
    print('TESTED TRANSCRIPTS SYMPTOM LABELS: ', pred_newY)
    return pred_y


# too many instances of Cluster 6 so duplicates instances that don't show Cluster 6
def balanceClusters(data):
    count = 0
    for fileID in data.keys():
        if not data[fileID]['cluster_presence']:
            count += 1
    for fileID in data.keys():
        createDuplicate = True
        if data[fileID]['cluster_presence']:
            for index in data[fileID]['cluster_presence']:
                if index == '6':
                    createDuplicate = False
                else:
                    continue
            if createDuplicate == True:
                data[fileID]['weight'] += 3
    for fileID in data.keys():
        createDuplicate = True
        if data[fileID]['cluster_presence']:
            for index in data[fileID]['cluster_presence']:
                if index == '6':
                    createDuplicate = False
                if index == '3':
                    createDuplicate = False
                else:
                    continue
            if createDuplicate == True:
                data[fileID]['weight'] += 2
    for fileID in data.keys():
        createDuplicate = True
        if data[fileID]['cluster_presence']:
            for index in data[fileID]['cluster_presence']:
                if index == '6':
                    createDuplicate = False
                if index == '3':
                    createDuplicate = False
                if index == '1':
                    createDuplicate = False
                else:
                    continue
            if createDuplicate == True:
                data[fileID]['weight'] += 22
    for fileID in data.keys():
        createDuplicate = True
        if data[fileID]['cluster_presence']:
            for index in data[fileID]['cluster_presence']:
                if index == '2':
                    continue
                else:
                    createDuplicate = False
            if createDuplicate == True:
                data[fileID]['weight'] += 40


# data organized for algorithm here into features and labels. screens for faulty data. calls one v rest.
def onevsrest(ctrn_meta):
    count = 0
    countother = 0
    sumCount = 0
    sumCountother = 0
    for fileID in ctrn_meta.keys():
        if ctrn_meta[fileID]['valid transcript'] == True:
            count += 1
        if ctrn_meta[fileID]['valid summary'] == True:
            sumCount += 1
    print('Amount of Valid Transcripts: ', count)
    print('Amount of Valid Summaries: ', sumCount)
    count_vect = TfidfVectorizer()
    dataX = []
    dataY = []
    dataYsum = []
    dataZ = []
    clusterPresence(ctrn_meta)
    #    balanceClusters(ctrn_meta)
    for fileID in ctrn_meta.keys():
        if ctrn_meta[fileID]['valid transcript'] == True:
            countother += 1
            i = 1
            for i in range(ctrn_meta[fileID]['weight']):
                dataX.append(ctrn_meta[fileID]['transcript'])
                dataY.append(ctrn_meta[fileID]['cluster_presence'])
                i += 1
        if ctrn_meta[fileID]['valid summary'] == True:
            try:
                sumCountother += 1
                i = 1
                for i in range(ctrn_meta[fileID]['weight']):
                    dataZ.append(ctrn_meta[fileID]['summary'])
                    dataYsum.append(ctrn_meta[fileID]['cluster_presence'])
                    i += 1
            except:
                print('ctrn_meta[fileID]: ', ctrn_meta[fileID])
    print('Amount of Valid Transcripts after clust pres: ', countother)
    print('Amount of Valid summaries after clust pres: ', sumCountother)
    count_vect.fit(dataX)
    print('length dataX: ', len(dataX))
    # using transcripts
    count_vect_X = count_vect.transform(dataX)
    mlb = MultiLabelBinarizer()
    count_vect_Y = mlb.fit_transform(dataY)
    newX_test = testingModelTranscripts()
    strictTestSet_X = count_vect.transform(newX_test)
    print('Using Transcripts:')
    predictor(count_vect_X, count_vect_Y, strictTestSet_X)
    # using summaries
    count_vect_Z = count_vect.transform(dataZ)
    count_vect_Ysum = mlb.fit_transform(dataYsum)
    print('Using Summaries:')
    predictor(count_vect_Z, count_vect_Ysum, strictTestSet_X)


def main():
    ctrn_meta = ctrn_metadata()
    onevsrest(ctrn_meta)


main()