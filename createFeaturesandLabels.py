import csv
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import re
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import numpy as np

textFileName = 'C:/Users/Boltak/Desktop/genpsych/meta.data.4.03.13.csv'

class_clusters = {'6' : ['depression emotion', 'anger', 'mania', 'sadness', 'confusion', 'paranoia', 'panic', 'apathy', 'restlessness', 'despair', 'guilt', 'resentment', 'indecisiveness'], 
                '3' : ['inattentiveness', 'suicidal ideation', 'fantasizing', 'obsessive behavior', 'isolation', 'racing thoughts', 'withdrawn', 'dreams', 'social inhibition', 'problems concentrating', 'compulsive behavior', 'suicidal behavior', 'severe sensitivity', 'cutting', 'aggression', 'acting out', 'danger to others', 'academic failure', 'detached behavior', 'disorganized thoughts', 'danger to self', 'rash', 'avoidance', 'deceitfulness'],
                '2' : ['insomnia'],
                '0' : ['fearfulness', 'suspiciousness'],
                '7' : ['hallucinations', 'dysphoria', 'hypersomnia', 'hyperphagia', 'enuresis', 'anhedonia', 'dysphagia', 'vomiting'],
                '1' : ['fatigue', 'chronic pain', 'crying', 'loss of appetite', 'delusions', 'general pain', 'tremors', 'nausea', 'headache', 'back pain', 'withdrawal sickness', 'sweating', 'fainting', 'itching', 'stuttering', 'seizures', 'phantom pain']}

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

def ctrn_metadata():
    csvfile = open('C:/Users/Boltak/Desktop/Usable Metadata.csv', 'r', encoding="latin-1")
    csvreader = csv.reader(csvfile, delimiter=',')
    data = {}
    i = 0
    count = 0
    symcount = 0
    num = 0
    no_Summary = 0
    validCount = 0
    summaryCount = 0
    corpus = corpus_dic('C:/Users/Boltak/Desktop/genpsych/General_psychtx_corpus_phase1.1.csv')
    for row in csvreader:
        if i != 0:
            if row[4] != '':
                getID = row[4].split('>')
                if getID[0] == 'NA':
                    count += 1
                if getID[0] != 'NA':
                    #some fileIDs don't have >
                    if len(getID) == 1:
                        fileID = str(getID[0])
                    else:
                        fileID = str(getID[1])
                    if fileID in corpus:
                        num += 1
                        data[fileID] = {}
                        data[fileID]['valid transcript'] = True
                        if ':' in row[5]:
                            summary = row[5].split(":")[1]
                            data[fileID]['summary'] = summary
                            summaryCount += 1
                        else:
                            no_Summary += 1
                        if row[21] == 'NA':
                            symcount += 1
                            data[fileID]['valid transcript'] = False
                        else:
                            symptoms = row[21].split(';')
                        if data[fileID]['valid transcript'] == True:
                            validCount += 1
                        data[fileID]['symptoms'] = symptoms
                        data[fileID]['cluster_symp'] = [0] * 8
                        data[fileID]['transcript'] = corpus[fileID]
                        data[fileID]['label_num'] = -1
                        data[fileID]['cluster_presence'] = []
        i = i+1
    print('How many transcripts we have?: ', num)
    print('How many summaries available?: ', summaryCount)
    print('How many files dont have a summary? ' , no_Summary)
    print('Transcripts with no symptoms: ', symcount)
    print('Transcripts with no file ID: ', count)
    print('Overall, number of valid transcripts?: ', validCount)
    
    
    return data

def symptom2cluster(symptom, class_clusters=class_clusters):
    cluster = ''
    symptom = symptom.strip().lower()
    symptom = re.sub(r'[^\w\s]','',symptom)
    for key in class_clusters.keys():
        if symptom in class_clusters[key]:
            cluster = int(key)
            break
    return cluster

def clusterPresence(data):
    for fileID in data.keys():
        for symp in data[fileID]['symptoms']:
            num = symptom2cluster(symp)
            if str(num) != '':
                if str(num) in data[fileID]['cluster_presence']:
                    continue
                else:
                    data[fileID]['cluster_presence'].append(str(num))
    
def clusterVectCreation(data):
    clusterIndex = -1
    for fileID in data.keys():
        for symptom in data[fileID]['symptoms']:
            clusterIndex = symptom2cluster(symptom)
            if clusterIndex == '':
                data[fileID]['valid transcript'] = False
            else:
                data[fileID]['cluster_symp'][clusterIndex] = 1
    
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

def replaceLabelVects(labels, data):
    for fileID in data.keys():
        data[fileID]['label_num'] = labels[str(data[fileID]['cluster_symp'])]
        
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

def predictor(count_vect_X, count_vect_Y):
    X_train, X_test, y_train, y_test = train_test_split(count_vect_X, count_vect_Y, test_size=0.33, random_state=42)
    print('length: ', len(y_train))
    print('length: ', len(y_test))
    ovr = OneVsRestClassifier(LinearSVC(random_state=0))
    ovr.fit(X_train, y_train)
    pred_y = ovr.predict(X_test)
    countPred(pred_y, y_train)
    print('F1 score avg = MICRO: ', f1_score(y_test, pred_y, average='micro', labels=np.unique(pred_y)))
    print('F1 score avg = NONE: ', f1_score(y_test, pred_y, average=None, labels=np.unique(pred_y)))
    recall = recall_score(y_test, pred_y, average = None)
    print('Recall: ', recall, '\n')
    return pred_y

def onevsrest(ctrn_meta):
    count_vect = TfidfVectorizer()
    dataX = []
    dataY = []
    dataZ = []
    count = 0
    clusterPresence(ctrn_meta)
    for fileID in ctrn_meta.keys():
        print('fileID: ', fileID)
        print('summary: ', fileID, ctrn_meta[fileID]['summary'])
        if ctrn_meta[fileID]['valid transcript'] == True:
            count += 1
            dataX.append(ctrn_meta[fileID]['transcript'])
            dataY.append(ctrn_meta[fileID]['cluster_presence'])
           # dataZ.append(ctrn_meta[fileID]['summary'])
    count_vect.fit(dataX)
    count_vect_X = count_vect.transform(dataX)
    mlb = MultiLabelBinarizer()
    count_vect_Y = mlb.fit_transform(dataY)
    print('total number of transcripts: ', count)
    print('Using Transcripts:')
    predictor(count_vect_X, count_vect_Y)
    #count_vect_Z = count_vect.transform(dataZ)
    print('Using Summaries:')
    print('count: ', count)
   # predictor(count_vect_Z, count_vect_Y)

def main():
   # f = open('recall.txt','w')
    ctrn_meta = ctrn_metadata()
    onevsrest(ctrn_meta)
   # f.write('Recall: ' + str(recall))
   # f.close()
main()


        
        
        
        
        