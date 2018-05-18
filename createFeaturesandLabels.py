import csv
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import re
from sklearn.metrics import recall_score
textFileName = 'C:/Users/Boltak/Desktop/genpsych/General_psychtx_corpus_phase1.1.csv'

class_clusters = {'6' : ['depression emotion', 'anger', 'mania', 'sadness', 'confusion', 'paranoia', 'panic', 'apathy', 'restlessness', 'despair', 'guilt', 'resentment', 'indecisiveness'], 
                '3' : ['inattentiveness', 'suicidal ideation', 'fantasizing', 'obsessive behavior', 'isolation', 'racing thoughts', 'withdrawn', 'dreams', 'social inhibition', 'problems concentrating', 'compulsive behavior', 'suicidal behavior', 'severe sensitivity', 'cutting', 'aggression', 'acting out', 'danger to others', 'academic failure', 'detached behavior', 'disorganized thoughts', 'danger to self', 'rash', 'avoidance', 'deceitfulness'],
                '2' : ['insomnia'],
                '0' : ['fearfulness', 'suspiciousness'],
                '7' : ['hallucinations', 'dysphoria', 'hypersomnia', 'hyperphagia', 'enuresis', 'anhedonia', 'dysphagia', 'vomiting'],
                '1' : ['fatigue', 'chronic pain', 'crying', 'loss of appetite', 'delusions', 'general pain', 'tremors', 'nausea', 'headache', 'back pain', 'withdrawal sickness', 'sweating', 'fainting', 'itching', 'stuttering', 'seizures', 'phantom pain']}

def ctrn_metadata():
    csvfile = open('C:/Users/Boltak/Desktop/genpsych/General_psychtx_corpus_phase1.1.csv', 'r')
    csvreader = csv.reader(csvfile, delimiter=',')
    i = 0
    data = {}
    for row in csvreader:
        if i != 0:
            try:
                fileID = int(float(row[2]))
                data[fileID] = {}
                summary = row[6].split(":")[1]
                symptoms = row[22]
                data[fileID]['summary'] = summary
                [x.lower() for x in symptoms]
                data[fileID]['symptoms'] = symptoms
                data[fileID]['cluster_symp'] = [0] * 8
                
            except:
                continue
        i = i+1
    return data

def symptom2cluster(symptom, class_clusters=class_clusters):

    cluster = ''
    symptom = symptom.strip().lower()
    symptom = re.sub(r'[^\w\s]','',symptom)
    for key in class_clusters.keys():
        if symptom in class_clusters[key]:
            cluster = key
#            break

    return cluster

def clusterVectCreation(data):
    clusterIndex = -1
    #iterate through data.keys which will be fileIDs
    for fileID in data.keys():
        for symptom in data[fileID]['symptoms']:
            clusterIndex = symptom2cluster(symptom)
            data[fileID]['cluster_symp'][clusterIndex] = data[fileID]['cluster_symp'][clusterIndex] + 1
            
#             for num in class_clusters.keys():
#                 for value in class_clusters[num]:
#                     if symptom == value:
#                         clusterIndex = int(num)
#                         data[fileID]['cluster_symp'][clusterIndex] = data[fileID]['cluster_symp'][clusterIndex] + 1
#                         
                
    #access symptom vector
    #for each symptom, iterate through classcluster.keys
    #iterate through classcluster.values
    #if symptom is found in that cluster, increment data[fileID]['cluster_symp'][cluster#] by one
    #these will serve as labels
    
def onevsrest(ctrn_meta):
    count_vect = TfidfVectorizer()
    dataX = []
    dataY = []
    for fileID in ctrn_meta.keys():
        dataX.append(ctrn_meta[fileID]['summary'])
        dataY.append(ctrn_meta[fileID]['cluster_symp'])
    print('data x: ', dataX)
    count_vect.fit(dataX)
    count_vect_X = count_vect.transform(dataX)
    count_vect_Y = dataY

    X_train, X_test, y_train, y_test = train_test_split(count_vect_X, count_vect_Y, test_size=0.33, random_state=42)

    pred_y = []
    pred_y = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X_train, y_train).predict(X_test)
    print('Recall Score: ', recall_score(y_test, pred_y))
    
def main():
    with open(textFileName, 'r') as csvfile:
        ctrn_meta = {}
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            i = 0
            i = i + 1
            if i == 10:
                break
            else:
                continue
        ctrn_meta = ctrn_metadata()
        print('dict: ', ctrn_meta)
        clusterVectCreation(ctrn_meta)
        onevsrest(ctrn_meta)

main()
    

        
        
        
        
        