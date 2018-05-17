import csv
textFileName = 'C:/Users/Boltak/Desktop/genpsych/General_psychtx_corpus_phase1.1'

class_clusters = {'6' : ['depression emotion', 'anger', 'mania', 'sadness', 'confusion', 'paranoia', 'panic', 'apathy', 'restlessness', 'despair', 'guilt', 'resentment', 'indecisiveness'], 
                '3' : ['inattentiveness', 'suicidal ideation', 'fantasizing', 'obsessive behavior', 'isolation', 'racing thoughts', 'withdrawn', 'dreams', 'social inhibition', 'problems concentrating', 'compulsive behavior', 'suicidal behavior', 'severe sensitivity', 'cutting', 'aggression', 'acting out', 'danger to others', 'academic failure', 'detached behavior', 'disorganized thoughts', 'danger to self', 'rash', 'avoidance', 'deceitfulness'],
                '2' : ['insomnia'],
                '0' : ['fearfulness', 'suspiciousness'],
                '7' : ['hallucinations', 'dysphoria', 'hypersomnia', 'hyperphagia', 'enuresis', 'anhedonia', 'dysphagia', 'vomiting'],
                '1' : ['fatigue', 'chronic pain', 'crying', 'loss of appetite', 'delusions', 'general pain', 'tremors', 'nausea', 'headache', 'back pain', 'withdrawal sickness', 'sweating', 'fainting', 'itching', 'stuttering', 'seizures', 'phantom pain']}

def ctrn_metadata(csvfile,min_count=50):
    csvfile = open(csvfile, 'rt')
    csvreader = csv.reader(csvfile, delimiter=',')
    i = 0
    data = {}
    symptoms_type = {}
    for row in csvreader:
        if i != 0:
            try:
                fileID = int(float(row[2]))
                data[fileID] = {}
                summary = row[6].split(":")[1]
                symptoms = row[22]
                data[fileID]['summary'] = summary
                data[fileID]['symptoms'] = symptoms
                data[fileID]['cluster_symp'] = [0] * 8

                symptoms = symptoms.split("; ")
                for symptom in symptoms:

                    if symptom != '':
                        print(symptom)
                        symptom = symptom2cluster(symptom=symptom)
                        print(symptom)
                        if symptom not in symptoms_type.keys():
                            symptoms_type[symptom] = 1
                        else:
                            symptoms_type[symptom] = symptoms_type[symptom] + 1
#                        symptoms_type.append(symptom)
#                print(fileID,"::",summary,"::",symptoms)
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

def corpus_dic(csvfile):

    csvfile = open(csvfile, 'rt')
    csvreader = csv.reader(csvfile, delimiter=',')

    corpus = {}
    i = 0
    for row in csvreader:
        if i != 0:
#            print(row[1])
            fileID = int(row[1])
            if fileID not in corpus.keys():
                corpus[fileID] = []
            corpus[fileID].append(row[-2]+"::"+row[-1])

        i = i + 1

    return corpus

def clusterVectCreation():
    #iterate through data.keys which will be fileIDs
    #access symptom vector
    #for each symptom, iterate through classcluster.keys
    #iterate through classcluster.values
    #if symptom is found in that cluster, increment data[fileID]['cluster_symp'][cluster#] by one
    
def onevsrest(ctrn_meta):

    count_vect = TfidfVectorizer()
    dataX = []
    dataY = []

    for fileID in ctrn_meta.keys():
#        if len(ctrn_meta[fileID].keys()) == 2:
#        print(ctrn_meta[fileID]['summary'], ctrn_meta[fileID]['symptoms'])
        dataX.append(ctrn_meta[fileID]['summary'])
        dataY.append(ctrn_meta[fileID]['symptoms'].split("; "))

    count_vect.fit(dataX)
    count_vect_X = count_vect.transform(dataX)

    mlb = MultiLabelBinarizer()
    count_vect_Y = mlb.fit_transform(dataY)
    print(mlb.classes_)

    X_train, X_test, y_train, y_test = train_test_split(count_vect_X, count_vect_Y, test_size=0.33, random_state=42)

    OneVsRestClassifier(LinearSVC(random_state=0)).fit(X_train, y_train).predict(X_test).tolist()[20:40])

def main():
    with open('General_psychtx_corpus_phase1.1.csv', 'rt') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            i = i + 1
            if i == 10:
                break
            else:
                print(" ,".join(row))
        ctrn_meta= = {}
        ctrn_meta = ctrn_metadata(csvfile)
        
    for fileID in ctrn_meta.keys():
        dataX.append(ctrn_meta[fileID][‘summary’])
        
    count_vect = TfidfVectorizer()
    count_vect.fit(dataX)
    count_vect_X = count_vect.transform(dataX)
        
        
        
        
        