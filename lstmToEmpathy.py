from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.layers import GlobalAveragePooling1D
from keras.layers import MaxPooling1D
from keras.layers import AveragePooling1D
import numpy as np
import matplotlib.pyplot as plt

empFilePath = '/home/nikki/clean_transcripts/'
fileName = 'transcript_splits_data_new.txt'
uttFilePath = r'/home/nikki/2012-10-20-coding_data-all.txt'

#convert each utterance into a vector of indices (add zeros if the vector is less than 50 words, otherwise cut it off)
def fileToUtteranceAndIndices(open_path, wordIndexDict):
    sessToUttDict = {}
    with open(open_path, 'r') as file:
        for line in file:
            values = line.split('\t')
            if len(values) == 10:
                if values[6] == 'T':
                    result = values[-1]
                    vec = []
                    for word in result.split(' '):
                        vec.append(wordIndexDict[word])
                    if len(vec) < 50:
                        zeroes = 50 - len(vec)
                        for zero in range(zeroes):
                            vec.append(0)
                    elif len(vec) > 50:
                        vec = vec[0:50]
                    else:
                        vec = vec
                    # append this vec to the transcript name so I can make a dict {session:utterances}
                    key = values[0]
                    if values[0] in sessToUttDict:
                        sessToUttDict[key].append(vec)
                    else:
                        sessToUttDict[key] = [vec]
    return sessToUttDict

def getEmpathyScore(sessToUttDict, sessions):
    empathyScoreVec = {}
    for transcript in sessToUttDict:
        if transcript in sessions:
            empathyScoreVec[transcript] = sessions[transcript]['empathy']
            print('emp in method', sessions[transcript]['empathy'])
    return empathyScoreVec

#assign every word an index
def wordToIndex(open_path):
    with open(open_path, 'r') as file:
        count = 0
        wordIndexDict = {}
        for line in file:
            values = line.split('\t')
            if len(values) == 10:
                if values[6] == 'T':
                    result = values[-1]
                    for word in result.split(' '):
                        if word in wordIndexDict:
                            continue
                        else:
                            count = count + 1
                            wordIndexDict[word] = count
    return wordIndexDict

def collectUtterances(file2AvgVecPerUtterance):
    listOfAvgVectors = []
    listOfLabels = []
    for key in file2AvgVecPerUtterance:
        for pair in file2AvgVecPerUtterance[key]:
            listOfAvgVectors.append(pair[0])
            listOfLabels.append(pair[1])
    listOfAvgVectors = np.array(listOfAvgVectors)
    listOfLabels = np.array(listOfLabels)
    return listOfAvgVectors, listOfLabels



def main():
    header = []
    data = []

    with open(empFilePath + fileName, 'r') as file:
        for line in file:
            if len(header) == 0:
                header = line.strip().split(',')
            else:
                cur_line = line.strip().split(',')
                dictionary = dict(zip(header, cur_line))
                data.append(dictionary)

    sessions = {}
    textFileNames = []

    for cur_data in data:
        session_filepath = empFilePath + cur_data['session'] + '.txt'
        textFileNames.append(session_filepath)
        sessions[cur_data['session']] = {
            'session_filepath': session_filepath,
            'empathy': float(cur_data['empathy'])
        }

    utt_t_model = load_model('my_model.h5')

    # with a Sequential model
    get_3rd_layer_output = K.function([utt_t_model.layers[0].input],
                                      [utt_t_model.layers[3].output])
    wordIndexDict = wordToIndex(uttFilePath)
    sessToUttDict = fileToUtteranceAndIndices(uttFilePath, wordIndexDict)
    empathyScoreVec = getEmpathyScore(sessToUttDict, sessions)

    empathy_score_Y = []
    utt_X = []
    for t in empathyScoreVec:
        utt_X.append(sessToUttDict[t])
        empathy_score_Y.append(empathyScoreVec[t])

    X_train, X_test, y_train, y_test = train_test_split(
            np.array(utt_X),
            np.array(empathy_score_Y), test_size=0.33, random_state=42)

    empathyModel = Sequential()
    empathyModel.add(Embedding(11864, 100, input_length=20))
    empathyModel.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    empathyModel.add(Dense(60, activation='relu'))
    empathyModel.add(Dense(40, activation='relu'))
    empathyModel.add(Dense(1, activation='softmax'))

    # Compile model
    empathyModel.compile(loss='mse', optimizer='adam')
    # Fit the model
    history = empathyModel.fit(X_train, y_train, epochs=20, batch_size=10)
    print('history', history.history)
    # evaluate the model
    empathyModel.evaluate(X_test, y_test)

    # raise Exception
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('loss.png')

main()
