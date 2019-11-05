from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Flatten
from sklearn.model_selection import train_test_split
from keras import backend as K
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

    all_trans_embs = {}
    for transcript in sessToUttDict:
        emb = get_3rd_layer_output([sessToUttDict[transcript]])[0]
        all_trans_embs[transcript] = emb

    empathyScoreVec = getEmpathyScore(sessToUttDict, sessions)

    # Match them up
    trans_embs_X = []
    empathy_score_Y = []
    for transcript in all_trans_embs:
        if transcript in empathyScoreVec:
            trans_embs_X.append(all_trans_embs[transcript])
            empathy_score_Y.append(empathyScoreVec[transcript])
    # Done matching

    pad_len = max([x.shape[0] for x in trans_embs_X])

    all_X = []
    for x in trans_embs_X:
        add_pad = pad_len - x.shape[0]
        pad_data = np.zeros((add_pad, x.shape[1]))
        x = np.concatenate([x, pad_data])
        all_X.append(x)

    all_X = np.array(all_X)

    X_train, X_test, y_train, y_test = train_test_split(all_X,
            np.array(empathy_score_Y), test_size=0.33)

    empathyModel = Sequential()
    empathyModel.add(AveragePooling1D(pool_size = 64))
    #empathyModel.add(LSTM(10, dropout=0.2, recurrent_dropout=0.2,
    #    input_shape=(None, 40)))
    empathyModel.add(Flatten())
    empathyModel.add(Dense(16, activation='relu'))
    empathyModel.add(Dense(1, activation='softmax'))

    # Compile model
    empathyModel.compile(loss='mse', optimizer='adam')

    # Fit the model
    history = empathyModel.fit(X_train, y_train, epochs=5, batch_size=10,
            verbose=2)
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