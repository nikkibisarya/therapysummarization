import numpy as np
import re
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras import backend as K
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from keras.layers import LSTM
from keras.layers import Embedding

filePath = r'/home/nikki/2012-10-20-coding_data-all.txt'
word2VecPath = r'/home/nikki/vec.txt'
def normalize_code(code):
    #    print(code,spk)
    code = re.sub(r"(.*)([+-]).*", r"\1\2", code).upper()
    if code in ["FA", "GI", "RES", "QUC", "QUO", "REC"]:
        pass
    elif code in ["ADW", "CO", "DI", "RCW", "WA"]:
        code = "NA"
    else:
        code = "COU"
    return code

# make a dictionary of file:utterance, label
def make_dictionary(open_path):
    transcriptToUtterances = {}
    FAcount = 0
    GIcount = 0
    REScount = 0
    QUCcount = 0
    QUOcount = 0
    RECcount = 0
    NAcount = 0
    COUcount = 0
    otherCount = 0
    with open(open_path, 'r') as file:
        for line in file:
            values = line.split('\t')
            if len(values) == 10:
                if values[6] == 'T':
                    result = values[-1]
                    label = str(values[5])
                    label_code = str(normalize_code(label))
                    if label_code == "FA":
                        label_code = [1, 0, 0, 0, 0, 0, 0, 0]
                        FAcount = FAcount + 1
                    elif label_code == "GI":
                        label_code = [0, 1, 0, 0, 0, 0, 0, 0]
                        GIcount = GIcount + 1
                    elif label_code == "RES":
                        label_code = [0, 0, 1, 0, 0, 0, 0, 0]
                        REScount = REScount + 1
                    elif label_code == "QUC":
                        label_code = [0, 0, 0, 1, 0, 0, 0, 0]
                        QUCcount = QUCcount + 1
                    elif label_code == "QUO":
                        label_code = [0, 0, 0, 0, 1, 0, 0, 0]
                        QUOcount = QUOcount + 1
                    elif label_code == "REC":
                        label_code = [0, 0, 0, 0, 0, 1, 0, 0]
                        RECcount = RECcount + 1
                    elif label_code == "NA":
                        label_code = [0, 0, 0, 0, 0, 0, 1, 0]
                        NAcount = NAcount + 1
                    elif label_code == "COU":
                        label_code = [0, 0, 0, 0, 0, 0, 0, 1]
                        COUcount = COUcount + 1
                    else:
                        otherCount = otherCount + 1
                    add_pair = [result, label_code]
                    key = values[0]
                    if key in transcriptToUtterances:
                        transcriptToUtterances[key].append(add_pair)
                    else:
                        transcriptToUtterances[key] = [add_pair]
    print("Other: ", otherCount, "FA: ", FAcount, 'COU: ', COUcount, "GI: ", GIcount, "RES: ", REScount, "REC: ", RECcount, "QUO: ", QUOcount, "QUC: ", QUCcount, "NA: ", NAcount)
    return transcriptToUtterances


def processWordVectors():
    with open(word2VecPath, 'r') as file:
        word2vecDict = {}
        word = ''
        for line in file:
            for value in line.split():
                try:
                    if word in word2vecDict:
                        word2vecDict[word].append(float(value))
                    else:
                        word2vecDict[word] = [float(value)]
                except Exception:
                    word = str(value)
    return word2vecDict


# I am making a dictionary of file:average vector of an utterance, label
def word2vec(transcriptToUtterances):
    word2vecDict = processWordVectors()
    file2AvgVecPerUtterance = {}

    for key in transcriptToUtterances:
        for utterance, label in transcriptToUtterances[key]:
            listOfWordVecs = []
            for word in utterance.split(' '):
                if word in word2vecDict:
                    listOfWordVecs.append(word2vecDict[word])
            if len(listOfWordVecs) == 0:
                continue
            listOfWordVecs = np.array(listOfWordVecs)

            avgVec = np.mean(listOfWordVecs, axis=0)

            add_pair = [avgVec, label]

            if key in file2AvgVecPerUtterance:
                file2AvgVecPerUtterance[key].append(add_pair)
            else:
                file2AvgVecPerUtterance[key] = [add_pair]

    return file2AvgVecPerUtterance


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

utters = make_dictionary(filePath)
trans_to_utters = word2vec(utters)
np.random.seed(7)
# load data
X, Y = collectUtterances(trans_to_utters)
print('length of Y: ', )
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

# create model
model = Sequential()
model.add(Embedding(11864, 100, input_length=20))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(60, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(8, activation='softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
history = model.fit(X_train, y_train, epochs=20, batch_size=10)
model.predict(X_train[0])
# evaluate the model
scores = model.evaluate(X_test, y_test)
# list all data in history
argmaxModel = model.predict(X_test)
argmaxModel2 = np.argmax(argmaxModel, axis=-1)
argmaxModelYTEST = np.argmax(y_test, axis=-1)
a = np.array(argmaxModel2)
b = np.zeros((23214, 8))
b[np.arange(23214), a] = 1
c = np.array(argmaxModelYTEST)
d = np.zeros((23214, 8))
d[np.arange(23214), c] = 1
print('train loss', history.history['loss'][-1])
print('test loss', scores[0])
print(classification_report(b, d))

raise Exception
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('accuracy.png')
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('loss.png')
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print(model.summary())
model.save('/home/nikki/utteranceNN.weights')
plot_model(model, to_file='model.png')