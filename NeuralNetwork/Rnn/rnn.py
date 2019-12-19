import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from scipy.signal import find_peaks 
import math as math

from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM
from keras.layers import GlobalMaxPooling1D
from keras.models import Model
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.layers import Input
from keras.layers.merge import Concatenate
from keras.layers import Bidirectional






scores = pd.read_csv("c:/Users/asuspc/Desktop/ML Project/RNN/stockfish.csv")

inputSize = len(scores["MoveScores"])

trainSetSize = 25000


data = list()

count = 0
for i in range(trainSetSize):
    try:
        data.append(list(map(int,scores["MoveScores"][i].split())))
    except:
        count += 1


improvementList = list()
for i in range(len(data)):
    blackImproves = 0
    whiteImproves = 0    
    for k in range(len(data[i]) - 1):
        if (data[i][k] - data[i][k+1]) < 0:
            blackImproves += 1
        if (data[i][k] - data[i][k+1]) > 0:
            whiteImproves += 1
    improvementList.append([whiteImproves,blackImproves, i + 1])

f = open("c:/Users/asuspc/Desktop/ML Project/RNN/data.pgn", "r")

elos = list()

whiteElo = ""
blackElo = ""
eventNo = ""

flag1 = 0
flag2 = 0
flag3 = 0

for line in f:
    if "WhiteElo" in line:
        flag1 = 1
        whiteElo = re.search('\"(.+?)\"', line).group(1)
    if "BlackElo" in line:
        flag2 = 1
        blackElo = re.search('\"(.+?)\"', line).group(1)
    if "Event" in line:
        flag3 = 1
        eventNo = re.search('\"(.+?)\"', line).group(1)
    if flag1 == 1 and flag2 == 1 and flag3 == 1:
        elos.append(list(map(int,[whiteElo, blackElo, eventNo])))
        flag1 = 0
        flag2 = 0
        flag3 = 0
    if len(elos) == 25000:
        break

f.close()



#########################################
################
#########################################

finalData = list()
for i in range(len(improvementList)):
    if improvementList[i][2] == elos[i][2]:
        finalData.append([improvementList[i][2], improvementList[i][0], improvementList[i][1], elos[i][0], elos[i][1]])
    else:
        k = i
        while True:
            if improvementList[i][2] == elos[k][2]:
                break
            k += 1
        finalData.append([improvementList[i][2], improvementList[i][0], improvementList[i][1], elos[k][0], elos[k][1]])


whiteElos = list()
blackElos = list()
for i in range(len(finalData)):
    whiteElos.append(finalData[i][0])
    blackElos.append(finalData[i][1])

whiteData = list()
blackData = list()
for i in range(len(data)):
    whiteData.append(data[i][0::2])
    blackData.append(data[i][1::2])

whiteBigestSize = 0
for i in range(len(whiteData)):
    if (len(whiteData[i]) > whiteBigestSize):
        whiteBigestSize = len(whiteData[i])

for i in range(len(whiteData)):
    if (len(whiteData[i]) < whiteBigestSize):
        for k in range(whiteBigestSize - len(whiteData[i])):
            whiteData[i].append(0)
whiteDataLength = len(whiteData)

blackBigestSize = 0
for i in range(len(blackData)):
    if (len(blackData[i]) > blackBigestSize):
        blackBigestSize = len(blackData[i])

for i in range(len(blackData)):
    if (len(blackData[i]) < blackBigestSize):
        for k in range(blackBigestSize - len(blackData[i])):
            blackData[i].append(0)
blackDataLength = len(blackData)

whiteData = np.array(whiteData).flatten()
whiteData = whiteData.reshape((whiteDataLength,whiteBigestSize,1)) 
blackData = np.array(blackData).flatten()
blackData = blackData.reshape((whiteDataLength ,blackBigestSize,1))
whiteElos = np.array(whiteElos).flatten()
blackElos = np.array(blackElos).flatten()

print(whiteData.shape)
print(blackElos.shape)



whiteTrain = whiteData[:int(len(whiteData)/2)]
whiteTrainElos = whiteElos[:int(len(whiteElos)/2)]


model = Sequential()
model.add(LSTM(2, activation='relu', input_shape=(165,1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

history = model.fit(whiteTrain, whiteTrainElos, epochs=10, validation_split=0.2, verbose=1)

test_input = whiteData[-1 * int(len(finalData)/2):]
test_output = model.predict(test_input, verbose=0)
print(test_output)

print('Mean Absolute Error for White Elo:', metrics.mean_absolute_error(whiteElos[-1 * int(len(whiteElos)/2):], test_output)) 
print(test_output)
