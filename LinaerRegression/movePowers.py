import numpy as np
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from scipy.signal import find_peaks 
import math as math

scores = pd.read_csv("stockfish.csv")

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

f = open("data.pgn", "r")

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


npFinal = np.array(finalData)
finalTrain = npFinal[:int(len(finalData)/2)]
finalTest = npFinal[-1 * int(len(finalData)/2):]


plt.scatter(finalTrain[:,3], finalTrain[:,1], s = 1)
plt.xlabel("White Elo")
plt.ylabel("Power Gain")
plt.savefig('Advantage Gain White Elo')
plt.show()

plt.scatter(finalTrain[:,4], finalTrain[:,2], s = 1)
plt.xlabel("Black Elo")
plt.ylabel("Power Gain")
plt.savefig('Advantage Gain Black Elo')
plt.show()

x = finalTrain[:,1].reshape((-1,1))
y = finalTrain[:,3].reshape((-1,1))

model = LinearRegression()
model.fit(x,y)

y_predict = model.predict(finalTest[:,1].reshape((-1,1)))

print('Advantage Gain => Mean Absolute Error for White Elo:', metrics.mean_absolute_error(finalTest[:,3], y_predict))  

x = finalTrain[:,2].reshape((-1,1))
y = finalTrain[:,4].reshape((-1,1))

model = LinearRegression()
model.fit(x,y)

y_predict = model.predict(finalTest[:,2].reshape((-1,1)))

print('Advantage Gain => Mean Absolute Error for Black Elo:', metrics.mean_absolute_error(finalTest[:,4], y_predict))

#######################################
######################
#######################################


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
    for k in range(len(data[i])):
        if (data[i][k]) < 0:
            blackImproves += 1
        if (data[i][k]) > 0:
            whiteImproves += 1
    improvementList.append([whiteImproves,blackImproves, i + 1])

f = open("data.pgn", "r")

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

npFinal = np.array(finalData)
finalTrain = npFinal[:int(len(finalData)/2)]
finalTest = npFinal[-1 * int(len(finalData)/2):]

plt.scatter(finalTrain[:,3], finalTrain[:,1], s = 1)
plt.xlabel("White Elo")
plt.ylabel("Domination")
plt.savefig('Domination White Elo')
plt.show()

plt.scatter(finalTrain[:,4], finalTrain[:,2], s = 1)
plt.xlabel("Black Elo")
plt.ylabel("Domination")
plt.savefig('Domination Black Elo')
plt.show()

x = finalTrain[:,1].reshape((-1,1))
y = finalTrain[:,3].reshape((-1,1))

model = LinearRegression()
model.fit(x,y)

y_predict = model.predict(finalTest[:,1].reshape((-1,1)))

print('Domination Time => Absolute Error for White Elo:', metrics.mean_absolute_error(finalTest[:,3], y_predict))  

x = finalTrain[:,2].reshape((-1,1))
y = finalTrain[:,4].reshape((-1,1))


model = LinearRegression()
model.fit(x,y)

y_predict = model.predict(finalTest[:,2].reshape((-1,1)))

print('Domination Time => Mean Absolute Error for Black Elo:', metrics.mean_absolute_error(finalTest[:,4], y_predict))

##########################
############
##########################

def getLongestPositiveSeq(a): 
    maxLen = 0
    currLen = 0
    for k in range(len(a)): 
        if a[k] > 0: 
            currLen +=1
            if currLen > maxLen:
                maxLen = currLen
        else: 
            if currLen > maxLen: 
                maxLen = currLen 
            currLen = 0
    if maxLen > 0: 
        return maxLen
    else: 
        return 0 

def getLongestNegativeSeq(a): 
    maxLen = 0
    currLen = 0
    for k in range(len(a)): 
        if a[k] < 0: 
            currLen +=1
            if currLen > maxLen:
                maxLen = currLen
        else: 
            if currLen > maxLen: 
                maxLen = currLen 
            currLen = 0
    if maxLen > 0: 
        return maxLen
    else: 
        return 0 



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
    blackImproves = getLongestNegativeSeq(data[i])
    whiteImproves = getLongestPositiveSeq(data[i])
    improvementList.append([whiteImproves,blackImproves, i + 1])

f = open("data.pgn", "r")

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

npFinal = np.array(finalData)
finalTrain = npFinal[:int(len(finalData)/2)]
finalTest = npFinal[-1 * int(len(finalData)/2):]

plt.scatter(finalTrain[:,3], finalTrain[:,1], s = 1)
plt.xlabel("White Elo")
plt.ylabel("Advantage Maintain")
plt.savefig('Advantage Maintain White Elo')
plt.show()

plt.scatter(finalTrain[:,4], finalTrain[:,2], s = 1)
plt.xlabel("Black Elo")
plt.ylabel("Advantage Maintain")
plt.savefig('Advantage Maintain Black Elo')
plt.show()

x = finalTrain[:,1].reshape((-1,1))
y = finalTrain[:,3].reshape((-1,1))

model = LinearRegression()
model.fit(x,y)

y_predict = model.predict(finalTest[:,1].reshape((-1,1)))

print('Advantage Maintain Feature => Mean Absolute Error for White Elo:', metrics.mean_absolute_error(finalTest[:,3], y_predict))  

x = finalTrain[:,2].reshape((-1,1))
y = finalTrain[:,4].reshape((-1,1))

model = LinearRegression()
model.fit(x,y)

y_predict = model.predict(finalTest[:,2].reshape((-1,1)))

print('Advantage Maintain Feature =>Mean Absolute Error for Black Elo:', metrics.mean_absolute_error(finalTest[:,4], y_predict)) 

##################################
#################
##################################

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
    blackMistake = 0
    whiteMistake = 0    
    peaks , _ = find_peaks(data[i])
    peaks = list(peaks)
    for k in range(len(peaks)):
        if (peaks[k]) % 2 == 0:
            whiteMistake += 1
        if (peaks[k]) % 2 == 1:
            blackMistake += 1
    improvementList.append([whiteMistake,blackMistake, i + 1])

f = open("data.pgn", "r")

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

finalData = list()

for i in range(len(improvementList)):
    if improvementList[i][2] == elos[i][2]:
        finalData.append([improvementList[i][2], improvementList[i][0] / (abs(elos[i][0] - elos[i][1]) + 1), improvementList[i][1] / (abs(elos[i][0] - elos[i][1]) + 1) , elos[i][0], elos[i][1]])
    else:
        k = i
        while True:
            if improvementList[i][2] == elos[k][2]:
                break
            k += 1
        finalData.append([improvementList[i][2], improvementList[i][0] / (abs(elos[k][0] - elos[k][1]) + 1), improvementList[i][1] / (abs(elos[k][0] - elos[k][1]) + 1), elos[k][0], elos[k][1]])

npFinal = np.array(finalData)
finalTrain = npFinal[:int(len(finalData)/2)]
finalTest = npFinal[-1 * int(len(finalData)/2):]

plt.scatter(finalTrain[:,3], finalTrain[:,1], s = 1)
plt.xlabel("White Elo")
plt.ylabel("Missing Capitalization Opportunity")
plt.savefig('Missing Capitalization Opportunity White Elo')
plt.show()

plt.scatter(finalTrain[:,4], finalTrain[:,2], s = 1)
plt.xlabel("Black Elo")
plt.ylabel("Missing Capitalization Opportunity")
plt.savefig('Missing Capitalization Opportunity Black Elo')
plt.show()

x = finalTrain[:,1].reshape((-1,1))
y = finalTrain[:,3].reshape((-1,1))

model = LinearRegression()
model.fit(x,y)

y_predict = model.predict(finalTest[:,1].reshape((-1,1)))

print('Missing Capitalization Opportunity feature => Mean Absolute Error for White Elo:', metrics.mean_absolute_error(finalTest[:,3], y_predict))  


x = finalTrain[:,2].reshape((-1,1))
y = finalTrain[:,4].reshape((-1,1))

model = LinearRegression()
model.fit(x,y)

y_predict = model.predict(finalTest[:,2].reshape((-1,1)))

print('Missing Capitalization Opportunity feature => Mean Absolute Error for Black Elo:', metrics.mean_absolute_error(finalTest[:,4], y_predict))
