# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 20:04:15 2019

@author: fahad
"""
import chess.pgn 
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk
import pgn
from keras.preprocessing.sequence import pad_sequences
from sklearn.linear_model import LinearRegression, LogisticRegression
import xgboost as xgb
##import tensorflow
#
#
#from mord import LogisticAT
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet 
from sklearn.metrics import mean_absolute_error, mean_squared_error

move_quality = pd.read_csv('stockfish.csv')
#elo = pd.read_csv('data.pgn', header=None)
##elo = np.genfromtxt('data.pgn')
#data1 = np.array(list(map(int,move_quality['MoveScores'][55].split())))
#data1 = np.array(float(move_quality['MoveScores'][0].split()))
#first_game = chess.pgn.read_game(data)
#second_game = chess.pgn.read_game(data)
#print(first_game)
#print(second_game)
#board = first_game.board()
#for move in first_game.mainline_moves():
#    board.push(move)
#board
#board('4r3/6P1/2p2P1k/1p6/pP2p1R1/P1B5/2P2K2/3r4 b - - 0 45')
#print(first_game, file=open("/dev/null", "w"), end="\n\n")


data = open('data.pgn')
games = []
for i in range(25000):
    games.append(chess.pgn.read_game(data))

#games[4].headers["WhiteElo"]
    
    
WhiteElos_train = []
BlackElos_train = []
for i in range (20000):
    WhiteElos_train.append(int(games[i].headers["WhiteElo"]))
    BlackElos_train.append(int(games[i].headers["BlackElo"]))
WhiteElos_train = np.array(WhiteElos_train)
BlackElos_train = np.array(BlackElos_train)

WhiteElos_test = []
BlackElos_test = []
for i in range(20000,25000):
    WhiteElos_test.append(int(games[i].headers["WhiteElo"]))
    BlackElos_test.append(int(games[i].headers["BlackElo"]))
WhiteElos_test = np.array(WhiteElos_test)
BlackElos_test = np.array(BlackElos_test)  
#abc = np.array((WhiteElos))
#print(abc.loc(2200))




wtrain_features = np.zeros(20000)
btrain_features = np.zeros(20000)
j = 0
for i in range(20000):
    t = move_quality['MoveScores'][i].split()
    if len(t) > 0:
        train_features = [x for x in t if x != 'NA']
        train_features = np.array(list(map(int,train_features)))
        length = len(train_features)
        move_strength = np.ones(length)
        move_strength[0] = train_features[0]
        move_strength[1::] = train_features[1::] - train_features[0:length-1]
        white_moves = move_strength[move_strength > 0]
        black_moves = move_strength[move_strength < 0]
        wtrain_features[i-j] = len(white_moves) * np.sum(white_moves) / length
        btrain_features[i-j] = len(black_moves) * np.sum(black_moves) / length
#        wtrain_features[i-j] = len(white_moves)  / length
#        btrain_features[i-j] = len(black_moves) / length
    else:
        BlackElos_train = np.delete(BlackElos_train, i)
        WhiteElos_train = np.delete(WhiteElos_train, i)
        j += 1 
wtrain_features = wtrain_features[:i-j+1]
btrain_features = btrain_features[:i-j+1]     
        
        
wtest_features = np.zeros(5000)
btest_features = np.zeros(5000)  
j = 0      
for i in range(20000,25000):
    t = move_quality['MoveScores'][i].split()
    if len(t) > 0:
        test_features = [x for x in t if x != 'NA']
        test_features = np.array(list(map(int,test_features)))
        length = len(test_features)
        move_strength = np.ones(length)
        move_strength[0] = test_features[0]
        move_strength[1::] = test_features[1::] - test_features[0:length-1]
        white_moves = move_strength[move_strength > 0]
        black_moves = move_strength[move_strength < 0]
        wtest_features[i-20000-j] = len(white_moves) * np.sum(white_moves) / length
        btest_features[i-20000-j] = len(black_moves) * np.sum(black_moves) / length
#        wtest_features[i-20000-j] = len(white_moves)  / length
#        btest_features[i-20000-j] = len(black_moves)  / length
    else:
        j +=1
        BlackElos_test = np.delete(BlackElos_test, i-20000)
        WhiteElos_test = np.delete(WhiteElos_test, i-20000)
       
wtest_features = wtest_features[:i-j+1-20000]
btest_features = btest_features[:i-j+1-20000]


#Linear Regression
from sklearn import linear_model, datasets
lm =linear_model.LinearRegression()

#one feature
modelw = lm.fit(wtrain_features.reshape(-1, 1), WhiteElos_train.reshape(-1, 1))
modelb = lm.fit(btrain_features.reshape(-1, 1), BlackElos_train.reshape(-1, 1))
predictionw = lm.predict(wtest_features.reshape(-1, 1))
predictionb = lm.predict(btest_features.reshape(-1, 1)) 

#evaluate one feature linear regression
from sklearn import metrics 
print(metrics.mean_absolute_error(WhiteElos_test, predictionw))
print(metrics.mean_absolute_error(BlackElos_test,predictionb))

plt.plot(wtest_features, predictionw, color = 'blue', linewidth = 3)
plt.scatter(wtest_features, WhiteElos_test, color = 'black')

train_features = list()
for i in range(20000):
    t = move_quality['MoveScores'][i].split()
    if len(t) > 0:
        t = [x for x in t if x != 'NA']
        train_features.append(list(map(int,t)))
padded_train = pad_sequences(train_features, maxlen=120, padding = 'post') 

test_features = list()
for i in range(20000,25000):
    t = move_quality['MoveScores'][i].split()
    if len(t) > 0:
        t = [x for x in t if x != 'NA']
        test_features.append(list(map(int,t)))
padded_test = pad_sequences(test_features, maxlen = 120, padding = 'post') 
train_features2 = list()
test_features2 = list()
train_features2 = train_features
test_features2 = test_features
#for i in range(np.size(train_features)):
#    t = train_features[i]
#    if np.size(t) > 120:
#        train_features[i] = t[:120]
#    if np.size(t) < 120:
#        a = np.ones(120 - np.size(t)) * t[np.size(t)-1]
#        train_features = np.hstack((train_features, a))
n = 200
for i in range(np.size(train_features2)):
    t = train_features2[i]
    if np.size(t) > n:
        train_features2[i] = t[:n]
    if np.size(t) < n:
        a = (np.ones(n - np.size(t)) * t[np.size(t)-1]).tolist()
        train_features2[i].extend(a) #(list(map(int,a)))
train_features2 = np.array(train_features2)      
        
for i in range(np.size(test_features2)):
    t = test_features2[i]
    if np.size(t) > n:
        test_features2[i] = t[:n]
    if np.size(t) < n:
        a = (np.ones(n - np.size(t)) * t[np.size(t)-1]).tolist()
        test_features2[i].extend(a) #(list(map(int,a)))
test_features2 = np.array(test_features2)      

    
        
#multiple features        
modelw = lm.fit(train_features2, WhiteElos_train.reshape(-1, 1))
modelb = lm.fit(train_features2, BlackElos_train.reshape(-1, 1))
predictionw = lm.predict(test_features2)
predictionb = lm.predict(test_features2) 

#mean and diff method
model_mean = lm.fit(train_features2, mean_elos_train.reshape(-1, 1))
prediction_diff = lm.predict(test_features2)
print(metrics.mean_absolute_error(mean_elos_test, prediction_mean))

model_diff = lm.fit(train_features2, elo_diff_train.reshape(-1, 1))
prediction_mean = lm.predict(test_features2)
print(metrics.mean_absolute_error(elo_diff_test, prediction_diff))

model = lm.fit(train_features2, Elos_train)
prediction = lm.predict(test_features2)
dfb = pd.DataFrame({'Actual':Elos_test, 'Predicted': prediction})
#evaluate multiple features linear regression


from sklearn import metrics 
print(metrics.mean_absolute_error(WhiteElos_test, predictionw))  
print('Mean Squared Error:', metrics.mean_squared_error(WhiteElos_test, predictionw))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(WhiteElos_test, predictionw)))
#dfw = pd.DataFrame({'Actual':WhiteElos_test, 'Predicted': predictionw})
print(metrics.mean_absolute_error(BlackElos_test,predictionb))
print('Mean Squared Error:', metrics.mean_squared_error(BlackElos_test,predictionb))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(BlackElos_test,predictionb)))
#dfb = pd.DataFrame({'Actual':BlackElos_test, 'Predicted': predictionb})


#aaaa = pd.DataFrame(games)
#pgn_text = open('data.pgn').read()
#pgn_game = pgn.PGNGame()
#
#print (pgn.loads(pgn_text)) # Returns a list of PGNGame
#print (pgn.dumps(pgn_game)) # Returns a string with a pgn game
#print(pgn_game.move(1))


#neural networks
#train_features = list()
#for i in range(20000):
#    t = move_quality['MoveScores'][i].split()
#    if len(t) > 0:
#        t = [x for x in t if x != 'NA']
#        train_features.append(list(map(int,t)))
#        
#padded = pad_sequences(train_features, padding = 'post') 
#
#
Elos_train = np.hstack((WhiteElos_train.reshape((len(WhiteElos_train),1)),BlackElos_train.reshape(len(BlackElos_train),1)))
Elos_test = np.hstack((WhiteElos_test.reshape((len(WhiteElos_test),1)),BlackElos_test.reshape(len(BlackElos_test),1)))
NN_model = Sequential()
NN_model.add(Dense(128, kernel_initializer='normal',input_dim = n, activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(2, kernel_initializer='normal',activation='relu'))
#
NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
NN_model.summary()
##kfold = KFold(n_splits=10)
##results = cross_val_score(estimator, padded_train, Elos_train, cv=kfold)
##print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
NN_model.fit(train_features2, Elos_train, epochs = 2000, batch_size=19982, validation_data=(test_features2, Elos_test))

#scores = NN_model.evaluate(padded_test,Elos_test)
predicted = NN_model.predict(test_features2)
##print(NN_model.metrics_names[1], scores[1]*100)
#print(scores)
#NN_model.fit(train, target, epochs=500, batch_size=32, validation_split = 0.2, callbacks=callbacks_list)
#estimator = KerasClassifier(build_fn = model, epochs = 1000, batch_size =20000, verbose = 0 )
#        
#round_bElo_train = np.round(BlackElos_train, -2)
#round_wElo_train = np.round(WhiteElos_train, -2)
#round_wElo_test = np.round(WhiteElos_test, -2)
#round_bElo_test = np.round(BlackElos_test, -2)

#random Forest Model
model = RandomForestRegressor()
model.fit(train_features2, Elos_train)
predicted = model.predict(test_features2)
MAE = mean_absolute_error(Elos_test, predicted)

    
    
XGBModel = xgb.XGBRegressor()
XGBModel.fit(padded_train,Elos_train , verbose=False)
from sklearn.metrics import mean_squared_error
# Get the mean absolute error on the validation data :
XGBpredictions = XGBModel.predict(padded_test)
MAE = mean_absolute_error(Elos_test, XGBpredictions)    
rmse = np.sqrt(mean_squared_error(Elos_test, XGBpredictions))   
#predicted =
    
#    num_col = np.size(t,1)
        
        
        
        
        
#        if num_col < x: 
#            t = np.pad(t,((0,0),(0,x-num_col)),mode='constant',constant_values=t[num_col])
#        if num_col > x:
#            t = t[0,0:x]
#        train_features = np.vstack((train_features, t))
#        
        
        

mean_elos_train = (BlackElos_train + WhiteElos_train)/2
mean_elos_test = (BlackElos_test + WhiteElos_test)/2
elo_diff_train = WhiteElos_train - BlackElos_train
elo_diff_test = WhiteElos_test - BlackElos_test





