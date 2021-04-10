import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def sigmoid_func(X, theta):
    z = np.dot(X, theta)
    return 1 / (1 + np.exp(-z))

def gradient_descent(X, h, y):
    return np.dot(X.T, (h - y))/y.shape[0]

def update_theta(theta, alpha, gradient):
    return theta - alpha*gradient

dataFrame = pd.read_csv("dataset_files/finalest_ipl1.csv") # dataset

seasonMapping = {2008: 1, 2009: 2, 2010: 3, 2011: 4, 2012: 5, 2013: 6, 2014: 7, 2015: 8, 2016: 9, 2017: 10, 2018: 11, 2019: 12, 2020: 13}
dataFrame["season"] = dataFrame["season"].map(seasonMapping)

team1Mapping = {"CSK": 2, "DC": 3, "RCB": 4, "RR": 5, "MI": 6, "KKR": 7, "KXIP": 8, "SRH": 9}
dataFrame["team1"] = dataFrame["team1"].map(team1Mapping)

team2Mapping = {"CSK": 2, "DC": 3, "RCB": 4, "RR": 5, "MI": 6, "KKR": 7, "KXIP": 8, "SRH": 9}
dataFrame["team2"] = dataFrame["team2"].map(team2Mapping)

toss_winnerMapping = {"CSK": 2, "DC": 3, "RCB": 4, "RR": 5, "MI": 6, "KKR": 7, "KXIP": 8, "SRH": 9}
dataFrame["toss_winner"] = dataFrame["toss_winner"].map(toss_winnerMapping)

toss_decisionMapping = {"bat": 1, "field": 2, "bowl": 2}
dataFrame["toss_decision"] = dataFrame["toss_decision"].map(toss_decisionMapping)

winnerMapping = {"CSK": 2, "DC": 3, "RCB": 4, "RR": 5, "MI": 6, "KKR": 7, "KXIP": 8, "SRH": 9}
dataFrame["winner"] = dataFrame["winner"].map(winnerMapping)

dataFrame.loc[dataFrame["winner"]==dataFrame["team1"],"winner"]=1
dataFrame.loc[dataFrame["winner"]!=dataFrame["team1"],"winner"]=0

dataFrame.loc[dataFrame["toss_winner"]==dataFrame["team1"],"toss_winner"]=1
dataFrame.loc[dataFrame["toss_winner"]!=dataFrame["team1"],"toss_winner"]=0

df = dataFrame.copy()

feature_names = ['season', 'team1', 'team2', 'toss_winner', 'toss_decision', 'team1_point', 'team1_home_advantage']

X = df[feature_names]
y = df['winner']


learning_rate = 0.1
iterations = 200

# print(X.to_string())

# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0 )

# x_train = pd.DataFrame([x_train])
# y_train = pd.DataFrame([y_train])
# x_test = pd.DataFrame([x_test])
# y_test = pd.DataFrame([y_test])

intercept = np.ones((X.shape[0], 1))

# print(intercept)

X = np.concatenate((intercept, X), axis=1)
# print(X)

theta = np.zeros(X.shape[1])
# theta = np.random.rand(X.shape[1],1)
# print(theta)

for i in range(iterations):
    y1 = sigmoid_func(X, theta)
    gradient = gradient_descent(X, y1, y)
    # print("gradient",gradient)
    theta = update_theta(theta, learning_rate, gradient)
    
print("Learning rate:",learning_rate)
print("Iterations:",iterations)

pred_y = sigmoid_func(X, theta)

pred_y_round_off = pd.DataFrame(np.around(pred_y, decimals=2))
pred_y_round_off.loc[pred_y_round_off[0] >= 0.5,"predicted"]=1
pred_y_round_off.loc[pred_y_round_off[0] < 0.5,"predicted"]=0


# print(pred_y_round_off['predicted'].to_string())

# count = 0
# for count in range( np.size( pred_y_round_off ) ) :
		
# 	if y_test[count] == pred_y_round_off[count] :			
# 		correctly_classified = correctly_classified + 1
			
# 	count = count + 1
		
# print( "Accuracy:", (correctly_classified / count ) * 100 )

accuracy = (pred_y_round_off.loc[pred_y_round_off['predicted']==y[0]].shape[0]/pred_y_round_off.shape[0])*100
print("Accuracy: ", accuracy)