
# coding: utf-8

# In[412]:

import pandas as pd
import numpy as np
import sys

import warnings
warnings.filterwarnings("ignore")


train_dataset_filepath= sys.argv[1]
test_dataset_filepath = sys.argv[2]
output_filepath = sys.argv[3]


#reading train data
df = pd.read_csv(train_dataset_filepath, sep=",")

#reading test data
test_data = pd.read_csv(test_dataset_filepath, sep=",")

# backup of train data
data=df.copy(deep=True)

#storing the labels and IDs from train data in labels and trainIds
labels=data.iloc[:,1].values
trainIds=data.iloc[:,0].values

# check if any outliers are there in the training data set
import matplotlib.pyplot as plt

plt.scatter(trainIds, labels, color='g')
plt.xlabel('id')
plt.ylabel('y')
plt.show()


# In[413]:

# we found that there is only one outlier in the training dataset whose y-value is more than 200.
# Therefore, we will get rid of this instance

data=data[data['y'] < 200]


#storing the labels and IDs from train data in new_labels and new_trainIds
new_labels=data.iloc[:,1].values
new_trainIds=data.iloc[:,0].values

#Now we can see that the outlier has been removed from the training dataset
plt.scatter(new_trainIds, new_labels, color='g')
plt.xlabel('id')
plt.ylabel('y')

plt.show()

#storing the IDs from test data in testIds
testIds=test_data.iloc[:,0].values

#removing label and IDs from train data
data.drop(data.columns[[0,1]], axis=1,inplace=True) 

#removing IDs from test data
test_data.drop(test_data.columns[0], axis=1,inplace=True) 

#appending test data to train data, so that all unique values for the features having categorical data can be encoded properly
data=data.append(test_data)


# In[414]:

from sklearn import preprocessing

#converting categorical data to numerical values i.e. integers
for col in range(0,8):
    data[data.columns[col]]=data[data.columns[col]].astype('category').cat.codes

# storing the processed test data into test_data
test_data=data.iloc[4208:,:]

# storing the processed train data into data
data=data.iloc[0:4208,:]


#converting the processed data into data frames
test_data=pd.DataFrame(test_data)
data=pd.DataFrame(data)

# removing those features from both train and test data set whose values are constant throughout the train data set

featuresDelete = []
for i in range(8,376):
    if(data.iloc[:,i].sum()==0 or data.iloc[:,i].sum()==4208):
        featuresDelete.append(i)

data.drop(data.columns[featuresDelete], axis=1,inplace=True) 
test_data.drop(test_data.columns[featuresDelete], axis=1,inplace=True) 

#normalizing the data
scaler = preprocessing.StandardScaler().fit(data)
data=scaler.transform(data) 
test_data=scaler.transform(test_data) 

#converting data and test_data into data frames
data=pd.DataFrame(data)
test_data=pd.DataFrame(test_data)


# In[415]:

#inserting IDs back to train data set
idx = 0
new_col = new_trainIds
data.insert(loc=idx, column='ID', value=new_col)

#inserting labels back to train data set
idx = 365
new_col = new_labels
data.insert(loc=idx, column='y', value=new_col)

#inserting IDs back to test data set
idx = 0
new_col = testIds
test_data.insert(loc=idx, column='ID', value=new_col)


#Principal COmponent Analysis
from sklearn.decomposition import PCA

pca = PCA(n_components=30)
pca.fit(data.iloc[:,1:365])
print("Explained variance ratio using principal component analysis: ")
print(pca.explained_variance_ratio_)
# not using PCA as we are not getting a higher variance within a feature


# In[416]:

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
X=data.iloc[:,1:364]
Y=data.iloc[:,365]

X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.4, random_state=0)

from sklearn.svm import LinearSVR
reg = LinearSVR(random_state=0,epsilon=1e-05)
r2_score=reg.fit(X_train, y_train).score(X_valid,y_valid)
pred = reg.predict(X_valid)
mean_sq_err = mean_squared_error(y_valid, pred)
print("R^2 Score: using Linear Support Vector Regressor is: ",r2_score)
print("Mean Squared Error: using Linear Support Vector Regressor is: ",mean_sq_err)


# In[417]:

from sklearn.svm import SVR
reg=SVR(kernel='rbf', C=1e3, gamma='auto')
r2_score=reg.fit(X_train, y_train).score(X_valid,y_valid)
pred = reg.predict(X_valid)
mean_sq_err = mean_squared_error(y_valid, pred)
print("R^2 Score: using Support Vector Regressor is: ",r2_score)
print("Mean Squared Error: using Support Vector Regressor is: ",mean_sq_err)


# In[418]:

from sklearn.neural_network import MLPRegressor
reg = MLPRegressor(hidden_layer_sizes=(5,5),activation='logistic', solver='adam', alpha=0.001,learning_rate='constant', learning_rate_init=0.01, max_iter=1000,random_state=0, tol=0.0001, early_stopping=False,epsilon=1e-08)
r2_score=reg.fit(X_train, y_train).score(X_valid,y_valid)
pred = reg.predict(X_valid)
mean_sq_err = mean_squared_error(y_valid, pred)
print("R^2 Score: using MLP Regressor is: ",r2_score)
print("Mean Squared Error: using MLP Regressor is: ",mean_sq_err)


# In[419]:

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
reg = AdaBoostRegressor(DecisionTreeRegressor(max_depth=10),n_estimators=20, random_state=0)
r2_score=reg.fit(X_train, y_train).score(X_valid,y_valid)
pred = reg.predict(X_valid)
mean_sq_err = mean_squared_error(y_valid, pred)
print("R^2 Score: using AdaBoostRegressor is: ",r2_score)
print("Mean Squared Error: using AdaBoost Regressor is: ",mean_sq_err)


# In[420]:

from sklearn.ensemble import GradientBoostingRegressor
reg = GradientBoostingRegressor(n_estimators=200, max_depth=5,min_samples_split=3, learning_rate=0.01, loss='huber',warm_start=True,random_state=0)
r2_score=reg.fit(X_train, y_train).score(X_valid,y_valid)
pred = reg.predict(X_valid)
mean_sq_err = mean_squared_error(y_valid, pred)
print("R^2 Score: using Gradient Boosting Regressor is: ",r2_score)
print("Mean Squared Error: using Gradient Boosting Regressor is: ",mean_sq_err)


# In[421]:

from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
reg = BaggingRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=100, max_features=200,max_samples=100,n_jobs=1, random_state=2)
r2_score=reg.fit(X_train, y_train).score(X_valid,y_valid)
pred = reg.predict(X_valid)
mean_sq_err = mean_squared_error(y_valid, pred)
print("R^2 Score: using Bagging Regressor is: ",r2_score)
print("Mean Squared Error: using Bagging Regressor is: ",mean_sq_err)


# In[422]:

# using Gradient Boosting Regressor as our best estimator to predict the testing time.
bestreg=GradientBoostingRegressor(n_estimators=200, max_depth=5,min_samples_split=3, learning_rate=0.01, loss='huber',warm_start=True,random_state=0)
predictions=bestreg.fit(X_train, y_train).predict(test_data.iloc[:,1:364])
predictions=pd.DataFrame(predictions)
idx = 0
new_col = testIds
predictions.insert(loc=idx, column='ID', value=new_col)
predictions=predictions.rename(columns={0:'y'})
predictions.to_csv(output_filepath,sep=',',index=False)

# For Hyper parameter tuning using Grid Search
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

models ={
         "Linear Support Vector Regressor": LinearSVR(),
         
         "Support Vector Regressor": SVR(),

         "Ada Boost Regressor": AdaBoostRegressor(),

         "Bagging Regressor": BaggingRegressor(),

         "Neural Net": MLPRegressor(),
         
         "Gradient Boosting Regressor": GradientBoostingRegressor()
         

}

 
parameters = {
              "Linear Support Vector Regressor":{
                                                "epsilon":(1e-05,1e-03,1e-04),
                                                "C":(1.0,5.0,3.0),
                                                "random_state":(0,1,2,3,4),
                                                "loss":('epsilon_insensitive','squared_epsilon_insensitive')
                                                },
              
              "Support Vector Regressor":{
                                          "C":(1.0,2.0,3.0,4.0,5.0),
                                          "max_iter":(100,200,300,400),
                                          "epsilon":(1e-05,1e-03,1e-04),
                                          "kernel":('rbf','poly'),
                                          "degree": (2,4,6),
                                          "tol":(1e-3,1e-4,1e-5),
                                          "max_iter":(100,200,300,400)
                                         },



              "Ada Boost Regressor":{
                                      "n_estimators":(20,70,100,200),
                                      "random_state":(0,1,2,3,4),
                                      "loss" : ('linear', 'square', 'exponential'),
                                      "learning_rate": (1,2,2.5,3)
                                    },
              
    
    
             "Bagging Regressor":  {
                                     "n_estimators":(20,70,100,200),
                                     "n_jobs":(1,2,3),
                                     "random_state":(0,1,2,3,4),
                                     "max_samples":(20,1,50,100,10),
                                     "max_features":(50,100,200,350)
                                    },

              
    
             "Neural Net":{
                        "activation" : ('logistic', 'relu', 'identity'),
                        "solver" : ('lbfgs','sgd','adam'),
                        "alpha":(0.001,0.002,0.005),
                        "power_t":(0.05,0.01,0.03),                   
                        "random_state":(0,1,2,3,4),
                        "learning_rate_init" :(0.01,0.05,0.1),
                        "learning_rate" : ('constant', 'invscaling', 'adaptive'),
                        "tol" : (0.0001,0.0003,0.05),
                        "epsilon":(1e-08,1e-05,1e-03),
                        "momentum":(0.1,0.5,1.0),
                        "early_stopping":(True,False),
                        "hidden_layer_sizes":((5,5),(10,),(6,)),
                        "max_iter":(100,200,300,400)
                         },

              "Gradient Boosting Regressor":{
                                              "loss" : ('ls', 'lad', 'huber', 'quantile'),
                                              "n_estimators":(20,70,100,200),
                                              "random_state":(0,1,2,3,4),
                                              "learning_rate": (.1,2,.5,3),
                                              "min_samples_split":(2,3,5),
                                              "min_samples_leaf":(3,5,7),
                                              "max_depth":(2,5,10),
                                              "alpha":(0.9,1.5,2,0.5),
                                              "presort":(True,False,'auto'),
                                              "criterion":('friedman_mse','mse','mae')
                                            }


             }

 

classifier = [
              "Linear Support Vector Regressor",
              "Support Vector Regressor",
               "Ada Boost Regressor",
              "Bagging Regressor",
              "Neural Net",
              "Gradient Boosting Regressor"
]

 

for c in classifier:

    clf = GridSearchCV(models[c],parameters[c],cv=5,n_jobs=5)

    clf = clf.fit(X_train, y_train)

    score = clf.score(X_valid,y_valid)
    
    y_pred = clf.predict(X_valid)
    
    mse = mean_squared_error(y_valid, y_pred)

    print("R^2 Score: using ",c," is: ",score)
    print("Mean Squared Error: using ",c," is: ",mse)
    print(clf.best_params_)
    print("-------------------------------------------") 


