# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 21:17:38 2022

@author: rvamsikrishna
"""

#https://analyticsindiamag.com/hands-on-tutorial-on-machine-learning-pipelines-with-scikit-learn/

#Machine learning has certain steps to be followed namely – data collection,
# data preprocessing(cleaning and feature engineering), model training, validation
# and prediction on the test data(which is previously unseen by model). 


#Here testing data needs to go through the same preprocessing as training data. 
#For this iterative process, pipelines are used which can automate the entire 
#process for both training and testing data. It ensures reusability of the model
# by reducing the redundant part, thereby speeding up the process. 
#This could prove to be very effective during the production workflow.

# Advantages of using Pipeline:
#-------------------------------
# Automating the workflow being iterative.
# Easier to fix bugs 
# Production Ready
# Clean code writing standards
# Helpful in iterative hyperparameter tuning and cross-validation evaluation


# Challenges in using Pipeline:
#---------------------------------------
# Proper data cleaning
# Data Exploration and Analysis
# Efficient feature engineering


#%%
#%%
#After loading the data, split it into training and testing then build pipeline 
#object wherein standardization is done using StandardScalar() and dimensionality
# reduction using PCA(principal component analysis) both of these with be 
#fit and transformed(these are transformers), lastly the model to use is 
#declared here it is LogisticRegression, this is the estimator. 
#The pipeline is fitted and the model performance score is determined.

#%%
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

#%%
iris_df=load_iris()

#%%
X_train,X_test,y_train,y_test=train_test_split(iris_df.data,iris_df.target,test_size=0.3,random_state=0)

#%%
pipeline_lr=Pipeline([('scalar1',StandardScaler()),
                     ('pca1',PCA(n_components=2)),                    
                     ('lr_classifier',LogisticRegression(random_state=0))])

#%%
model = pipeline_lr.fit(X_train, y_train)
model.get_params() #is used to see all the hyperparameter.

model.score(X_test,y_test)

#%%
#Use the following two lines of code inside the Pipeline object for filling
# missing values and change categorical values to numeric. 
#(Since iris dataset doesn’t contain these we are not using)

('imputer', SimpleImputer(strategy='most_frequent')) #filling missing values

('onehot', OneHotEncoder(handle_unknown='ignore'))    #convert categorical 


#%%
#%%
#%%
#Stacking Multiple Pipelines to Find the Model with the Best Accuracy
#----------------------------------------------------------------

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm

pipeline_lr=Pipeline([('scalar1',StandardScaler()),
                     ('pca1',PCA(n_components=2)), 
                     ('lr_classifier',LogisticRegression())])

pipeline_dt=Pipeline([('scalar2',StandardScaler()),
                     ('pca2',PCA(n_components=2)),
                     ('dt_classifier',DecisionTreeClassifier())])

pipeline_svm = Pipeline([('scalar3', StandardScaler()),
                      ('pca3', PCA(n_components=2)),
                      ('clf', svm.SVC())])

pipeline_knn=Pipeline([('scalar4',StandardScaler()),
                     ('pca4',PCA(n_components=2)),
                     ('knn_classifier',KNeighborsClassifier())])


pipelines = [pipeline_lr, pipeline_dt, pipeline_svm, pipeline_knn]


pipe_dict = {0: 'Logistic Regression', 1: 'Decision Tree', 2: 'Support Vector Machine',3:'K Nearest Neighbor'}


for pipe in pipelines:
  pipe.fit(X_train, y_train)


for i,model in enumerate(pipelines):
    print("{} Test Accuracy:{}".format(pipe_dict[i],model.score(X_test,y_test)))

#%%
# Identify the most accurate model on test data
best_acc = 0.0
best_clf = 0
best_pipe = ''
for idx, model in enumerate(pipelines):
	if model.score(X_test, y_test) > best_acc:
		best_acc = model.score(X_test, y_test)
		best_pipe = model
		best_clf = idx
print('Classifier with best accuracy: %s' % pipe_dict[best_clf])

# Save pipeline to file
joblib.dump(best_pipe, 'best_pipeline.pkl', compress=1)
print('Saved %s pipeline to file' % pipe_dict[best_clf]    

#%%
#%%
#%%
#Hyperparameter Tuning in Pipeline
#====================================
#With pipelines, you can easily perform a grid-search over a set of parameters
# for each step of this meta-estimator to find the best performing parameters.
# To do this you first need to create a parameter grid for your chosen model.
# One important thing to note is that you need to append the name that you 
#have given the classifier part of your pipeline to each parameter name. 
#In my code above I have called this ‘randomforestclassifier’ so I have added 
#randomforestclassifier__ to each parameter. Next, I created a grid search 
#object which includes the original pipeline. When I then call fit, the 
#transformations are applied to the data, before a cross-validated grid-search 
#is performed over the parameter grid.
        
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier

pipe = make_pipeline((RandomForestClassifier()))    

grid_param = [{"randomforestclassifier": [RandomForestClassifier()],
"randomforestclassifier__n_estimators":[10,100,1000],                 
"randomforestclassifier__max_depth":[5,8,15,25,30,None],                
"randomforestclassifier__min_samples_leaf":[1,2,5,10,15,100],
"randomforestclassifier__max_leaf_nodes": [2, 5,10]}]

gridsearch = GridSearchCV(pipe, grid_param, cv=5, verbose=0,n_jobs=-1) 

best_model = gridsearch.fit(X_train,y_train)

best_model.score(X_test,y_test)

#%%

















