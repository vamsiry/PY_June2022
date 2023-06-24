# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 06:28:31 2022

@author: rvamsikrishna
"""

#%%
#%%
#***************************************************************************
#**************************new one link******************************

#https://pythonguides.com/scikit-learn-pipeline/

#Scikit learn Pipeline custom function
#==========================================
#Scikit learn custom function is used to returns the two-dimension 
#array of value or also used to remove the outliers.

# zipfile_path = os.path.join(our_path, “housing.tgz”) is used to set the zip file path.
# urllib.request.urlretrieve(our_data_url, zipfile_path) is used to get the file from the URL.
# ourfile_path = os.path.join(our_path, “housing.csv”) is used to settng the csv file path.
# return pds.read_csv(ourfile_path) is used to read the pandas file.
# imputer = SimpleImputer(strategy=”median”) is used to calculate the median value for each column.
# ourdataset_num = our_dataset.drop(“ocean_proximity”, axis=1) is used to remove the ocean proximity.
# imputer.fit(ourdataset_num) is used to fit the model.
# our_text_cats = our_dataset[[‘ocean_proximity’]] isused to selecting the textual attribute.
# rooms_per_household = x[:, rooms] / x[:, household] is used to getting the three extra attribute by dividing appropriate attribute.

import os
import tarfile
from six.moves import urllib

OURROOT_URL = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
OURPATH = "datasets/housing"
OURDATA_URL = OURROOT_URL + OURPATH + "/housing.tgz"

def get_data(our_data_url=OURDATA_URL, our_path=OURPATH):
      if not os.path.isdir(our_path):
            os.makedirs(our_path)
      
      zipfile_path = os.path.join(our_path, "housing.tgz")

      urllib.request.urlretrieve(our_data_url, zipfile_path)
      ourzip_file = tarfile.open(zipfile_path)
      ourzip_file.extractall(path=our_path)
      ourzip_file.close()

get_data()

#%%
import pandas as pds

def load_our_data(our_path=OUR_PATH):
    ourfile_path = os.path.join(our_path, "housing.csv")
    return pds.read_csv(ourfile_path)

#%%
ourdataset = load_our_data()
ourdataset.head()

#%%
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")

ourdataset_num = our_dataset.drop("ocean_proximity", axis=1)

imputer.fit(ourdataset_num)

#transforming using the learnedparameters
x = imputer.transform(ourdataset_num)

#setting the transformed dataset to a DataFrame
ourdataset_numeric = pds.DataFrame(x, columns=ourdataset_num.columns)

#%%
from sklearn.preprocessing import OrdinalEncoder

our_text_cats = our_dataset[['ocean_proximity']]
our_encoder = OrdinalEncoder()

#transforming it
our_encoded_dataset = our_encoder.fit_transform(our_text_cats)

#%%
import numpy as num
from sklearn.base import BaseEstimator, TransformerMixin

#initialising column numbers
rooms,  bedrooms, population, household = 4,5,6,7

class CustomTransformer(BaseEstimator, TransformerMixin):
    #the constructor
    def __init__(self, add_bedrooms_per_room = True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    #estimator method
    def fit(self, x, y = None):
        return self
    #transfprmation
    def transform(self, x, y = None):
        rooms_per_household = x[:, rooms] / x[:, household]
        population_per_household = x[:, population] / x[:, household]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = x[:, bedrooms] / x[:, rooms]
            return num.c_[x, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return num.c_[x, rooms_per_household, population_per_household]


attrib_adder = CustomTransformer()
our_extra_attributes = attrib_adder.transform(our_dataset.values)

#%%
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

#the numeric attributes transformation pipeline
numericpipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CustomTransformer()),
    ])
    
numericattributes = list(our_dataset_numeric)

#the textual transformation pipeline
text_attribs = ["ocean_proximity"]


#setting the order of the two pipelines
our_full_pipeline = ColumnTransformer([
        ("numeric", numericpipeline, numericattributes),
        ("text", OrdinalEncoder(), text_attribs),
    ])
    
our_dataset_prepared = our_full_pipeline.fit_transform(our_dataset)
our_dataset_prepared



#%%
#%%
#***************************************************************************
#**************************new one link******************************

#https://pythonguides.com/scikit-learn-pipeline/

#Scikit learn Pipeline feature selection 
#===========================================

#Feature selection is defined as a method to select the features or 
#repeatedly select the features of the pipeline.

# sklearn.feature_selection module can be used for feature selection/dimensionality 
#reduction on sample sets, either to improve estimators’ accuracy scores
# or to boost their performance on very high-dimensional datasets.

#https://scikit-learn.org/stable/modules/feature_selection.html


#creating custom data set by uning make_classification from sklearn.datasets
#https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html
from sklearn.datasets import make_classification #Generate a random n-class classification problem.
from sklearn.model_selection import train_test_split

x, y = make_classification( 
    n_features=22,
    n_informative=5,
    n_redundant=0,
    n_classes=4,
    n_clusters_per_class=4,
    random_state=44,
)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)

#%%
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

anovafilter = SelectKBest(f_classif, k=5)


classifier = LinearSVC()


anova_svm = make_pipeline(anovafilter, classifier)
anova_svm.fit(x_train, y_train)


from sklearn.metrics import classification_report
y_pred = anova_svm.predict(x_test)
print(classification_report(y_test, y_pred))


#%%
#%%
#***************************************************************************
#**************************new one link******************************

#https://pythonguides.com/scikit-learn-pipeline/

#Scikit learn Pipeline cross validation
#===========================================

#Scikit learn pipeline cross-validation technique is defined as a process 
#for evaluating the result of a statical model that will spread to unseen data.

import numpy as np
import matplotlib.pyplot as plot

from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def lower_bound(cv_result):
    
    bestscore_idx = np.argmax(cv_result["mean_test_score"])

    return (
        cv_result["mean_test_score"][bestscore_idx]
        - cv_result["std_test_score"][bestscore_idx]
    )


def best_low_complexity(cv_result):
    
    threshold = lower_bound(cv_result)
    candidate_idx = np.flatnonzero(cv_result["mean_test_score"] >= threshold)
    best_idx = candidate_idx[
        cv_result["param_reduce_dim__n_components"][candidate_idx].argmin()
    ]
    return best_idx


pipeline = Pipeline(
    [
        ("reduce_dim", PCA(random_state=42)),
        ("classify", LinearSVC(random_state=42, C=0.01)),
    ]
)

param_grid = {"reduce_dim__n_components": [6, 8, 10, 12, 14]}

grid = GridSearchCV(
    pipeline,
    cv=10,
    n_jobs=1,
    param_grid=param_grid,
    scoring="accuracy",
    refit=best_low_complexity,
)

x, y = load_digits(return_X_y=True)
grid.fit(x, y)

n_components = grid.cv_results_["param_reduce_dim__n_components"]
test_scores = grid.cv_results_["mean_test_score"]

plot.figure()
plot.bar(n_components, test_scores, width=1.4, color="r")

lower = lower_bound(grid.cv_results_)
plot.axhline(np.max(test_scores), linestyle="--", color="y", label="Best score")
plot.axhline(lower, linestyle="--", color=".6", label="Best score - 1 std")

plot.title("Balance Model Complexity And Cross-validated Score")
plot.xlabel("Number Of PCA Components Used")
plot.ylabel("Digit Classification Accuracy")
plot.xticks(n_components.tolist())
plot.ylim((0, 1.0))
plot.legend(loc="upper left")

best_index_ = grid.best_index_

print("The best_index_ is %d" % best_index_)
print("The n_components selected is %d" % n_components[best_index_])
print(
    "The corresponding accuracy score is %.2f"
    % grid.cv_results_["mean_test_score"][best_index_]
)
plot.show()

#%%
#%%
#%%
#***************************************************************************
#**************************new one link******************************

#https://pythonguides.com/scikit-learn-pipeline/

#Scikit learn Pipeline grid search
#===========================================

#Scikit learn pipeline grid search is an operation that defines the 
#hyperparameters and it tells the user about the accuracy rate of the model.

import numpy as np
import matplotlib.pyplot as plot
from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA, NMF
from sklearn.feature_selection import SelectKBest, chi2


x, y = load_digits(return_X_y=True)

pipeline = Pipeline([
        # the reduce_dim stage is populated by the param_grid
        ("reduce_dim", "passthrough"),
        ("classify", LinearSVC(dual=False, max_iter=10000))])


n_feature_options = [4, 8, 12]
c_options = [1, 10, 100, 1000]

param_grid = [
    {
        "reduce_dim": [PCA(iterated_power=7), NMF()],
        "reduce_dim__n_components": n_feature_options,
        "classify__C": c_options,
    },
    {
        "reduce_dim": [SelectKBest(chi2)],
        "reduce_dim__k": n_feature_options,
        "classify__C": c_options,
    }]


reducer_labels = ["PCA", "NMF", "KBest(chi2)"]


grid = GridSearchCV(pipeline, n_jobs=1, param_grid=param_grid)
grid.fit(x, y)


mean_scores = np.array(grid.cv_results_["mean_test_score"])

mean_scores = mean_scores.reshape(len(c_options), -1, len(n_feature_options))

mean_scores = mean_scores.max(axis=0)

bar_offsets = np.arange(len(n_feature_options)) * (len(reducer_labels) + 1) + 0.5

plot.figure()
COLORS = "cyr"
for i, (label, reducer_scores) in enumerate(zip(reducer_labels, mean_scores)):
    plot.bar(bar_offsets + i, reducer_scores, label=label, color=COLORS[i])

plot.title("Comparing Feature Reduction Techniques")
plot.xlabel("Reduced Number Of Features")
plot.xticks(bar_offsets + len(reducer_labels) / 2, n_feature_options)
plot.ylabel("Digit Classification Accuracy")
plot.ylim((0, 1))
plot.legend(loc="Upper Left")

plot.show()


#%%
#%%
#%%
#***************************************************************************
#**************************new one link******************************

#https://pythonguides.com/scikit-learn-pipeline/

#Scikit learn Pipeline Pandas
#===========================================

#Scikit learn pipeline pandas is defined as a process that allows us the 
#string together various user-defined functions for building a pipeline.

import numpy as np
import matplotlib.pyplot as plot
import pandas as pd

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


x_digits, y_digits = datasets.load_digits(return_X_y=True)

pca = PCA()

scaler = StandardScaler()

logisticregr = LogisticRegression(max_iter=10000, tol=0.2)

pipeline = Pipeline(steps=[("scaler", scaler), ("pca", pca), ("logistic", logisticregr)])

param_grid = {
    "pca__n_components": [5, 15, 30, 45, 60],
    "logistic__C": np.logspace(-4, 4, 4),
}

searchgrid = GridSearchCV(pipeline, param_grid, n_jobs=2)

searchgrid.fit(x_digits, y_digits)

print("Best parameter (CV score=%0.3f):" % searchgrid.best_score_)
print(searchgrid.best_params_)

pca.fit(x_digits)

fig,(axis0, axis1) = plot.subplots(nrows=2, sharex=True, figsize=(6, 6))

axis0.plot(np.arange(1, pca.n_components_ + 1), 
           pca.explained_variance_ratio_, "+", linewidth=2)

axis0.set_ylabel("PCA explained variance ratio")

axis0.axvline(
    searchgrid.best_estimator_.named_steps["pca"].n_components,
    linestyle=":",
    label="n_components chosen")

axis0.legend(prop=dict(size=12))

# For each number of components, find the best classifier results
results = pd.DataFrame(searchgrid.cv_results_)

components_col = "param_pca__n_components"

best_classifications = results.groupby(components_col).apply(
    lambda g: g.nlargest(1, "mean_test_score"))

best_classifications.plot(x=components_col, y="mean_test_score", 
                          yerr="std_test_score", legend=False, ax=axis1)

axis1.set_ylabel("Classification accuracy (val)")

axis1.set_xlabel("n_components")

plot.xlim(-1, 70)

plot.tight_layout()
plot.show()


#%%
#%%
#%%
#***************************************************************************
#**************************new one link******************************

#https://pythonguides.com/scikit-learn-pipeline/

#Scikit learn Pipeline Pickle
#===========================================

#Scikit learn pipeline pickle is defined as a process to save the file in a 
#periodic manner or we can say that the way of serializing the objects.

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
import pickle
import numpy as np


pipeline = make_pipeline(RandomForestClassifier())

x_train = np.array([[3,9,6],[5,8,3],[2,10,5]])
y_train = np.array([27, 30, 19])

pipeline.fit(x_train, y_train)

model = pipeline.named_steps['randomforestclassifier']
outfile = open("model.pkl", "wb")
pickle.dump(model, outfile)
outfile.close()
model

#%%
#%%
#%%
#***************************************************************************
#**************************new one link******************************

#https://pythonguides.com/scikit-learn-pipeline/

#Scikit learn Pipeline one-hot encoding
#===========================================

#Scikit learn pipeline one-hot encoding is defined or represents the categorical
# variables. In this, the need for the categorical variable is 
#mapped into the integer value.

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PoissonRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

nsamples, nfeatures = 1000, 20
range = np.random.RandomState(0)
x = range.randn(nsamples, nfeatures)

y = range.poisson(lam=np.exp(x[:, 5]) / 2)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=range)

glm = PoissonRegressor()

gbdt = HistGradientBoostingRegressor(loss="poisson", learning_rate=0.01)

glm.fit(x_train, y_train)
gbdt.fit(x_train, y_train)
print(glm.score(x_test, y_test))
print(gbdt.score(x_test, y_test))


from sklearn import set_config
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression

set_config(display="diagram")

num_proc = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

cat_proc = make_pipeline(
    SimpleImputer(strategy="constant", fill_value="missing"),
    OneHotEncoder(handle_unknown="ignore"),
)

preprocessor = make_column_transformer(
    (num_proc, ("feat1", "feat3")), (cat_proc, ("feat0", "feat2"))
)

classifier = make_pipeline(preprocessor, LogisticRegression())
classifier

#%%    











