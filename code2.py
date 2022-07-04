# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# Common imports
import numpy as np
import os

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "ensembles"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
    
#%%  import data
import pandas as pd

mnistX = pd.read_pickle("Xtraining") #data
mnistY = pd.read_pickle("Ytraining") #labels
#mnistX.flags


#%%
X, y = mnistX, mnistY
X.shape
y.shape

#%%Split data into training and testing
#X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
from sklearn.model_selection import train_test_split
X_train_full, X_test = train_test_split(mnistX, test_size=0.2, random_state=42)
y_train_full, y_test = train_test_split(mnistY, test_size=0.2, random_state=42)
#split 20 train 80 test
X_train, X_valid = train_test_split(X_train_full, test_size=0.2, random_state=42)
y_train, y_valid = train_test_split(y_train_full, test_size=0.2, random_state=42)
#split training again for validation data

#%% nonlinear SVM with polynomial kernel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
pksvm_clf = Pipeline([ 
("scaler", StandardScaler()), 
("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5, probability = True)) 
]) 

pksvm_clf.fit(X_train, y_train)

#%% multilabel classification
from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier(weights = 'uniform', n_neighbors = 10, n_jobs = -1)
knn_clf.fit(X_train, y_train)

#%% random forest and ensemble with voting classifier
print("start of block 12")
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC



rnd_clf = RandomForestClassifier(n_estimators=250, max_leaf_nodes=16, n_jobs=-1, random_state = 42)
rnd_clf.fit(X_train, y_train)

svm_clf = SVC(gamma="scale", random_state=42)

voting_clf = VotingClassifier(estimators=[('knn',knn_clf),('rnd', rnd_clf), ('pksvm', pksvm_clf)],voting='soft', n_jobs=-1)

voting_clf.fit(X_train, y_train)
"""
param_grid = [
{'n_estimators': [10, 100, 250, 500], 'max_leaf_nodes': [2, 4, 6, 8, 16]}]
forest_reg = RandomForestClassifier()
grid_search = GridSearchCV(forest_reg, param_grid, cv=4,
scoring='neg_mean_squared_error',
return_train_score=True)
grid_search.fit(X_train, y_train)
print( grid_search.best_params_)
"""
"""
from sklearn.metrics import accuracy_score

for clf in ( rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
 """   
#%% Accuracy predict training
from sklearn.metrics import accuracy_score

pksvm_pred = pksvm_clf.predict(X_valid)
knn_pred = knn_clf.predict(X_valid)
rnd_pred = rnd_clf.predict(X_valid)
voting_pred = voting_clf.predict(X_valid)

"""
print("pksvm prediction accuracy: "+ str(pksvm_pred))
print("knn prediction accuracy: "+ str(knn_pred))
print("rnd_ prediction accuracy: "+ str(rnd_pred))
print("voting/ensemble prediction accuracy: "+ str(voting_pred))
"""
#print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
#%% Accuracy score training
pksvm_acc = accuracy_score(y_valid, pksvm_pred)
knn_acc = accuracy_score(y_valid, knn_pred)
rnd_acc = accuracy_score(y_valid, rnd_pred)
voting_acc = accuracy_score(y_valid, voting_pred)
print("pksvm training accuracy: "+ str(pksvm_acc))
print("knn training accuracy: "+ str(knn_acc))
print("rnd_ training accuracy: "+ str(rnd_acc))
print("voting/ensemble training accuracy: "+ str(voting_acc))

#%% Valid scores
pksvm_predv = pksvm_clf.predict(X_test)
knn_predv = knn_clf.predict(X_test)
rnd_predv = rnd_clf.predict(X_test)
voting_predv = voting_clf.predict(X_test)

#print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
#%% valid score training
pksvm_accv = accuracy_score(y_test, pksvm_predv)
knn_accv = accuracy_score(y_test, knn_predv)
rnd_accv = accuracy_score(y_test, rnd_predv)
voting_accv = accuracy_score(y_test, voting_predv)
print("pksvm valid accuracy: "+ str(pksvm_accv))
print("knn valid accuracy: "+ str(knn_accv))
print("rnd_ valid accuracy: "+ str(rnd_accv))
print("voting/ensemble valid accuracy: "+ str(voting_accv))
#%% confusion matrix
from sklearn.metrics import confusion_matrix
pksvm_con = confusion_matrix(y_test, pksvm_predv)
knn_con = confusion_matrix(y_test, knn_predv)
rnd_con = confusion_matrix(y_test, rnd_predv)
voting_con = confusion_matrix(y_test, voting_predv)
#print("pksvm confusion matrix:")
#print( confusion_matrix(y_test, pksvm_predv))


#%% plot matrix
plt.matshow(pksvm_con, cmap=plt.cm.gray)
plt.show()
plt.savefig("pksvm_con.png")

plt.matshow(knn_con, cmap=plt.cm.gray)
plt.show()
plt.savefig("knn_con.png")

plt.matshow(rnd_con, cmap=plt.cm.gray)
plt.show()
plt.savefig("rnd_con.png")

plt.matshow(voting_con, cmap=plt.cm.gray)
plt.show()
plt.savefig("voting_con.png")

#%% Full dataset ################################################################
#%% nonlinear SVM with polynomial kernel
#X, y = X[:60], y[:60]
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

from sklearn.svm import SVC
fpksvm_clf = Pipeline([ 
("scaler", StandardScaler()), 
("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5)) 
]) 

fpksvm_clf.fit(X, y)
#%% multilabel classification
full_knn_clf = KNeighborsClassifier(weights = 'uniform', n_neighbors = 10, n_jobs = -1)
full_knn_clf.fit(X, y)

#%% random forest and ensemble with voting classifier


full_rnd_clf = RandomForestClassifier(n_estimators=250, max_leaf_nodes=16, n_jobs=-1)
full_rnd_clf.fit(X, y)

full_svm_clf = SVC(gamma="scale", random_state=42)

full_voting_clf = VotingClassifier(estimators=[('full_knn',full_knn_clf),('full_rf', full_rnd_clf), ('full_svc', full_svm_clf)],voting='soft', n_jobs=-1)

full_voting_clf.fit(X, y)

#%% save models

joblib.dump(pksvm_clf, "pksvm_clf.pkl")
joblib.dump(knn_clf, "knn_clf.pkl")
joblib.dump(rnd_clf, "rnd_clf.pkl")
joblib.dump(voting_clf, "voting_clf.pkl")


joblib.dump(fpksvm_clf, "fpksvm_clf.pkl")
joblib.dump(full_knn_clf, "full_knn_clf.pkl")
joblib.dump(full_rnd_clf, "full_rnd_clf.pkl")
joblib.dump(full_voting_clf, "full_voting_clf.pkl")

fpksvm_clf = joblib.load("./fpksvm_clf.pkl")
full_knn_clf = joblib.load("./full_knn_clf.pkl")
full_rnd_clf = joblib.load("./full_rnd_clf.pkl")
full_voting_clf = joblib.load("./full_voting_clf.pkl")

pksvm_pkl = joblib.load("./pksvm_clf.pkl")
knn_pkl = joblib.load("./knn_clf.pkl")
rnd_pkl = joblib.load("./rnd_clf.pkl")
voting_pkl = joblib.load("./voting_clf.pkl")

all_models = [pksvm_pkl, knn_pkl, rnd_clf, voting_pkl]
joblib.dump(all_models, "model2.pkl")
all_loaded_models = joblib.load("./model2.pkl")

m1 = all_loaded_models[0]
m2 = all_loaded_models[1]
m3 = all_loaded_models[2]
m4 = all_loaded_models[3]






