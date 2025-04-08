import numpy as np
import pandas as pd
import random
import sys

from scipy.stats import randint
from scipy.stats.mstats import winsorize


from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.base import BaseEstimator, TransformerMixin


# unpack commands line args
valid_transformations = ['means', 'medians', 'sds', 'rmssds'] # finalized these as the most important features on 12/2
# valid_transformations = ['means', 'medians', 'iqrs', 'ranges', 'sds', 'rmssds', 'entropy', 'maxs', 'mins']
assert len(sys.argv[1:]) == 1, "ERROR: Expected one command-line argument"
name = sys.argv[1]
if name not in valid_transformations:
     raise ValueError(f"ERROR: '{name}' is not a valid transformation. Choose from {valid_transformations}.")


# set seeds and k
seed = 123
np.random.seed(seed)
random.seed(seed)
k = 5


# load the ids and labels
train_ids_labs = pd.read_csv('../objects/train_ids_labs.csv')
test_ids_labs = pd.read_csv('../objects/test_ids_labs.csv')


# unpack into ids, y
id_train, id_test = train_ids_labs['PATIENT_ID'], test_ids_labs['PATIENT_ID']
y_train, y_test = train_ids_labs['dep'], test_ids_labs['dep']


# load the features
X_train = pd.read_csv('../objects/train_' + name + '.csv')
X_test = pd.read_csv('../objects/test_' + name + '.csv')


# define winsorization class
class Winsorizer(BaseEstimator, TransformerMixin):

    def __init__(self, limits=(0.05, 0.05)):
        self.limits = limits

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.apply(lambda col: winsorize(col, limits=self.limits))


# set up pipeline object
pipeline = Pipeline([
    ('winsorizer', Winsorizer()),
    ('scaler', StandardScaler()),
    ('model', SVC(class_weight='balanced', kernel='rbf', random_state=seed, probability=True))
])


# define the grid
grid = {
    'model__C': [1, 10, 100], # regularization parameter
}


# set up the randomized search
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=grid,
    scoring='average_precision',  # optimizes for both precision and recall, but is threshold-agnostic
    cv=k,
    verbose=3
)


# loop through and transform train/test sets into lower dimension
fold = 0
meta_train = np.zeros(X_train.shape[0])
meta_test = np.zeros((X_test.shape[0], k))
meta_models = {}
kf = KFold(n_splits=k, shuffle=True, random_state=seed)

for train_idx, oof_idx in kf.split(X_train):

    print(f'Fold: {str(fold+1)}')

    # chop up the data
    X, y = X_train.iloc[train_idx], y_train.iloc[train_idx]
    X_oof = X_train.iloc[oof_idx]

    # search the grid and find best fit
    grid_search.fit(X, y)
    best_fit = grid_search.best_estimator_

    # generate predictions
    meta_train[oof_idx] = best_fit.predict_proba(X_oof)[:,1]
    meta_test[:,fold] = best_fit.predict_proba(X_test)[:,1]
    meta_models[fold] = best_fit

    # manage control flow
    fold += 1
    print('\n')


# compile and write results
out_train = pd.DataFrame({name: meta_train})
out_test = pd.DataFrame({name: meta_test.mean(axis=1)})
out_train.to_csv('../ensemble/train_base_' + name + '.csv', index=False)
out_test.to_csv('../ensemble/test_base_' + name + '.csv', index=False)
