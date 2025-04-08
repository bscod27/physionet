import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline


# unpack files and read in meta learner X data
path = '../ensemble/'
train_files = [file for file in os.listdir(path) if 'train_' in file]
test_files = [file for file in os.listdir(path) if 'test_' in file]
model_files = [file for file in os.listdir(path) if 'models_' in file]


# read in meta learner data
def read_data(files):
    df = pd.DataFrame()
    for i in range(len(files)):
        df = pd.concat([df,pd.read_csv(path + files[i])], axis=1)
    return df

X_train = read_data(train_files)
X_test = read_data(test_files)


# read in the labels and outcomes
train_ids_labs = pd.read_csv('../objects/train_ids_labs.csv')
test_ids_labs = pd.read_csv('../objects/test_ids_labs.csv')
id_train, id_test = train_ids_labs['PATIENT_ID'], test_ids_labs['PATIENT_ID']
y_train, y_test = train_ids_labs['dep'], test_ids_labs['dep']


# view the distributions of features to ensure they are on the same scale
print(pd.DataFrame([X_train.mean(axis=0), X_train.std(axis=0)], index=['mean', 'std']))

# train the meta learner
model = Pipeline([
    ('scaler', MinMaxScaler()),  # to get things on a probability distribution
    ('logreg', LogisticRegression(class_weight='balanced', random_state=123))
])

model.fit(X_train, y_train)
model.named_steps['logreg'].coef_


# extract the probs and actuals
probs = model.predict_proba(X_test)[:,1]
actuals = y_test


# write to folder
out = pd.DataFrame({'id': id_test, 'probs':probs, 'actuals':actuals})
out.head()
out.to_csv('../results/preds.csv', index=False)
