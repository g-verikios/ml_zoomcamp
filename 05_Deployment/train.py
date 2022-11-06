# %%
#@ IMPORTS

import pandas as pd
import numpy as np

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

from tqdm.auto import tqdm

import pickle

# PARAMETERS (normally given through CLI for eg.)

n_splits = 5
C = 1.0
output_file = f'model_C={C}.bin'

# %% 
# DATA PREPARATION
df = pd.read_csv('../03_Logistic_Regression/WA_Fn-UseC_-Telco-Customer-Churn.csv')

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(0)

df.columns = df.columns.str.lower().str.replace(' ', '_')

string_columns = list(df.dtypes[df.dtypes == 'object'].index)

for col in string_columns:
    df[col] = df[col].str.lower().str.replace(' ', '_')

df.churn = (df.churn == 'yes').astype(int)

# %% 
# SPLIT THE DATASET
df_full_train, df_test = train_test_split(df, test_size=0.2,  random_state=1)

df_full_train = df_full_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

# %%
# Columns to keep for training
categorical = ['gender', 'seniorcitizen', 'partner', 'dependents',
               'phoneservice', 'multiplelines', 'internetservice',
               'onlinesecurity', 'onlinebackup', 'deviceprotection',
               'techsupport', 'streamingtv', 'streamingmovies',
               'contract', 'paperlessbilling', 'paymentmethod']
numerical = ['tenure', 'monthlycharges', 'totalcharges']

columns = categorical + numerical

# %% 
# Functions for training and predicting

# %%
def train(df_train, y_train, columns, C):
    
    dicts = df_train[columns].to_dict(orient = 'records')
    
    dv = DictVectorizer(sparse = False)
    X_train = dv.fit_transform(dicts)
    
    model = LogisticRegression(solver='liblinear', C=C) # we use max_iter to avoid warnings
    model.fit(X_train, y_train)
    
    return dv, model

def predict(df, dv, model, columns):
    dicts = df[columns].to_dict(orient = 'records')
    
    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1] # probabilities only for positive examples
    
    return y_pred

# %% 
# VALIDATION

print(f'Doing validation with C={C}...')

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

fold = 0 # keep the fold number 
scores = []

for train_idx, val_idx in tqdm(kfold.split(df_full_train)): 

    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.churn.values
    y_val = df_val.churn.values

    dv, model = train(df_train, y_train, columns, C=C)

    y_pred = predict(df_val, dv, model, columns)

    auc = roc_auc_score(y_val, y_pred)

    scores.append(auc)

    print(' ')
    print(f'auc on fold {fold} is {auc}')
    fold = fold +1

print('Validation results:')
print('C=%s, %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))

# %% 
# ### FINAL MODEL

print('Training the final model...')

dv, model = train(df_full_train, df_full_train.churn.values, columns, C)

y_test = df_test.churn.values

y_pred = predict(df_test, dv, model, columns)

auc = roc_auc_score(y_test, y_pred)

print(f'auc={auc}')

# %% 
# SAVE THE MODEL

with open(output_file, 'wb') as f_out:

    pickle.dump((dv, model), f_out)

print(f'the model is saved to {output_file}')
