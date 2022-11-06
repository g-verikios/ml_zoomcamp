# %%
import pickle

# %%
model_file = 'model_C=1.0.bin'

# %%
with open(model_file, 'rb') as f_in: # now we read the file, its important to avoid to overwrite the file creating one with zero bytes

    (dv, model) = pickle.load(f_in)

# %%

customer = {
    'gender': 'female',
    'seniorcitizen': 0,
    'partner': 'yes',
    'dependents': 'no',
    'phoneservice': 'no',
    'multiplelines': 'no_phone_service',
    'internetservice': 'dsl',
    'onlinesecurity': 'no',
    'onlinebackup': 'yes',
    'deviceprotection': 'no',
    'techsupport': 'no',
    'streamingtv': 'no',
    'streamingmovies': 'no',
    'contract': 'month-to-month',
    'paperlessbilling': 'yes',
    'paymentmethod': 'electronic_check',
    'tenure': 1,
    'monthlycharges': 29.85,
    'totalcharges': 29.85
}

# %%
X = dv.transform([customer]) # remember that DictVectorizer expects a list

y_pred = model.predict_proba(X)[0,1] # Probabillity of a customer to churn

print('input', customer)

print('churn probability', y_pred)