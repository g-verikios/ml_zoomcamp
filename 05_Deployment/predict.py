# %% IMPORTS
import pickle

from flask import Flask
from flask import request
from flask import jsonify

# %% LOAD MODEL
model_file = 'model_C=1.0.bin'

with open(model_file, 'rb') as f_in: # now we read the file, its important to avoid to overwrite the file creating one with zero bytes

    (dv, model) = pickle.load(f_in)


# %% PREDICT FUNCTION & APP

app = Flask('churn') # create a flask app

@app.route('/predict', methods = ['POST']) # POST to send info about the customer


def predict():
    customer = request.get_json()

    ### This should be inside a separate function ideally 
    X = dv.transform([customer]) # remember that DictVectorizer expects a list
    y_pred = model.predict_proba(X)[0,1] # Probabillity of a customer to churn
    churn = y_pred >= 0.5
    ####
    result = {
        'churn_probability': float(y_pred),
        'churn': bool(churn)
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host ='localhost', port=9696) 