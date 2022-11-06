
from flask import Flask

app = Flask('ping') # create a flask app

# Now we need to put a decorator on our function

@app.route('/ping', methods = ['GET']) 

def ping():
    return 'PONG'

if __name__ == '__main__':
    app.run(debug=True, host ='0.0.0.0', port=9696) 