from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

with open('/home/GundaYasaswini/House/flask_app.py', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    bed = int(request.form['bedrooms'])
    bath = int(request.form['bathroms'])
    loc = int(request.form['location'])
    size = int(request.form['area'])
    status = int(request.form['status'])
    facing = int(request.form['facing'])
    Type = int(request.form['type'])

    input_data = np.array([[bed, bath, loc, size, status, facing, Type]])

    prediction = model.predict(input_data)[0]

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)






