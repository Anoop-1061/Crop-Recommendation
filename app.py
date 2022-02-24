from flask import Flask, render_template, request
from pickle import load
import numpy as np

app = Flask(__name__) # Instantiating Flask class.

model = load(open('RF.pkl', 'rb')) # Loading the trained model for prediction.
label = load(open('label.pkl', 'rb')) # Loading the label encoder to inverse transform the predicted class by the model.
scaler = load(open('scaler.pkl', 'rb')) # Loading the scaler for scaling the inputs.

@app.route('/')
def main():
    return render_template('home.html')

@app.route('/predict', methods = ['POST'])
def home():
    # Retrieving the data entered by the user.
    nitrogen = request.form['N']
    phosphorus = request.form['P']
    potassium = request.form['K']
    temperature = request.form['Temp']
    humidity = request.form['Humid']
    pH = request.form['pH']
    rainfall = request.form['Rain']
    # Converting the values entered by the user into NumPy array to feed the model.
    data = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, pH, rainfall]]) 
    predicted = model.predict(scaler.transform(data)) # Prediction.
    predicted_class = label.inverse_transform(predicted)[0] # Inverse transforming the predicted class.
    return render_template('result.html', data = predicted_class)

if __name__ == '__main__':
    app.run(debug = True)
    
