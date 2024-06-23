import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)

data = pd.read_csv('transactions.csv')

def preprocess_data(data):
    data = pd.get_dummies(data)
    data.fillna(data.mean(), inplace=True)
    return data

def feature_engineering(data):
    features = data.drop('isFraud', axis=1)
    labels = data['isFraud']
    return features, labels

data = preprocess_data(data)
X, y = feature_engineering(data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    transaction = request.json
    transaction_df = pd.DataFrame([transaction])
    transaction_df = preprocess_data(transaction_df)
    prediction = model.predict(transaction_df)
    result = 'Fraud' if prediction[0] == 1 else 'Not Fraud'
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
