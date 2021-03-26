# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 13:07:55 2021
@author: Paleti Krishnasai
"""
import numpy as np
import pandas as pd
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

def diabetes_risk_prediction(glucose, bp, skinthickness, insulin, bmi, age):
    indicator_list = [glucose, bp, skinthickness, insulin, bmi, age]
    predictions = log_model.predict_proba(np.array(indicator_list).reshape(1, -1))
    risk = round(float(predictions[0,1]),2)
    print("risk=",risk)
    print(type(risk))
    out=""
    #out = out+ '"-"*len("Health Indicator Analysis"))'+'+'
    print("Health Indicator Analysis")
    print("-"*len("Health Indicator Analysis"))
    if risk <= 0.30:
        out=out+"You are probably in good health, keep it up."+"+"
    elif risk > 0.30 and risk < 0.70:
        out=out+"Your Health is in an acceptable state, But you can do better!."+"+"
    elif risk >= 0.90:
        out=out+"Go to a hospital. Odds are high you have diabetes."+"+"
    elif risk >= 0.70:
        out=out+"See a doctor as soon as you can . You might be on the way to developing diabetes if you don't change your lifestyle."+"+"

    return out+"Your Diabetes Risk Index is {:.2f}/50.".format(risk*0.5*100)

def diabetes_risk_prediction2(glucose, bp, skinthickness, insulin, bmi, age):
    weight = {
    1:1.05,
    0:1}

    log_model2 = LogisticRegression(class_weight = weight)
    log_model2.fit(dataset[features], dataset['Outcome'])

    indicator_list = [glucose, bp, skinthickness, insulin, bmi, age]
    predictions = log_model2.predict_proba(np.array(indicator_list).reshape(1, -1))
    risk = predictions[0,1]
    return risk

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/y_predict',methods=['POST'])
def y_predict():
    '''

    For rendering results on HTML GUI
    '''
    # projectpath = request.form['projectFilepath']
    '''
    x_test = [int(x) for x in request.form.values()]
    print(x_test)
    output=diabetes_risk_prediction(x_test)
    print(output)
    '''
    glucose=int(request.form['Glucose'])
    bp=int(request.form['BloodPressure'])
    skinthickness=int(request.form['Skin Thickness'])
    insulin=int(request.form['Insulin'])
    bmi=int(request.form['BMI'])
    age=int(request.form['Age'])
    output=diabetes_risk_prediction(glucose, bp, skinthickness, insulin, bmi, age)
    out= output.split("+")
    print('output = ',out)

    return render_template('index1.html',output=out)

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    #For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.y_predict([np.array(list(data.values()))])

    #output = prediction[0]
    # return jsonify(output)


if __name__ == "__main__":
    dataset = pd.read_csv('diabetes.csv')
    cols = dataset.columns.tolist()
    dataset = dataset.drop_duplicates(keep='first')
    dataset = dataset.drop(dataset[dataset['BMI']==0].index)
    dataset = dataset.drop(dataset[dataset['BloodPressure']==0].index)
    dataset = dataset.drop(dataset[dataset['Insulin']==0].index)
    dataset = dataset.drop(dataset[dataset['Glucose']==0].index)
    dataset = dataset.drop(dataset[dataset['SkinThickness']==0].index)
    features = cols.copy()
    features.remove('Outcome')
    features.remove('DiabetesPedigreeFunction')
    features.remove('Pregnancies')
    # Instantiate the model
    weight = {
    1:1.05,
    0:1
    }
    log = LogisticRegression(class_weight = weight)

    kf = KFold(n_splits=6)
    score = cross_val_score(log, dataset[features], dataset['Outcome'], cv=kf, scoring='accuracy')
    print(score)
    print("The mean accuracy is:",score.mean())
    weight = {
    1:1.05,
    0:1
    }

    log_model = LogisticRegression(class_weight = weight)
    log_model.fit(dataset[features], dataset['Outcome'])
    log_model.predict_proba(np.array(dataset[features].iloc[0]).reshape(1, -1))

    app.run(debug=True)
