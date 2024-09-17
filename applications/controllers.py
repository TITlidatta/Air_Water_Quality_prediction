from flask import Flask, request,render_template
from flask import jsonify
from flask import current_app as app
from applications.ML import predict_Air,predict_Water
import pandas as pd
import requests
import json

def setup_routes(app,sgd,ada,ss):
    @app.route('/Awq',methods=["POST","GET"])
    def Awq():
        return render_template('home.html')
    @app.route('/direct/<city_name>',methods=["GET"])
    def direct(city_name):
        city=city_name
        url="http://api.waqi.info/feed/"+city+"/?token=4972207b139ecd458711f43a0edbdeaa9347ed23"
        ans = requests.get(url).json()
        Y_pred=ans['data']['aqi']
        print(Y_pred)
        if Y_pred >=0 and Y_pred<=50 :
            return jsonify('Good Air')
        elif Y_pred >= 51 and Y_pred <= 100:
            return jsonify('Moderate Air ')
        elif Y_pred >= 101 and Y_pred <= 150:
            return jsonify('Unhealthy Air for sensitive people')
        elif Y_pred >= 151 and Y_pred <= 200:
            return jsonify('Unhealthy Air')
        elif Y_pred >= 201 and Y_pred <= 300:
            return jsonify('Very Unhealthy Air')
        elif Y_pred >= 301 and Y_pred <= 500:
            return jsonify('Hazardous Air')
        else:
            return jsonify('Severe Air Conditions')
    @app.route('/man',methods=["POST"])
    def man():
        values= request.get_json()
        all=[]
        air=[]
        water=[]
        #print(values)
        for val in values['allValues']:
            all.append(float(val))
        for i in range(0,11):
            air.append(all[i])
        for i in range(11,31):
            water.append(all[i])
        s,y=predict_Air(pd.DataFrame([air]),sgd,ss)
        print(y)
        s1=predict_Water(pd.DataFrame([water]),ada)
        return jsonify({'Status':s+' '+s1})