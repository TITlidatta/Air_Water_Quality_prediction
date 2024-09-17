# app.py

import os
from flask import Flask,current_app
from applications.controllers import setup_routes 
from applications.ML import model_Air,model_Water

app = None

def create_app():
    app = Flask(__name__, template_folder="templates",static_folder='static')
    if os.getenv('ENV', "development") == "production":
        raise Exception("it's not for production")
    else:
        print("starting local development")
         # Initialize the database within the app context
    sgd,ss = model_Air()
    ada=model_Water()
    setup_routes(app,sgd,ada,ss)
    return app

app = current_app or create_app()
app.app_context().push()

if __name__ == '__main__':
    app.run(debug=True)