from flask import Flask, request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
print("!!!! Flask Read !!!!")

pickle_in = open('classifier.pkl', 'rb')
classifer = pickle.load(pickle_in)

print(classifer.predict([[0,1,2,1]]))

@app.route("/")
def Welcome():
    return "Welcome to my first Flask Application"


#example: http://127.0.0.1:5000/Process?variance=1&skewness=3&curtosis=-4&entropy=-0.2
@app.route("/Process")
def ProcessData():
    variance = request.args.get("variance")
    skewness = request.args.get("skewness")
    curtosis = request.args.get("curtosis")
    entropy = request.args.get("entropy")
    predictedValue = classifer.predict([[variance,skewness,curtosis,entropy]])
    return "Predicted value is "+ str(predictedValue)

@app.route("/ProcessFile", methods=["POST"])
def ProcessFile():
    try:
        df_test = pd.read_csv(request.files.get("file"))
        predictedValue = classifer.predict(df_test)
        return "Predicted values from the file are "+ str(list(predictedValue))
    except:
        return "There is a exception handling file content !!!"

if __name__== "__main__":
    app.run()