from flask import Flask, render_template, request
import model
import json
app = Flask(__name__)

@app.route("/", methods = ['POST'])
def get_predictions():
    query = request.form['query']
    return model.predict(query)

@app.route("/app", methods =['GET', 'POST'])
def frontend():
    if request.method == "GET":
        return render_template("query.html")
    if request.method == "POST":
        query = request.form['query']
        prediction =  model.predict(query)
        return render_template('answer.html' ,prediction=prediction)
    