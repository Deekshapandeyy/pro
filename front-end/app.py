from flask import Flask,redirect,url_for,render_template,request
from fileinput import filename
from machine import *
app = Flask(__name__, template_folder='template',static_folder='static')
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/success', methods=['POST'])
def success():
    if request.method == 'POST':  
        remove_files()
        f = request.files['file']
        f.save(f.filename)  
        result=predict_output(f.filename,"model.sav")
        return render_template("Output.html", name = result[0]) 


if __name__=='__main__':
    app.run(debug=True)
