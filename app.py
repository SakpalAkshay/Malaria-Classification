import os
from flask import Flask, render_template, request
from resnet.resnet50 import prediction
import numpy as np
from keras import backend as K

app = Flask(__name__)

UPLOAD_FOLDER = os.path.basename('static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/result', methods = ['POST'])
def result():
    if request.method == 'POST':
        file = request.files['image']
        f = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(f)

    K.clear_session()
    result = prediction(f)
    print(result)
    result = np.argmax(result[0], axis = 0)
    data = file.filename
    print(result)

    return render_template("result.html", data = data, result = result)

if __name__ == '__main__':
    app.run(debug = True)