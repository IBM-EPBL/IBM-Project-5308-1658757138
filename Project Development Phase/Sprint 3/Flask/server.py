import base64
import os
import re
import sys
from io import BytesIO

import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify
from imageio.v2 import imread
import cv2
import model
sys.path.append(os.path.abspath("./model"))

app = Flask(__name__)
global model, graph
model, graph = model.init()


@app.route('/')
def index():
    return render_template("default.html")


@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    # get data from drawing canvas and save as image
    parseImage(request.get_data())

    # read parsed image back in 8-bit, black and white mode (L)
    x = imread('output.png', mode='L')
    x = np.invert(x)
    x = cv2.resize(x, (28, 28))

    # reshape image data for use in neural network
    x = x.reshape(1, 28, 28, 1)
    with graph.as_default():
        out = model.predict(x)
        print(out)
        print(np.argmax(out, axis=1))
        response = np.array_str(np.argmax(out, axis=1))
        return response


@app.route("/process", methods=["GET", "POST"])
def process():
    data_url = str(request.get_data())
    offset = data_url.index(',')+1
    img_bytes = base64.b64decode(data_url[offset:])
    image_bytes = BytesIO(img_bytes)
    image_bytes.seek(0)
    img = Image.open(image_bytes)
    img = img.convert('L')
    img = img.resize((28, 28))
    # img.save(r'templates\image.png')
    img = np.array(img)
    img = img.reshape((1, 28, 28, 1))
    with graph.as_default():
        out = model.predict(img)
        print(out)
        print(np.argmax(out, axis=1))
        response = np.array_str(np.argmax(out, axis=1))
        return response

def parseImage(imgData):
    # parse canvas bytes and save as output.png
    imgstr = re.search(b'base64,(.*)', imgData).group(1)
    with open('output.png', 'wb') as output:
        output.write(base64.decodebytes(imgstr))


if __name__ == '__main__':
    app.debug = True
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)