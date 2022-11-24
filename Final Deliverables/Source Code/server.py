import base64
import os
import re
import sys
from io import BytesIO
import random
import numpy as np
from PIL import Image , ImageFilter
from flask import Flask, render_template, request, jsonify
from imageio.v2 import imread
import cv2
import model
sys.path.append(os.path.abspath("./model"))

app = Flask(__name__)
global model, graph

import tensorflow as tf
from tensorflow import keras
import keras.models
import cv2
from keras import layers

@app.route('/')
def index():
    return render_template("default.html")


@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    # get data from drawing canvas and save as image
    parseImage(request.get_data())

    # read parsed image back in 8-bit, black and white mode (L)
    x = imread('output.jpg', mode='L')
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
    tf.compat.v1.disable_eager_execution()
    num_classes = 10
    img_rows, img_cols = 28, 28
    input_shape = (img_rows, img_cols, 1)
    # model = Sequential()

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.load_weights("Detector.h5")
    print("Loaded Model from disk")
    print(model.summary())

    # compile and evaluate loaded model
    model.compile(loss="categorical_crossentropy",optimizer="Adam",metrics=["accuracy"]) #(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])


    graph = tf.compat.v1.get_default_graph()



    data_url = str(request.get_data())
    offset = data_url.index(',')+1
    img_bytes = base64.b64decode(data_url[offset:])
    image_bytes = BytesIO(img_bytes)
    image_bytes.seek(0)
    img = Image.open(image_bytes)

    pil_image = img.convert('RGB')
    open_cv_image = np.array(pil_image)
    # Convert RGB to BGR
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    result = open_cv_image.copy()

    kernel = np.ones((5, 5), np.uint8)
    open_cv_image = cv2.dilate(open_cv_image, kernel, iterations=3)
    open_cv_image = cv2.GaussianBlur(open_cv_image, (7, 7), 0)

    imgray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 5, 255, cv2.THRESH_BINARY)
    contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    colour = (255, 0, 0)
    thickness = 1
    i = 0

    Rectangles = []

    for cntr in contours:
        x1, y1, w, h =  cv2.boundingRect(cntr)

        if w>h:
            h = w
        else:
            w = h

        Rectangles += [ ( x1-int(w/2) , y1 , x1+w , y1+h ) ]
        x1, y1, w, h = Rectangles[-1]
        x2 = w
        y2 = h
        cv2.rectangle(result, (x1, y1), (x2, y2), colour, thickness)
        print("Object:", i + 1, "x1:", x1, "x2:", x2, "y1:", y1, "y2:", y2)
        i += 1

    Rectangles = sorted(Rectangles , key=lambda x : x[0])


    img = img.filter(ImageFilter.GaussianBlur(5))

    with graph.as_default():
        total = ""

        for Rect in Rectangles:

            subimg = img.crop( Rect )
            #subimg.show()
            subimg = subimg.convert('L')
            subimg = subimg.resize((28, 28))
            print("Type : ", type(subimg))
            subimg = np.array(subimg)
            subimg = subimg.reshape(1, 28, 28, 1)


            out = model.predict( subimg )
            print(np.argmax(out, axis=1))
            total  += np.array_str(np.argmax(out, axis=1))

        total = total.replace( "[" , "" )
        total = total.replace( "]" , " ")
        response = {
            'data': total,
            'probencoded': str("probencoded"),
            'interpretencoded': str("interpretencoded"),
        }
        return jsonify(response)



def parseImage(imgData):
    # parse canvas bytes and save as output.png
    imgstr = re.search(b'base64,(.*)', imgData).group(1)
    with open('output.jpg', 'wb') as output:
        output.write(base64.decodebytes(imgstr))


if __name__ == '__main__':
    app.debug = True
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)