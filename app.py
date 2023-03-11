from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import cv
#import numpy as np

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(_name_)
app.config['MAX_CONTENT_LENGTH'] = 2*1024*1024
'

# Load your trained model
model = load_model('inceptionv3.h5')
model._make_predict_function()
print('Model loaded. Check http://127.0.0.1:5000/')

def save_video_frames(video_path):
    count = 0
    cap = cv2.VideoCapture(video_path)   # capturing the video from the given path
    frameRate = cap.get(5) #frame rate
    x=1
    names = []
    while(cap.isOpened()):

      frameId = cap.get(1) #current frame number
      ret, frame = cap.read()

      if (ret != True):
          break
      if (frameId % math.floor(frameRate) == 0):
          # storing the frames in a new folder named train_1
          
          filename ='./train' + video_path.replace(SOURCE,'') +"_frame%d.jpg" % count;count+=1
          cv2.imwrite(filename, frame)
    cap.release()

def predict(frame):
    # Load the pre-trained Inception v3 model
    model = tf.keras.applications.InceptionV3(weights='imagenet')
    # pred_df = pd.DataFrame(columns=['image','class'])
    # Load the image
    img = cv2.imread(frame)

    # Preprocess the image
    img = cv2.resize(img, (299, 299))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Run the image through the model to get predictions
    predictions = model.spredict(img)

    # Decode the predictions
    decoded_predictions = tf.keras.applications.imagenet_utils.decode_predictions(predictions)

    preds = []
     # Print the top 5 predictions
    for i in range(5):
        # print(decoded_predictions[0][i][1])
        preds.append({'Class':decoded_predictions[0][i][1],'Source':frame})
        print(decoded_predictions[0][i][1])
        # print((decoded_predictions[0][i][1],frame))


    return preds

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(_file_)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        result = str(pred_class[0][0][1])               # Convert to string
        return result
    return None


if _name_ == '_main_':
    app.run(debug=True)
