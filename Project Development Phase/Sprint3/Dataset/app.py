import numpy as np
import keras.models
from keras.models import model_from_json
import os

from flask import Flask, render_template,request
from werkzeug.utils import secure_filename

from load import *
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
#initalize our flask app
app = Flask(__name__)
#global vars for easy reusability
global model, graph
#initialize these variables
model, graph = init()

def model_predict(image_path,model):

    Img=image.load_img(image_path,target_size=(224,224))
    Img = np.expand_dims(Img, axis=0)

    # Be careful how your trained model deals with the input
     # otherwise, it won't make correct prediction!
    Img = preprocess_input(Img, mode='caffe')
    with graph.as_default():
        preds = model.predict(Img)
    return preds

@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
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
if __name__ == '__main__':
    app.run(port=5002, debug=True)



