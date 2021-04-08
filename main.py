import tensorflow as tf
from tensorflow import keras
import numpy as np
from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import base64

app = Flask(__name__)

# Load prebuilt model
model = keras.models.load_model('app/my_model3.h5')

# Handle GET request
@app.route('/', methods=['GET'])
def drawing():
    return render_template('canvas.html')

# Handle POST request
@app.route('/', methods=['POST'])
def canvas():
    # Recieve base64 data from the user form
    canvasdata = request.form['canvasimg']
    encoded_data = request.form['canvasimg'].split(',')[1]

    # Decode base64 image to python array
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Convert 3 channel image (RGB) to 1 channel image (GRAY)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize to (28, 28)
    gray_image = cv2.resize(gray_image, (28, 28), interpolation=cv2.INTER_LINEAR)

    # Expand to numpy array dimenstion to (1, 28, 28)
    print(img.shape)
    img = np.expand_dims(gray_image, axis=0)
    img = img/256
    print(gray_image.shape)

    try:
        prediction = np.argmax(model.predict(img))
        print(f"Prediction Result : {str(prediction)}")
        return render_template('canvas.html', response=str(prediction), canvasdata=canvasdata, success=True)
    except Exception as e:
        return render_template('canvas.html', response=str(e), canvasdata=canvasdata)


if __name__ == '__main__':
    app.run(debug=True)