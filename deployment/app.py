# from flask import Flask,render_template,request
# from keras.preprocessing.image import load_img
# from keras.preprocessing.image import img_to_array
# import numpy as np
# import tensorflow as tf
# import cv2
# import imghdr
# import os
# # Load the trained model
# model = tf.keras.models.load_model("svm_classifier.h5")

# def preprocess(image):
#     img=cv2.imread(image)
#     img=cv2.resize(img,(256,256))
#     img=img/255.0
#     return img
# app=Flask(__name__)
# @app.route('/',methods=['GET'])
# def hello_word():
#     return render_template('index.html')

# @app.route('/',methods=['POST'])
# def predict():
#     imagefile=request.files['imagefile']
#     image_path="./images/"+imagefile.filename
#     imagefile.save(image_path)
#     image=load_img(image_path,target_size=(224,224))
#     image=img_to_array(image)
#     image=image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))
#     image=preprocess(image)
#     yhat=model.predict(image)
#     label=label[0][0]
#     classification='%s(%.2f%%)'%(label[1],label[2]*100)
#     return render_template('index.html',predicition=classification)

# if __name__ == '__main__':
#     app.run(port=3000,debug=True)
from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from PIL import Image
import io

# Load the trained model
model = load_model("model_with_dropout.h5")

app = Flask(__name__)

# Define a function to preprocess the uploaded image
def preprocess_image(image_file):
    # Convert the file storage object to a PIL image
    img = Image.open(io.BytesIO(image_file.read()))
    img = img.resize((224, 224))  # Resize the image
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array[:, :, :, :3]  # Ensure only 3 channels are included
    return img_array

# Define a function to make predictions
def predict_fracture(image_file):
    processed_image = preprocess_image(image_file)
    prediction = model.predict(processed_image)
    if prediction[0][0] > 0.5:
        return "Fractured"
    else:
        return "Non-fractured"

# Define a route to handle the homepage
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get the uploaded image file
        uploaded_file = request.files["file"]
        if uploaded_file.filename != "":
            # Make a prediction
            result = predict_fracture(uploaded_file)
            return render_template("result.html", result=result)
    return render_template("index.html")

if __name__ == "__main__":
 app.run(port=3000,debug=True)
