from flask import Flask, jsonify, request
from keras.models import load_model
from keras.utils import load_img
#from keras.utils.image import img_to_array
import numpy as np
from flask_cors import CORS
#import cv2

app = Flask(__name__)
'''
CORS(app, resources={r"/*": {"origins": ["http://localhost",
                                          "http://localhost:8080",
                                          "https://myapp.herokuapp.com",
                                          "http://www.e-hospital.ca/Skincancer",
                                          "http://www.e-hospital.ca",
                                          "http://localhost:5000"]}})

'''
model = load_model('my_model.hdf5')

model.make_predict_function()

def predict_label(img_path):
	i = load_img(img_path, target_size=(244,244))
	
	i = i.resize((244,244))
	
	i = np.expand_dims(i, axis=0)
	
	p=model.predict(i) 
	classes_x=np.argmax(p,axis=1)
	if p[0][0] < 0:
		return "Not Cancer"
	else:
		return "Cancer"

@app.route("/", methods=['GET'])
def main():
	return {'message': 'Welcome to our Project!'}


@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the POST request has a file part.
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    # If the user does not select a file, the browser submits an empty file.
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Save the file to disk.
    file.save(file.filename)
    p = predict_label(file.filename)

    return jsonify({'message': p}), 200


if __name__ == '__main__':
    app.run(threaded=True, port=5000)
